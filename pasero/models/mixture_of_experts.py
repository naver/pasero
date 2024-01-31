# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import regex
import itertools
import logging
import torch
import numpy as np
import torch.nn as nn
from typing import Optional
from torch import Tensor, BoolTensor, LongTensor
from .transformer import DummyEncoder, Transformer, TransformerEncoder, TransformerDecoder
from .transformer import TransformerEncoderLayer, TransformerDecoderLayer
from . import modules
from .modules import Embedding, Expert
from pasero import utils
from pasero.config import register_model, MOETransformerConfig, DistributedConfig


logger = logging.getLogger('mixture_of_experts')


def get_moe_cls(moe_impl: str):
    if moe_impl == 'fused':
        return FusedMixtureOfExperts
    elif moe_impl == 'tutel':
        return TutelMixtureOfExperts
    else:
        return BasicMixtureOfExperts


class Top2Gate(nn.Linear):
    def __init__(self, embed_dim: int, expert_count: int, use_fp32=False):
        super().__init__(embed_dim, expert_count, bias=False)
        self.use_fp32 = use_fp32

    def forward(
        self,
        input: torch.Tensor,  # SxD
        input_mask: Optional[torch.Tensor] = None,  # S (=B*T)
    ) -> tuple[Tensor, Tensor, Tensor, dict]:
        # S: number of tokens (batch_size * sequence length)
        # E: expert count
        # D: expert dim
        logits = super().forward(input)
        if self.use_fp32:
            logits = logits.float()
        gate_value = logits.softmax(dim=-1)  # SxE

        nt, ec = gate_value.shape
        input_mask = input_mask.unsqueeze(1)

        # zero out gate values for padding tokens
        gate_value = gate_value.masked_fill(input_mask, 0)

        # compute expert weights:
        mask = torch.ones_like(gate_value, dtype=torch.bool)
        topk_indices = gate_value.topk(2).indices
        top1 = topk_indices[:,0]
        top2 = topk_indices[:,1]
        range = torch.arange(nt, device=mask.device)
        mask[range, top1] = 0
        top1_mask = mask.clone()
        mask[range, top2] = 0
        mask.unsqueeze_(-1)
        gate_weights = gate_value.unsqueeze(-1).masked_fill(mask, 0)
        sum = gate_weights.sum(dim=1, keepdim=True)
        sum[sum == 0] = 1   # to avoid dividing by zero for padding tokens
        gate_weights = gate_weights / sum  # normalized weights (that sum to 1)

        # compute the load balancing loss:
        nt_real = (~input_mask).sum()  # number of non-padding tokens
        top1_mask.masked_fill_(input_mask.reshape(-1, 1), 1)  # exclude padding tokens
        lb_loss = ec * (((~top1_mask).sum(0) / nt_real) * (gate_value.sum(0) / nt_real)).sum()
        return lb_loss, gate_weights.to(input.dtype), gate_value, {}


class FusedMixtureOfExperts(nn.Module):
    is_sharded: bool = False

    def __init__(
        self,
        embed_dim: int,
        expert_dim: int,
        local_expert_count: int,
        global_expert_count: int,
        activation_fn: str = 'relu',
        has_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert global_expert_count >= 2
        assert local_expert_count == global_expert_count
        assert activation_fn not in ('swiglu', 'geglu')
        self.embed_dim = embed_dim
        self.expert_dim = expert_dim
        self.expert_count = global_expert_count
        self.gate = Top2Gate(embed_dim, self.expert_count, use_fp32=True)
        self.fc1 = modules.Linear(embed_dim, self.expert_count * expert_dim, has_bias=has_bias)
        self.fc2 = modules.Linear(expert_dim, self.expert_count * embed_dim, has_bias)
        self.activation_fn = modules.get_activation_fn(activation_fn)

    @utils.disable_logging()
    def forward(self, x: Tensor, padding_mask: BoolTensor) -> tuple[Tensor, Tensor]:
        """
        Shape:
            x: (B, T, D)
            padding_mask: (B, T)
        """
        bsz, seq_len, _ = x.shape
        ec = self.expert_count
        x = x.view(bsz * seq_len, -1)  # BxTxD -> SxD
        padding_mask = padding_mask.view(bsz * seq_len)  # BxT -> S
        
        # compute gate outputs:
        self.lb_loss, gate_weights, gate_value, metadata = self.gate(x, padding_mask)

        # compute expert outputs:
        x = self.fc1(x)   # (BxT)x(ExD)
        x = self.activation_fn(x)
        x = x.view(-1, ec, self.expert_dim)
        w = self.fc2.weight.view(ec, self.embed_dim, self.expert_dim)
        if self.fc2.bias is not None:
            b = self.fc2.bias.view(ec, self.embed_dim)
        x = torch.einsum('beh,edh->bed', x, w) + b  # (BxT)xExD
        # multiply the expert outputs by their weights:
        x = (x * gate_weights).sum(dim=1)
        
        x = x.view(bsz, seq_len, -1)
        gate_value = gate_value.view(bsz, seq_len, -1)  # SxE -> BxTxE
        return x, gate_value, metadata

    @staticmethod
    def update_state_dict(state_dict: dict[str, Tensor]) -> None:
        # convert from NLLB-200/basic format
        # encoder.layers.2.moe_layer.experts.EXPERT_ID.fc1.weight * EXPERT_COUNT ->
        # encoder.layers.2.moe_layer.fc1.weight (concatenated at the first dimension)
        stack_expert_params(state_dict, concat=True)


class BasicMixtureOfExperts(nn.Module):
    is_sharded: bool = False

    def __init__(
        self,
        embed_dim: int,
        expert_dim: int,
        local_expert_count: int,
        global_expert_count: int,
        activation_fn: str = 'relu',
        has_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert global_expert_count >= 2
        assert local_expert_count == global_expert_count
        self.embed_dim = embed_dim
        self.expert_dim = expert_dim
        self.expert_count = global_expert_count
        self.gate = Top2Gate(embed_dim, self.expert_count, use_fp32=True)
        self.experts = nn.ModuleList([
            Expert(embed_dim, expert_dim, activation_fn=activation_fn, has_bias=has_bias)
            for _ in range(self.expert_count)
        ])

    @utils.disable_logging()
    def forward(self, x: Tensor, padding_mask: BoolTensor) -> tuple[Tensor, Tensor]:
        """
        Shape:
            x: (B, T, D)
            padding_mask: (B, T)
        """
        bsz, seq_len, _ = x.shape
        x = x.view(bsz * seq_len, -1)  # BxTxD -> SxD
        padding_mask = padding_mask.view(bsz * seq_len)  # BxT -> S
        
        # compute gate outputs:
        self.lb_loss, gate_weights, gate_value, metadata = self.gate(x, padding_mask)

        # compute expert outputs:
        expert_out = 0
        for expert_id, expert in enumerate(self.experts):
            weight = gate_weights[:, expert_id]
            if weight.sum() > 0:  # skip this expert if it unused
                expert_out += weight * expert(x)
        x = expert_out
        
        x = x.view(bsz, seq_len, -1)
        gate_value = gate_value.view(bsz, seq_len, -1)  # SxE -> BxTxE
        return x, gate_value, metadata

    @staticmethod
    def update_state_dict(state_dict: dict[str, Tensor]) -> None:
        # convert from "fused" or "tutel" format
        # encoder.layers.2.moe_layer.fc1.weight (concatenated or stacked at the first dimension) ->
        # encoder.layers.2.moe_layer.experts.EXPERT_ID.fc1.weight * EXPERT_COUNT
        TutelMixtureOfExperts.convert_to_fused_format(state_dict)
        unstack_expert_params(state_dict)


class TutelMixtureOfExperts(nn.Module):
    is_sharded: bool = True

    def __init__(
        self,
        embed_dim: int,
        expert_dim: int,
        local_expert_count: int,
        global_expert_count: int,
        capacity_factor: Optional[int] = None,
        activation_fn: str = 'relu',
        has_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert local_expert_count >= 2 and global_expert_count % local_expert_count == 0
        assert activation_fn not in ('swiglu', 'geglu')
        self.embed_dim = embed_dim
        self.expert_dim = expert_dim
        self.local_expert_count = local_expert_count
        self.global_expert_count = global_expert_count
        # capacity_factor > 0 freezes without this option:
        self.inequivalent_tokens = capacity_factor and global_expert_count > local_expert_count

        import tutel.moe
        self.tutel_moe_layer = tutel.moe.moe_layer(
            gate_type={'type': 'top', 'k': 2, 'capacity_factor': capacity_factor, 'fp32_gate': False},
            # FIXME: setting 'fp32_gate' to True requires --find-unused-parameters, why?
            model_dim=self.embed_dim,
            experts={
                'count_per_node': self.local_expert_count,
                'type': 'ffn',
                'hidden_size_per_expert': self.expert_dim,
                'activation_fn': modules.get_activation_fn(activation_fn),
                'has_fc1_bias': has_bias,
                'has_fc2_bias': has_bias,
            },
            scan_expert_func=(lambda name, param: setattr(param, '_is_sharded', True)),  # used by DDP to avoid
            # syncing expert parameters
        ) if local_expert_count else None

    @utils.disable_logging()
    def forward(self, x: Tensor, padding_mask: BoolTensor) -> tuple[Tensor, Tensor]:
        """
        Shape:
            x: (B, T, D)
            padding_mask: (B, T)
        """
        # T: sequence length
        # B: batch size
        # E: expert count
        # D: expert dim
        gate = self.tutel_moe_layer.gates[0]
        gate_value = gate(x).softmax(dim=-1).float()  # BxTxE
        # zero out gate values for padding tokens
        gate_value = gate_value.masked_fill(padding_mask.unsqueeze(2), 0)
        x = self.tutel_moe_layer(x, inequivalent_tokens=self.inequivalent_tokens)
        self.lb_loss = self.tutel_moe_layer.l_aux

        return x, gate_value, {}

    @staticmethod
    def update_state_dict(state_dict: dict[str, Tensor]) -> None:
        # convert from NLLB-200/basic format
        # encoder.layers.2.moe_layer.experts.EXPERT_ID.fc1.weight x EXPERT_COUNT ->
        # encoder.layers.2.moe_layer.fc1.weight (stacked at the first dimension)
        stack_expert_params(state_dict, concat=False)

        for name in list(state_dict):
            new_name = (
                name
                .replace('.moe_layer.fc1.weight', '.moe_layer.tutel_moe_layer.experts.batched_fc1_w')
                .replace('.moe_layer.fc2.weight', '.moe_layer.tutel_moe_layer.experts.batched_fc2_w')
                .replace('.moe_layer.fc1.bias', '.moe_layer.tutel_moe_layer.experts.batched_fc1_bias')
                .replace('.moe_layer.fc2.bias', '.moe_layer.tutel_moe_layer.experts.batched_fc2_bias')
                .replace('.moe_layer.gate.weight', '.moe_layer.tutel_moe_layer.gates.0.wg.weight')
            )
            
            value = state_dict.pop(name)
            if name.endswith('.moe_layer.fc2.weight'):
                value = value.transpose(1, 2)

            state_dict[new_name] = value

        for name in list(state_dict):
            if name.endswith('.moe_layer.tutel_moe_layer.gates.0.wg.weight'):
                prefix = name.removesuffix('.gates.0.wg.weight')
                value = state_dict[name]
                expert_count = value.size(0)
                # This is required in recent tutel versions:
                state_dict[f'{prefix}._num_global_experts'] = torch.tensor(expert_count)

    @staticmethod
    def convert_to_fused_format(state_dict: dict[str, Tensor]) -> None:
        for name in list(state_dict):
            new_name = (
                name
                .replace('.moe_layer.tutel_moe_layer.experts.batched_fc1_w', '.moe_layer.fc1.weight')
                .replace('.moe_layer.tutel_moe_layer.experts.batched_fc2_w', '.moe_layer.fc2.weight')
                .replace('.moe_layer.tutel_moe_layer.experts.batched_fc1_bias', '.moe_layer.fc1.bias')
                .replace('.moe_layer.tutel_moe_layer.experts.batched_fc2_bias', '.moe_layer.fc2.bias')
                .replace('.moe_layer.tutel_moe_layer.gates.0.wg.weight', '.moe_layer.gate.weight')
            )
            
            value = state_dict.pop(name)
            if name.endswith('.moe_layer.tutel_moe_layer.experts.batched_fc2_w'):
                value = value.transpose(1, 2)

            if not name.endswith('._num_global_experts'):
                state_dict[new_name] = value


def stack_expert_params(state_dict: dict[str, Tensor], concat: bool = False) -> None:
    """
    Modify `state_dict` in place to replace expert params in the "basic" format (with one weight per expert) with 
    params in the "fused" format (concatenated expert weights) or "stacked" format (stacked expert weights).
    If `state_dict` is already in the correct format, it won't be modified.
    
    encoder.layers.2.moe_layer.experts.EXPERT_ID.fc1.weight * EXPERT_COUNT ->
    encoder.layers.2.moe_layer.fc1.weight (stacked or concatenated at the first dimension)

    Args:
        - state_dict: model parameters to convert, can contain both dense parameters (which will be left untouched)
            and expert parameters, which will be stacked.
        - concat: whether to concatenate the expert weights (giving the "fused" format). If False, they will be 
            stacked. It is easy to go from stacked to fused, by just reshaping the tensors to flatten their first two
            dimensions.
    """
    # regex that identifies expert params in the basic format
    pattern = (
        r'(?P<prefix>.*\.moe_layer)'
        r'\.experts\.(?P<expert_id>\d+)'
        r'\.(?P<suffix>(fc1|fc2).(weight|bias))'
    )
    # maps "basic" expert param name to our naming convention
    get_new_name = lambda m: m.group('prefix') + '.' + m.group('suffix')
    # find params that are in the "basic" format
    expert_params = [m for m in (regex.fullmatch(pattern, name) for name in state_dict) if m]
    expert_params.sort(key=get_new_name)
    # group params that belong to the same MoE layer
    for new_name, matches in itertools.groupby(expert_params, key=get_new_name):
        matches = list(matches)
        # sort params by expert id
        old_names = [
            m.string for m in sorted(matches, key=lambda m: int(m.group('expert_id')))
        ]
        # stack experts params into a single param with the right name
        values = [state_dict.pop(old_name) for old_name in old_names]
        value = torch.cat(values, dim=0) if concat else torch.stack(values, dim=0)
        state_dict[new_name] = value


def unstack_expert_params(state_dict: dict[str, Tensor], shard_count: int = 1) -> None:
    """
    Modify `state_dict` in place to replace expert params in the "fused" format (concatenated or stacked expert 
    weights) with params in the "basic" format (with one weight per expert).
    If `state_dict` is already in the correct format, it won't be modified. Note that this won't work with the Tutel
    format. `TutelMixtureOfExperts.convert_to_fused_format` needs to be called first.
    
    encoder.layers.2.moe_layer.fc1.weight (stacked or concatenated at the first dimension) ->
    encoder.layers.2.moe_layer.experts.EXPERT_ID.fc1.weight * EXPERT_COUNT

    Args:
        - state_dict: model parameters to convert, can contain both dense parameters (which will be left untouched)
            and expert parameters, which will be unstacked.
        - shard_count: the total number of shards this checkpoint has (i.e., `state_dict` contains `1/shard_count` of 
            all experts)
    """
    # regex that identifies expert params in the "fused" format
    pattern = (
        r'(?P<prefix>.*\.moe_layer)'
        r'\.(?P<suffix>(fc1|fc2).(weight|bias))'
    )
    for name in list(state_dict):
        if (m := regex.fullmatch(pattern, name)):
            prefix = m.group('prefix')
            suffix = m.group('suffix')

            gate_name = f'{prefix}.gate.weight'
            gate_value = state_dict[gate_name]

            # In the "fused" format, there is no indication to the number of experts are in `state_dict`. This could be
            # inferred from the first dimension of the expert weights if we knew the expert inner dim, but we don't have
            # this information. Instead, by knowing the total number of checkpoints (`shard_count`) and the total
            # number of experts (first dimension of the gates), we can guess the local expert count.
            total_expert_count = gate_value.size(0)
            expert_count = total_expert_count // shard_count

            stacked_values = state_dict.pop(name)
            
            # if the expert weights are concatenated and not stacked, reshape them
            if name.endswith('.bias') and stacked_values.dim() == 1:
                stacked_values = stacked_values.view(expert_count, -1)
            elif name.endswith('.weight') and stacked_values.dim() == 2:
                stacked_values = stacked_values.view(expert_count, -1, stacked_values.size(-1))
            
            assert stacked_values.size(0) == expert_count

            for expert_id in range(expert_count):
                expert_value = stacked_values[expert_id]
                new_name = f'{prefix}.experts.{expert_id}.{suffix}'
                state_dict[new_name] = expert_value


class MOETransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, cfg: MOETransformerConfig, dist_cfg: DistributedConfig, layer_id: int):
        super().__init__(cfg, dist_cfg, layer_id)
        assert dist_cfg.tp_size == 1, 'tensor parallelism is not supported for mixtures of expert'
        del self.fc1
        del self.fc2
        del self.fc3
        expert_count = cfg.encoder_expert_count
        expert_count = expert_count.get(layer_id) if isinstance(expert_count, dict) else expert_count
        expert_count = expert_count or 0
        
        moe_cls = get_moe_cls(cfg.moe_impl)

        world_size = dist_cfg.dp_size if moe_cls.is_sharded else 1
        assert expert_count % world_size == 0
        local_expert_count = expert_count // world_size
        assert local_expert_count >= 2, 'there should be at least 2 experts per GPU'
        
        self.moe_layer = moe_cls(
            embed_dim=cfg.embed_dim,
            expert_dim=cfg.encoder_expert_dim or cfg.encoder_ffn_dim,
            local_expert_count=local_expert_count,
            global_expert_count=expert_count,
            capacity_factor=cfg.capacity_factor,
            activation_fn=cfg.activation_fn,
            has_bias=cfg.has_bias,
        )
        self.gate_key = f'{self.name}_gate'

    def ffn(self, x: Tensor, residual: Tensor, padding_mask: BoolTensor) -> Tensor:
        x, gate_value, metadata = self.moe_layer(x, padding_mask)
        if 'capacity' in metadata:
            self.layer_outputs[f'{self.name}_capacity'] = metadata['capacity']
        if self.gate_key in self.return_layers:
            self.layer_outputs[self.gate_key] = gate_value
        return x


class MOETransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, cfg: MOETransformerConfig, dist_cfg: DistributedConfig, layer_id: int):
        super().__init__(cfg, dist_cfg, layer_id)
        assert dist_cfg.tp_size == 1, 'tensor parallelism is not supported for mixtures of expert'
        del self.fc1
        del self.fc2
        del self.fc3
        expert_count = cfg.decoder_expert_count
        expert_count = expert_count.get(layer_id) if isinstance(expert_count, dict) else expert_count
        expert_count = expert_count or 0

        moe_cls = get_moe_cls(cfg.moe_impl)

        world_size = dist_cfg.dp_size if moe_cls.is_sharded else 1
        assert expert_count % world_size == 0
        local_expert_count = expert_count // world_size
        assert local_expert_count >= 2, 'there should be at least 2 experts per GPU'
        
        self.moe_layer = moe_cls(
            embed_dim=cfg.embed_dim,
            expert_dim=cfg.decoder_expert_dim or cfg.decoder_ffn_dim,
            local_expert_count=local_expert_count,
            global_expert_count=expert_count,
            capacity_factor=cfg.capacity_factor,
            activation_fn=cfg.activation_fn,
            has_bias=cfg.has_bias,
        )
        self.gate_key = f'{self.name}_gate'

    def ffn(self, x: Tensor, residual: Tensor, padding_mask: BoolTensor) -> Tensor:
        x, gate_value, metadata = self.moe_layer(x, padding_mask)
        if 'capacity' in metadata:
            self.layer_outputs[f'{self.name}_capacity'] = metadata['capacity']
        if self.gate_key in self.return_layers:
            self.layer_outputs[self.gate_key] = gate_value
        return x


class MOETransformerEncoder(TransformerEncoder):
    cfg: MOETransformerConfig
    
    def build_layer(self, layer_id: int) -> TransformerEncoderLayer:
        if (
            self.cfg.encoder_expert_layer_ids is not None and layer_id in self.cfg.encoder_expert_layer_ids or
            self.cfg.encoder_expert_interval and (layer_id + 1) % self.cfg.encoder_expert_interval == 0
        ):
            layer = MOETransformerEncoderLayer(self.cfg, self.dist_cfg, layer_id)
        else:
            layer = TransformerEncoderLayer(self.cfg, self.dist_cfg, layer_id)
        return modules.checkpoint_wrapper(layer, activate=self.cfg.checkpoint_activations)


class MOETransformerDecoder(TransformerDecoder):
    cfg: MOETransformerConfig
    
    def build_layer(self, layer_id: int) -> TransformerDecoderLayer:
        if (
            self.cfg.decoder_expert_layer_ids is not None and layer_id in self.cfg.decoder_expert_layer_ids or
            self.cfg.decoder_expert_interval and (layer_id + 1) % self.cfg.decoder_expert_interval == 0
        ):
            layer = MOETransformerDecoderLayer(self.cfg, self.dist_cfg, layer_id)
        else:
            layer = TransformerDecoderLayer(self.cfg, self.dist_cfg, layer_id)
        return modules.checkpoint_wrapper(layer, activate=self.cfg.checkpoint_activations)


@register_model('moe_transformer')
class MOETransformer(Transformer):
    """
    There are two implementations of mixture-of-experts Transformer:
    - TutelMixtureOfExperts (activated with --moe-impl tutel), where each GPU receives a portion of all experts
    - FusedMixtureOfExperts, where all GPUs hold a copy of all experts (whose parameters are fused into a single
        tensor)
    
    TutelMixtureOfExperts is usually faster (except on 1 GPU with a small number of experts).

    Note that training with TutelMixtureOfExperts will result in sharded checkpoints (one checkpoint per GPU),
    which won't be directly compatible with other world sizes or with our custom "fused" implementation.
    """
    cfg: MOETransformerConfig

    @property
    def moe_cls(self):
        return get_moe_cls(self.cfg.moe_impl)

    @property
    def shard_count(self):
        return self.dist_cfg.dp_size if self.moe_cls.is_sharded else 1

    @property
    def shard_id(self):
        return self.dist_cfg.distributed_rank if self.moe_cls.is_sharded else 0

    def parallelize(self, devices: list[str]) -> None:
        if len(devices) > 1:
            assert self.moe_cls != TutelMixtureOfExperts, 'Tutel MoEs do not support pipeline parallelism'
        super().parallelize(devices)
    
    def build_encoder(self, embed: Optional[Embedding] = None) -> MOETransformerEncoder:
        if self.cfg.model_type == 'decoder':
            return DummyEncoder()
        else:
            return MOETransformerEncoder(
                self.cfg,
                self.dist_cfg,
                self.task,
                embed=embed,
            )
    
    def build_decoder(self, embed: Optional[Embedding] = None) -> MOETransformerDecoder:
        return MOETransformerDecoder(
            self.cfg,
            self.dist_cfg,
            self.task,
            embed=embed,
        )

    @utils.benchmark('loss')
    def compute_loss(
        self,
        logits: Tensor,
        target: LongTensor,
        layer_outputs: dict,
        *args,
        **kwargs,
    ) -> tuple[Tensor, dict[str, float]]:
        loss, logs = super().compute_loss(logits, target, layer_outputs, *args, **kwargs)
        
        capacity = []
        for key in list(layer_outputs):
            if key.endswith('_capacity'):
                capacity.append(layer_outputs[key])
                layer_outputs.pop(key)
        
        logs['capacity'] = sum(capacity) / max(1, len(capacity))

        if self.cfg.load_balancing:  # load balancing scale factor != 0
            load_balancing_loss = sum(
                layer.moe_layer.lb_loss.float() if hasattr(layer, 'moe_layer') else 0.0
                for layer in self.encoder.layers + self.decoder.layers
            )
            num_tokens = (target != self.padding_idx).sum()
            load_balancing_loss = load_balancing_loss * num_tokens
            loss += self.cfg.load_balancing * load_balancing_loss
            logs['lb_loss'] = load_balancing_loss.item()
        
        return loss, logs

    def update_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        # remap experts to the correct format (e.g., basic format -> tutel format)
        if hasattr(self.moe_cls, 'update_state_dict'):
            self.moe_cls.update_state_dict(state_dict)

        if self.training:
            self_state_dict = self.state_dict()
            for k, v in self_state_dict.items():
                if k not in state_dict and 'moe_layer' in k:
                    logger.warning(f"parameter '{k}' was not found in checkpoint, initializing at random")
                    state_dict[k] = v
                    
        return super().update_state_dict(state_dict)

    def set_ddp_params_and_buffers_to_ignore(self):
        params_to_ignore = getattr(self, '_ddp_params_and_buffers_to_ignore', [])
        self._ddp_params_and_buffers_to_ignore = params_to_ignore
        for name, param in self.named_parameters():
            if getattr(param, '_is_sharded', False):
                params_to_ignore.append(name)

    @classmethod
    def shard_state_dict(
        cls,
        state_dict: dict[str, Tensor],
        shard_id: int,  # which part of this state_dict should go to this rank
        shard_count: int,  # number of parts this state_dict should be split into
        total_shard_count: int,  # total number of checkpoint shards
    ) -> dict:
        state_dict = dict(state_dict)  # avoid modifying state_dict in place

        # First convert state_dict from any format to a "fused"-like format but with stacked expert weigths (i.e.,
        # the first dimension is the expert count)
        TutelMixtureOfExperts.convert_to_fused_format(state_dict)  # from tutel to fused: doesn't do anything if 
        # state_dict is not in Tutel format
        unstack_expert_params(state_dict, shard_count=total_shard_count)  # from fused to basic:
        # doesn't do anything if state_dict is already in basic format
        stack_expert_params(state_dict, concat=False)  # from basic to stacked (like "fused" but with the first 
        # dimension is the expert count, which is flattened with the second dimension in the "fused" format)

        sharded_state_dict = {}
        pattern = r'(?P<prefix>.*\.moe_layer)\.(?P<suffix>(fc1|fc2).(weight|bias))'
        for k, v in state_dict.items():
            if regex.fullmatch(pattern, k):
                expert_count = v.size(0)
                assert expert_count % shard_count == 0
                local_expert_count = expert_count // shard_count
                sharded_state_dict[k] = v[
                    shard_id * local_expert_count:
                    (shard_id + 1) * local_expert_count
                ]
            else:
                sharded_state_dict[k] = v

        return sharded_state_dict

    @classmethod
    def unshard_state_dict(cls, *state_dicts: dict[str, Tensor], total_shard_count: int) -> dict:
        unsharded_state_dict = {}
        pattern = r'(?P<prefix>.*\.moe_layer)\.(?P<suffix>(fc1|fc2).(weight|bias))'

        for state_dict in state_dicts:

            # First convert state_dict from any format to the "fused" format, whose expert weight are easy to merge.
            state_dict = dict(state_dict)  # avoid modifying state_dict in place
            TutelMixtureOfExperts.convert_to_fused_format(state_dict)  # from tutel to fused: doesn't do anything if 
            # state_dict is not in Tutel format
            stack_expert_params(state_dict, concat=True)  # from basic to fused format: doesn't do anything if 
            # state_dict is already in fused format

            for k, v in state_dict.items():
                # fused experts can be simply concatenated
                if k in unsharded_state_dict and regex.fullmatch(pattern, k):
                    unsharded_state_dict[k] = torch.cat([unsharded_state_dict[k], v], dim=0)
                else:
                    unsharded_state_dict[k] = v
        
        # then convert from fused to basic format
        model_shard_count = total_shard_count // len(state_dicts)
        unstack_expert_params(unsharded_state_dict, shard_count=model_shard_count)
        return unsharded_state_dict


def gather_gate_stats(layer_outputs: list[dict]) -> dict[str, np.ndarray]:
    """
    Collect and aggregate gate statistics from layer outputs. Return a dictionary containing gate statistics per
    Transformer layer.

    Args:
        - layer_outputs: layer outputs obtained by passing `return_layers=['enc_{enc_layer_id}_gate', \
            'dec_{dec_layer_id}_gate', ...]`, with the relevant layer ids to `decoding.search`. This is a list, as 
            `decoding.search` returns one dict of layer outputs per hypothesis.

    Returns: 
        - dict mapping layer name and type of statistic to the aggregated statistics of that type for this layer, which
            correspond to a numpy array with one value per expert in that layer.
            For instance: 'enc_0_rank' -> average rank of each expert in the first encoder layer
    """
    gate_stats = {}

    # list[dict] -> dict[list]
    layer_outputs = {k: [output[k] for output in layer_outputs] for k in layer_outputs[0]}

    for gate_key, gate_values in layer_outputs.items():
        m = regex.fullmatch(r'((enc|dec)_\d+)_gate', gate_key)
        if not m:
            continue
        layer_id = m.group(1)

        gate_value = np.concatenate(gate_values, axis=0)   # (TxB)xE
        nt, ec = gate_value.shape

        padding_mask = np.equal(gate_value, 0)[:,:1]  # (TxB)x1

        # get ranking
        c = np.stack([np.ones(ec) * t for t in range(nt)]).astype(np.int64)
        indices = (-gate_value).argsort(axis=-1).astype(np.int64)
        rank = gate_value.copy()
        rank[c, indices] = np.arange(ec).astype(np.float32)
        rank = (rank + 1).astype(np.int64)
        rank *= ~padding_mask
        top1_mask = rank == 1
        top2_mask = rank == 2

        # count the number of non-padding tokens
        num_tokens = (~padding_mask).sum()
        num_tokens = max(1, num_tokens)

        with np.errstate(divide='ignore', invalid='ignore'):
            gate_stats.update({
                # Average gate value of each expert
                f'{layer_id}_mean': gate_value.sum(axis=0) / num_tokens,
                # Average rank (1 to N) of each expert
                f'{layer_id}_rank': rank.sum(axis=0) / num_tokens,
                # Percentage of times each expert is ranked 1st
                f'{layer_id}_top1': top1_mask.sum(axis=0) / num_tokens,
                # Percentage of times each expert is ranked 2nd
                f'{layer_id}_top2': top2_mask.sum(axis=0) / num_tokens,
                # Average gate value when rank is 1
                f'{layer_id}_conf1': (gate_value * top1_mask).sum(axis=0) / top1_mask.sum(axis=0),
                # Average gate value when rank is 2
                f'{layer_id}_conf2': (gate_value * top2_mask).sum(axis=0) / top2_mask.sum(axis=0),
            })
        
    for value in gate_stats.values():
        np.nan_to_num(value, copy=False)

    return gate_stats
