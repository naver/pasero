# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import math
import regex
import torch
import logging
import functools
import contextlib
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional, Union
from torch import Tensor, LongTensor, BoolTensor
from pasero import utils
from collections import defaultdict
from torch.utils.checkpoint import checkpoint


try:
    from flash_attn import flash_attn_func
    assert torch.cuda.get_device_capability()[0] >= 8
except:
    flash_attn_func = None


logger = logging.getLogger('models')


class Identity(nn.Identity):
    def __init__(self, return_tuple=False):
        super().__init__()
        self.return_tuple = return_tuple
    
    """ Version of nn.Identity whose forward function accepts any number of dummy arguments """
    def forward(self, input: Tensor, *args, **kwargs) -> Tensor:
        if self.return_tuple:
            return input, *args
        else:
            return input


_fast_init = False  # global variable used in Linear to skip (or not) random initialization

@contextlib.contextmanager
def fast_init(device=None, dtype=None):
    """
    Speed up model initialization at inference by skipping the random initialization phase & by creating the
    parameters directly on the target device and with the right type
    """
    global _fast_init
    _dtype_old = torch.get_default_dtype()
    _fast_init_old = _fast_init
    _fast_init = True
    if dtype:
        torch.set_default_dtype(dtype)
    if device is not None and hasattr(torch, 'set_default_device'):
        torch.set_default_device(device)
    yield
    # restore old settings
    _fast_init = _fast_init_old
    torch.set_default_dtype(_dtype_old)
    if hasattr(torch, 'set_default_device'):  # torch 2.0
        torch.set_default_device(None)


class Linear(nn.Linear):
    """
    Version of nn.Linear that supports parameter-efficient finetuning with LoRA,
    as well as lightning-fast initialization at inference.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device = None,
        dtype = None,
        lora_rank: int = 0,
        lora_alpha: int = 1,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora = AdapterLayer.LoRA(
            self.in_features,
            lora_rank,
            self.out_features,
            lora_alpha,
        ) if lora_rank else None

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        if self.lora is not None:
            output = output + self.lora(input)
        return output

    def reset_parameters(self) -> None:
        if not _fast_init:
            super().reset_parameters()


class MegatronLMEnter(torch.autograd.Function):
    """
    Implements intra-layer model parallelism as in the "Megatron LM" paper (AKA "tensor parallelism").
    This function should be applied to the input of the FFN and attention blocks.
    
    Also implements sequence-parallelism: https://arxiv.org/abs/2205.05198
    """
    tp_group: dist.ProcessGroup = None
    sequence_parallel: bool = False

    @classmethod
    def forward(cls, ctx, x: Tensor) -> Tensor:
        if cls.sequence_parallel:
            tp_size = dist.get_world_size(group=cls.tp_group)
            shape = (x.shape[0] * tp_size, *x.shape[1:])
            out = torch.zeros(*shape, dtype=x.dtype, device=x.device)
            dist.all_gather_into_tensor(out, x, group=cls.tp_group)
            return out
        else:
            return x
    
    @classmethod
    def backward(cls, ctx, gradient: Tensor) -> Tensor:
        if cls.sequence_parallel:
            tp_size = dist.get_world_size(group=cls.tp_group)
            shape = (gradient.shape[0] // tp_size, *gradient.shape[1:])
            out = torch.zeros(*shape, dtype=gradient.dtype, device=gradient.device)
            dist.reduce_scatter_tensor(out, gradient, group=cls.tp_group)
            return out
        else:
            dist.all_reduce(gradient, group=cls.tp_group)
        return gradient


class MegatronLMExit(torch.autograd.Function):
    """
    Implements intra-layer model parallelism as in the "Megatron LM" paper (AKA "tensor parallelism").
    This function should be applied to the output of the FFN and attention blocks.
    """
    tp_group: dist.ProcessGroup = None
    sequence_parallel: bool = False

    @classmethod
    def forward(cls, ctx, x: Tensor) -> Tensor:
        if cls.sequence_parallel:
            tp_size = dist.get_world_size(group=cls.tp_group)
            shape = (x.shape[0] // tp_size, *x.shape[1:])
            out = torch.zeros(*shape, dtype=x.dtype, device=x.device)
            dist.reduce_scatter_tensor(out, x, group=cls.tp_group)
            return out
        else:
            dist.all_reduce(x, group=cls.tp_group)
            return x
    
    @classmethod
    def backward(cls, ctx, gradient: Tensor) -> Tensor:
        if cls.sequence_parallel:
            tp_size = dist.get_world_size(group=cls.tp_group)
            shape = (gradient.shape[0] * tp_size, *gradient.shape[1:])
            out = torch.zeros(*shape, dtype=gradient.dtype, device=gradient.device)
            dist.all_gather_into_tensor(out, gradient, group=cls.tp_group)
            return out
        else:
            return gradient


def set_sequence_parallel(enable: bool = True):
    MegatronLMEnter.sequence_parallel = enable
    MegatronLMExit.sequence_parallel = enable


def set_tp_group(tp_group: Optional[dist.ProcessGroup] = None):
    MegatronLMEnter.tp_group = tp_group
    MegatronLMExit.tp_group = tp_group


def clamp(x: Tensor) -> Tensor:
    # Copied from the HuggingFace implementation of T5
    # T5 does not run unless applying this after every Transformer block
    if x.dtype == torch.float16:
        clamp_value = torch.where(
            torch.isinf(x).any(),
            torch.finfo(x.dtype).max - 1000,
            torch.finfo(x.dtype).max,
        )
        x = torch.clamp(x, min=-clamp_value, max=clamp_value)
    return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()   # FIXME: is this necessary with bfloat16?
        x = x * torch.rsqrt((x**2).mean(-1, keepdim=True) + self.eps) * self.weight
        return x.to(dtype)


class LayerNormWithoutBias(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias = None


# Those dummy modules are used by FSDP to find which parameters to wrap
class WrappableLinear(Linear):
    pass
class WrappableLayerNorm(nn.LayerNorm):
    pass
class WrappableRMSNorm(RMSNorm):
    pass


def get_activation_fn(activation_fn: str = 'relu'):
    if activation_fn in ('gelu_tanh', 'geglu'):
        return nn.GELU(approximate='tanh')
    elif activation_fn == 'swiglu':
        return nn.SiLU()
    elif activation_fn == 'gelu':
        return nn.GELU(approximate='none')
    else:
        return nn.ReLU()


class Expert(nn.Module):
    def __init__(self, embed_dim: int, expert_dim: int, activation_fn: str = 'relu', has_bias: bool = True) -> None:
        super().__init__()
        self.fc1 = Linear(embed_dim, expert_dim, bias=has_bias)
        self.fc2 = Linear(expert_dim, embed_dim, bias=has_bias)
        self.fc3 = Linear(embed_dim, expert_dim, bias=has_bias) if activation_fn in ('swiglu', 'geglu') else None
        self.activation_fn = get_activation_fn(activation_fn)
    
    def forward(self, x: Tensor) -> Tensor:
        y = self.fc1(x)
        y = self.activation_fn(y)
        if self.fc3 is not None:
            y = y * self.fc3(x)
        x = self.fc2(y)
        return x


class AdapterLayer(nn.Module):
    """
    Adapter module (i.e., layer to be inserted and trained for Parameter-Efficient Fine-Tuning), with the same 
    formulation as in Bapna et al., 2019 (https://arxiv.org/abs/1909.08478)

    This module can also be configured to do Low-Rank Adaptation (aka LoRA), by disabling the activation, residual,
    layer norm and biases.
    """
    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        output_dim: Optional[int] = None,
        zero_init: bool = False,
        layer_norm: bool = True,
        bias: bool = True,
        residual: bool = True,
        activation_fn: str = 'relu',
        scaling: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.projection_dim = projection_dim
        self.zero_init = zero_init
        self.has_layer_norm = layer_norm
        self.bias = bias
        self.residual = residual
        assert not self.residual or self.output_dim == self.input_dim
        if activation_fn is None or activation_fn == 'none':
            self.activation = Identity()
        elif activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'gelu':
            self.activation = nn.GELU(approximate='tanh')
        else:
            raise NotImplementedError
        self.down = None
        self.scaling = scaling
        self._init()

    @classmethod
    def LoRA(cls, input_dim: int, projection_dim: int, output_dim: int, lora_alpha: int) -> 'AdapterLayer':
        # Low-Rank Adaptation as described in https://arxiv.org/abs/2106.09685
        return cls(
            input_dim,
            projection_dim,
            output_dim,
            layer_norm=False,
            bias=False,
            residual=False,
            activation_fn=None,
            zero_init=True,
            scaling=lora_alpha / projection_dim,
        )

    def _init(self) -> None:
        self.disabled = False
        device = None if self.down is None else self.down.weight.device
        dtype = None if self.down is None else self.down.weight.dtype
        self.down = nn.Linear(self.input_dim, self.projection_dim, bias=self.bias, device=device, dtype=dtype)
        self.up = nn.Linear(self.projection_dim, self.output_dim, bias=self.bias, device=device, dtype=dtype)
        self.layer_norm = (
            nn.LayerNorm(self.input_dim, device=device, dtype=dtype) if self.has_layer_norm
            else Identity()
        )
        if self.zero_init:
            # same init as in the LoRA paper, can also be used to at inference to have adapters that return zero, i.e.,
            # default to the identity function (lets the user disable adapters at some layers by just removing them
            # from the checkpoint)
            # nn.init.normal_(self.down.weight, std=1/self.projection_dim)
            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.weight)
        else:
            delta = 1e-6
            nn.init.uniform_(self.down.weight, -delta, delta)
            nn.init.uniform_(self.up.weight, -delta, delta)
        if self.bias:
            nn.init.zeros_(self.down.bias)
            nn.init.zeros_(self.up.bias)

    def enable(self) -> None:
        if self.down is not None:  # not permanently disabled
            self.disabled = False

    def disable(self) -> None:
        self.disabled = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Shape:
            x: (B, S, D)
        
        Returns: tensor of shape (B, S, D)
        """
        if self.disabled:
            return x
        residual = x
        x = self.layer_norm(x)
        x = self.down(x)
        x = self.activation(x)
        x = self.up(x) * self.scaling
        return x + residual if self.residual else x

    def _load_from_state_dict(self, state_dict: dict[str, Tensor], prefix: str, *args, **kwargs) -> None:
        # At inference, automatically disable adapters that are missing from checkpoint
        if not self.training and prefix + 'down.weight' not in state_dict:
            self.disable()
            self.up = self.down = self.layer_norm = None  # disable permanently
            logger.warning(f"missing adapter '{prefix.rstrip('.')}', disabling it")
        # At inference, automatically change the bottleneck dimension of the adapters to match that of the 
        # checkpoint
        elif not self.training:
            projection_dim = state_dict[prefix + 'down.weight'].size(0)
            if projection_dim != self.projection_dim:
                logger.warning(f"changed dimension of adapter '{prefix.rstrip('.')}' from {self.projection_dim} "
                               f"to {projection_dim}")
                self.projection_dim = projection_dim
                self._init()

        return super()._load_from_state_dict(
            state_dict, prefix, *args, **kwargs
        )


class WordDropout(nn.Dropout2d):
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: tensor of shape (B, T, D)
        Returns:
            tensor of shape (B, T, D) where entire D columns are zeroed out
        """
        if self.p == 0:
            return x
        return super().forward(x.unsqueeze(-1)).squeeze(-1)


def checkpoint_wrapper(module: nn.Module, activate: bool = True) -> nn.Module:
    # modify the forward function of 'module' to use activation checkpointing
    if activate:
        module._no_ckpt_forward = module.forward
        module.forward = functools.partial(checkpoint, module._no_ckpt_forward, use_reentrant=False)
    return module


def PositionalEmbedding(type: str, num_embeddings: int, embedding_dim: int, shift: int = 2):
    # Positional embeddings are shifted by two positions by default for legacy reasons with fairseq. There is not real 
    # reason to do this.
    if type in ('alibi', 'rotary', 't5'):
        return DummyPositionalEmbedding(num_embeddings, embedding_dim)
    elif type == 'learned':
        return LearnedPositionalEmbedding(num_embeddings, embedding_dim, shift=shift)
    elif type == 'sinusoidal':
        return SinusoidalPositionalEmbedding(num_embeddings, embedding_dim, shift=shift)
    else:
        raise NotImplementedError


class DummyPositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()

    def forward(self, input: Tensor, offset: Union[LongTensor, int] = 0) -> Tensor:
        return 0.0


class SinusoidalPositionalEmbedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, shift: int = 0):
        """ Copied from fairseq """
        super().__init__()
        self.shift = shift
        num_embeddings += shift
        self.embedding_dim = embedding_dim
        half_dim = embedding_dim // 2
        weight = math.log(10000) / (half_dim - 1)
        weight = torch.exp(torch.arange(half_dim, dtype=torch.float) * -weight)
        weight = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * weight.unsqueeze(0)
        weight = torch.cat([torch.sin(weight), torch.cos(weight)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            weight = torch.cat([weight, torch.zeros(num_embeddings, 1)], dim=1)  # zero pad
        self.weight = weight
        self.register_buffer('_float_tensor', torch.FloatTensor(1))  # since self.weight is not a parameter,
        # it is not affected by "module.cuda()". But this dummy buffer is, which lets us know which device or data type
        # to use.

    def forward(self, length: int, offset: Union[LongTensor, int] = 0) -> Tensor:
        """
        Shape:
            offset: integer or tensor of shape (B,)
        
        Returns: tensor of shape (B, T, D) or (1, T, D)
        """
        assert length + self.shift - 1 < self.weight.size(0), (
            f'input sequence is too long: {length}, '
            f'positional embedding size: {self.weight.size(0)}'
        )
        
        self.weight = self.weight.to(self._float_tensor)
        positions = torch.arange(length, dtype=torch.long, device=self.weight.device)[None] + self.shift  # 1xT
        if torch.is_tensor(offset) and offset.dim() == 1:
            offset = offset.unsqueeze(1)  # BxT
        positions += offset

        return (
            self.weight.index_select(0, positions.view(-1))
            .view(-1, length, self.embedding_dim)
            .detach()
        )  # BxTxD or 1xTxD


class LearnedPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int, shift: int = 0):
        self.shift = shift  # arbitrary shift of 2 in fairseq
        super().__init__(num_embeddings + self.shift, embedding_dim)
        utils.embed_init(self.weight)

    def forward(self, length: int, offset: Union[LongTensor, int] = 0) -> Tensor:
        """
        Shape:
            offset: integer or tensor of shape (B,)
        
        Returns: tensor of shape (B, T, D) or (1, T, D)
        """
        assert length + self.shift - 1 < self.weight.size(0), (
            f'input sequence is too long: {length}, '
            f'positional embedding size: {self.weight.size(0)}'
        )

        positions = torch.arange(length, dtype=torch.long, device=self.weight.device)[None] + self.shift  # 1xT
        if torch.is_tensor(offset) and offset.dim() == 1:
            offset = offset.unsqueeze(1)  # BxT
        positions += offset
        
        return super().forward(positions)  # BxTxD or 1xTxD


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, shard_id: int = 0, shard_count: int = 1,
                 positional_encoding: str = 'none', lora_rank: int = 0, max_len: Optional[int] = None,
                 causal: bool = False, has_bias: bool = True, kv_heads: Optional[int] = None, key_bias: bool = True, 
                 sliding_window: Optional[int] = None, layer_id: int = 0, scaled: bool = True, rope_base: int = 10000,
                 alibi_max_bias: int = 8, max_qkv: Optional[float] = None, lora_alpha: int = 1):
        super().__init__()
        self.layer_id = layer_id
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert not max_qkv or max_qkv > 0
        self.max_qkv = max_qkv
        kv_heads = kv_heads or num_heads  # for grouped-query attention
        self.kv_heads = kv_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads  # does not work with T5 small
        self.kv_dim = self.kv_heads * self.head_dim
        self.q_dim = self.num_heads * self.head_dim
        assert num_heads % shard_count == 0
        assert kv_heads % shard_count == 0
        assert num_heads % kv_heads == 0
        self.scaled = scaled
        self.has_bias = has_bias
        self.sliding_window = sliding_window
        if sliding_window and flash_attn_func is None:
            utils.warn_once(logger, 'flash-attention is not installed: disabling the sliding window')
        
        self.k_proj = Linear(
            embed_dim,
            self.kv_dim // shard_count,
            bias=self.has_bias and key_bias,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        self.v_proj = Linear(
            embed_dim,
            self.kv_dim // shard_count,
            bias=self.has_bias,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        self.q_proj = Linear(
            embed_dim,
            self.q_dim // shard_count,
            bias=self.has_bias,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        self.out_proj = Linear(
            self.q_dim // shard_count,
            embed_dim,
            bias=self.has_bias and shard_id == 0,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        
        self.shard_count = shard_count
        self.shard_id = shard_id  # for tensor parallelism
        self.num_heads //= shard_count
        self.kv_heads //= shard_count
        self.max_len = max_len
        
        self.rotary_embed = self.alibi = self.t5_embed = None
        if positional_encoding == 'alibi':
            self.alibi = AlibiEmbedding(num_heads, shard_id, shard_count, causal=causal, max_bias=alibi_max_bias)
        elif positional_encoding == 'rotary':
            self.rotary_embed = RotaryEmbedding(self.head_dim, base=rope_base)
        elif positional_encoding == 't5':
            self.t5_embed = T5Embedding(self.num_heads, causal=causal)
        
        if not _fast_init:
            self.reset_parameters()

        self.causal = causal
        self.causal_mask = torch.empty(0, dtype=torch.bool)

    def reset_parameters(self) -> None:
        # we don't want tensor parallelism to affect how weights are initialized
        n = self.shard_count
        gain = math.sqrt((n + 1) / (2*n))

        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight, gain=gain)

        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    @utils.benchmark('attention')
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[BoolTensor] = None,
        state: Optional[dict[str, Tensor]] = None,
        return_attn: bool = False,
    ) -> tuple[Tensor, Tensor, dict]:
        """
        Args:
            attn_mask: boolean tensor with True at masked (key, value) positions
        
        Shape:
            query: (B, T, D)
            key:   (B, S, D)
            value: (B, S, D)
            attn_mask: (B, S) or (B, T, S)
        
        Returns: tuple (attn, attn_weights) with
            attn: tensor of shape (B, T, D)
            attn_weights: tensor of shape (B, T, H, S) or None
        """
        if attn_mask is not None and attn_mask.dim() == 2 and self.causal:
            # attn_mask can be a simple padding mask, in which case it is useless for causal attention (assuming the
            # padding tokens are always at the end...)
            attn_mask = None  # set to None to allow the use of fast causal attention below

        batch_size, tgt_len, embed_dim = query.size()
        src_len = key.size(1)

        q = self.q_proj(query)  # BxTxD
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim)  # BxTxHxD'
        k = self.k_proj(key)  # BxSxD
        k = k.view(batch_size, -1, self.kv_heads, self.head_dim)  # BxSxH'xD'
        v = self.v_proj(value)
        v = v.view(batch_size, -1, self.kv_heads, self.head_dim)  # BxSxH'xD'

        if self.max_qkv:
            for x in q, k, v:
                x.clamp_(min=-self.max_qkv, max=self.max_qkv)

        pos_offset = state['key'].size(1) if state else 0
        if self.rotary_embed is not None:
            q, k = self.rotary_embed(q, k, offset=pos_offset)

        if state is not None and 'key' in state:  # incremental mode with step > 0
            prev_key = state['key']  # BxSxHxD'
            prev_value = state['value']  # BxSxHxD'

            if self.max_len is not None:  # truncate the incremental state if needed
                prev_len = state['key'].size(1)
                delta = max(0, prev_len + src_len - self.max_len)
                prev_key = prev_key[:, delta:]
                prev_value = prev_value[:, delta:]

            k = torch.cat([prev_key, k], dim=1)
            v = torch.cat([prev_value, v], dim=1)
            src_len = k.size(1)

        if state is not None:  # incremental mode
            state['key'] = k
            state['value'] = v

        r = self.num_heads // self.kv_heads
        if r > 1:
            v = v.repeat_interleave(r, dim=2)
            k = k.repeat_interleave(r, dim=2)

        if (
            return_attn or  # Pytorch's `dot_product_attention` does not return attention scores
            attn_mask is not None or 
            self.alibi is not None or  # ALiBi applies a bias to the attention scores
            self.t5_embed is not None  # T5 applies a bias to the attention scores
        ):  # custom masking
            device = q.device
            dtype = q.dtype

            if attn_mask is None:
                pass
            elif attn_mask.dim() == 2:  # BxS
                attn_mask = attn_mask[:,None,None,:]  # Bx1x1xS
            else:  # BxTxS
                attn_mask = attn_mask[:,None]  # Bx1xTxS

            if self.causal:
                if self.causal_mask.size(0) < src_len:
                    size = 256 * math.ceil(src_len / 256)
                    self.causal_mask = torch.ones(size, size, dtype=torch.bool, device=device)
                    self.causal_mask = torch.triu(self.causal_mask, 1)
                    
                self.causal_mask = self.causal_mask.to(device)
                causal_mask = self.causal_mask[:src_len, :src_len]
                causal_mask = causal_mask[-tgt_len:]
                causal_mask = causal_mask.view(1, 1, tgt_len, src_len)  # 1x1xTxS
                attn_mask = causal_mask if attn_mask is None else (attn_mask + causal_mask)
            
            if attn_mask is not None:
                attn_mask = attn_mask.to(dtype).masked_fill(attn_mask, -float('inf'))  # bool -> float

            if self.alibi is not None:
                position_bias = self.alibi(q, k, offset=pos_offset)  # 1xHxTxS
                attn_mask = position_bias if attn_mask is None else position_bias + attn_mask
            elif self.t5_embed is not None:
                position_bias = self.t5_embed(q, k, offset=pos_offset)  # 1xHxTxS
                attn_mask = position_bias if attn_mask is None else position_bias + attn_mask

        dropout_p = self.dropout if self.training else 0
        scale = None if self.scaled else 1.0
        is_causal = self.causal and tgt_len > 1 and attn_mask is None
        use_flash_attn = (
            flash_attn_func is not None and
            q.dtype != torch.float32 and
            attn_mask is None and
            not return_attn
        )

        if use_flash_attn:
            utils.log_once(logger, 'using flash attention', level=logging.DEBUG)
            window_size = (self.sliding_window or -1, self.sliding_window or -1)
            attn: Tensor = flash_attn_func(
                q, k, v,
                causal=is_causal,
                dropout_p=dropout_p,
                softmax_scale=scale,
                window_size=window_size,
            )  # BxTxHxD'
            attn_weights = None
        elif not return_attn and hasattr(F, 'scaled_dot_product_attention'):
            utils.log_once(logger, 'using torch attention', level=logging.DEBUG)
            q = q.transpose(1, 2)  # BxHxTxD'
            k = k.transpose(1, 2)  # BxHxSxD'
            v = v.transpose(1, 2)  # BxHxSxD'
            attn: Tensor = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=is_causal,
                dropout_p=dropout_p,
                scale=scale,
            )  # BxHxTxD'
            attn_weights = None
            attn = attn.transpose(1, 2)
        else:
            utils.log_once(logger, 'using custom attention', level=logging.DEBUG)
            # use custom attention if we need the attention weights or flash attention is not available (e.g., 
            # Pytorch version that is too old)
            q = q.transpose(1, 2)  # BxHxTxD'
            k = k.transpose(1, 2)  # BxHxSxD'
            v = v.transpose(1, 2)  # BxHxSxD'
            attn, attn_weights = self.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                scale=scale
            )  # BxHxTxD'
            attn = attn.transpose(1, 2)
        
        attn = attn.reshape(batch_size, tgt_len, -1)  # BxTxD
        attn = self.out_proj(attn)

        return attn, attn_weights

    @classmethod
    def scaled_dot_product_attention(
        cls,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor,
        dropout_p: float = 0.0,
        scale: Optional[float] = None,
    ) -> tuple[Tensor, Tensor]:
        # q: BxHxTxD
        # k: BxHxSxD
        # v: BxHxSxD
        # attn_mask: BxHxTxS
        if scale is None:
            head_dim = q.shape[-1]
            scale = 1.0 / head_dim ** 0.5

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale  # BxHxTxS

        if attn_mask is not None:
            attn_weights += attn_mask
        
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float)
        attn_weights_float = attn_weights_float.nan_to_num()  # NaNs can happen with BLOOM models where the beginning
        # of sentence is replaced by a padding token
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_weights_ = F.dropout(attn_weights, p=dropout_p)  # BxHxTxS
        attn = torch.matmul(attn_weights_, v)  # BxHxTxD
        attn_weights = attn_weights.transpose(1, 2)  # BxTxHxS
        return attn, attn_weights


class ConvolutionSubsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int] = (3, 3),
        strides: Optional[tuple[int]] = None,
        activation: str = 'glu',
    ):
        super(ConvolutionSubsampler, self).__init__()
        strides = strides or tuple(2 for _ in kernel_sizes)
        assert len(strides) == len(kernel_sizes)
        r = 2 if activation == 'glu' else 1
        
        self.conv_layers = nn.ModuleList()
        for conv_id, (kernel_size, stride) in enumerate(zip(kernel_sizes, strides)):
            is_first = (conv_id == 0)
            is_last = (conv_id == len(kernel_sizes) - 1)
            conv_layer = nn.Conv1d(
                in_channels if is_first else mid_channels // r,
                out_channels * r if is_last else mid_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
            self.conv_layers.append(conv_layer)
        
        self.activation = nn.GLU(dim=1) if activation == 'glu' else nn.GELU()

    def get_new_length(self, length: LongTensor) -> LongTensor:
        """
        Shape:
            length: (B,)
        
        Returns: tensor of shape (B,) with the new lengths
        """
        for conv in self.conv_layers:
            length = 1 + torch.div(
                length - conv.kernel_size[0] + 2 * conv.padding[0],
                conv.stride[0],
                rounding_mode='floor',
            )
        return length

    def forward(self, x: Tensor, length: LongTensor) -> tuple[Tensor, LongTensor]:
        """
        Shape:
            x: (B, S, D)
            length: (B,)
        
        Returns: tuple (x, new_length) with
            x: tensor of shape (B, S', D')
            new_length: tensor of shape (B,) with the new lengths (<= S')
        """
        x = x.transpose(1, 2).contiguous()  # -> BxDxS
        for conv in self.conv_layers:
            x = conv(x)
            x = self.activation(x)
        x = x.transpose(1, 2).contiguous()  # -> BxS'xD'
        return x, self.get_new_length(length)


def remove_unused_parameters(
    model: nn.Module,
    state_dict: dict,
    param_regex: Optional[str] = None,
) -> dict[str, Tensor]:
    """
    Removes all parameters from state_dict whose name matches `param_regex` and that are unused by the model.

    If `param_regex` is None, all unused parameters are removed from state_dict.

    `param_regex` should have two capturing groups: one for the main part of the model this parameter belongs to
    (e.g., encoder or decoder) and one for the name of the layer or parameter set this parameter belongs to 
    (e.g., "lang:de" adapters)
    """
    model_state_dict = model.state_dict()
    unused_uids = defaultdict(set)
    unused_params = {}
    parameter_names = set(model_state_dict)
    for name in list(state_dict):
        match = regex.match(param_regex, name) if param_regex else None
        if name not in parameter_names and (param_regex is None or match):
            unused_params[name] = state_dict.pop(name).cpu()
            if match:
                module, uid = match.groups()
                unused_uids[module].add(uid)
    for module, uids in unused_uids.items():
        logger.warning(f'found {len(uids)} set(s) of unused {module} params: ' + ', '.join(sorted(uids)))
    return unused_params


def add_missing_parameters(model: nn.Module, state_dict: dict, param_regex: Optional[str] = None) -> None:
    """
    Adds all missing parameters from state_dict whose name matches `param_regex`. The current value in the model
    (typically a randomly initialized value) is used
    """
    model_state_dict = model.state_dict()
    missing_uids = defaultdict(set)
    for name, param in model_state_dict.items():
        match = regex.match(param_regex, name) if param_regex else None
        if name not in state_dict and (param_regex is None or match):
            state_dict[name] = param
            if match:
                module = match.groupdict().get('module')
                uid = match.groupdict().get('uid')
                if module is not None and uid is not None:
                    missing_uids[module].add(uid)
    for module, uids in missing_uids.items():
        logger.warning(
            f'missing {module} params initialized at random: ' +
            ', '.join(sorted(uids))
        )


class Embedding(nn.Embedding):
    """
    Embeddings which can be partially frozen
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        freeze_mask: Optional[BoolTensor] = None,
    ):
        """
        Shape:
            freeze_mask: (V,)
        """
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        if not _fast_init:
            utils.embed_init(self.weight, padding_idx)

        if freeze_mask is not None:
            self.freeze_mask = freeze_mask
            self.frozen_embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
            self.frozen_embedding.requires_grad = False
        else:
            self.frozen_embedding = None

    def forward(self, input: LongTensor) -> Tensor:
        """
        Shape:
            input: (B, S)
        Returns: tensor of shape (B, S, D)
        """
        device = input.device
        input = input.to(self.weight.device).clip(min=0)
        max_index = input.max() if input.numel() > 0 else 0
        embed_size = self.weight.size(0)
        assert max_index < embed_size, f'index {max_index} is outside of the embedding matrix ({embed_size})'
        embed = super().forward(input)
        if self.frozen_embedding is not None:
            frozen = self.frozen_embedding(input)
            self.freeze_mask = self.freeze_mask.to(self.weight.device)
            mask = self.freeze_mask[input].unsqueeze(-1)
            embed = (~mask) * embed + mask * frozen
        return embed.to(device)

    def projection(self, input: Tensor) -> Tensor:
        """
        Shape:
            input: (B, T, D)
        Returns: tensor of shape (B, T, V)
        """
        input = input.to(self.weight.device)
        out = torch.matmul(input, self.weight.T)
        if self.frozen_embedding is not None:
            frozen = torch.matmul(input, self.frozen_embedding.weight.T)
            mask = self.freeze_mask[None, None]
            out = (~mask) * out + mask * frozen
        return out


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embeddings as described in this paper: https://arxiv.org/abs/2104.09864v2

    This implementation is inspired from the HuggingFace GPT-J implementation:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/modeling_gptj.py

    Note that it also works with Llama, provided that the query and key weight matrices are reshaped accordingly 
    beforehand, like so:
    ```
    w.reshape(num_heads, -1, 2, w.size(-1)).transpose(1, 2).reshape(-1, w.size(-1))
    ```
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.build(max_len=256)  # will be automatically extended if needed

    def build(self, max_len: int):
        t = torch.arange(max_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos()
        self.sin = emb.sin()
        self.max_len = max_len

    def rotate(self, x):
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, query: Tensor, key: Tensor, offset: Union[LongTensor, int]) -> tuple[Tensor, Tensor]:
        """
        Args:
            query: tensor of shape (B, T, H, D)
            key: tensor of shape (B, S, H, D)
            offset: number of previous tokens (during incremental decoding), can be a tensor of shape (B,) if the
                length of the each sequence in the batch is different (due to different padding)
        
        Returns: rotated query and key
        """
        bsz, seq_len, _, dim = query.shape

        total_len = offset.max() if torch.is_tensor(offset) else offset
        total_len += seq_len
        if total_len > self.max_len:  # extend the size of the embeddings if needed
            new_max_len = 2**math.ceil(math.log2(total_len))  # closest power of 2
            self.build(new_max_len)

        self.cos = self.cos.to(query)  # device and dtype
        self.sin = self.sin.to(query)

        if torch.is_tensor(offset) and offset.dim() == 1:
            cos = self.cos.repeat(bsz, 1, 1)  # B x MAX_LEN x D
            sin = self.sin.repeat(bsz, 1, 1)  # B x MAX_LEN x D
            positions = torch.arange(seq_len, device=query.device).unsqueeze(0) + offset.unsqueeze(1)  # BxT
            positions = positions.unsqueeze(-1).repeat(1, 1, cos.size(-1))  # BxTxD
            cos = torch.gather(cos, dim=1, index=positions)  # BxTxD
            sin = torch.gather(sin, dim=1, index=positions)  # BxTxD
            cos = cos.unsqueeze(2)  # BxTx1xD
            sin = sin.unsqueeze(2)  # BxTx1xD
        else:
            cos = self.cos[None, offset : offset + seq_len, None]  # 1xTx1xD
            sin = self.sin[None, offset : offset + seq_len, None]  # 1xTx1xD

        q = query[..., :self.dim]  # BxTxHxD
        k = key[..., :self.dim]
    
        q = (q * cos) + (self.rotate(q) * sin)
        k = (k * cos) + (self.rotate(k) * sin)

        if self.dim < dim:
            q = torch.cat([q, query[..., self.dim:]], dim=-1)
            k = torch.cat([k, key[..., self.dim:]], dim=-1)
        return q, k


class AlibiEmbedding(nn.Module):
    """
    ALiBi positional embeddings as described in this paper: https://arxiv.org/abs/2108.12409, which are used 
    by the BLOOM models.

    Note that this implementation is probably slower than HuggingFace's (which uses the translation invariance property
    of the softmax), but it also works with encoder (non-causal) self-attention.
    
    For encoder self-attention, the `causal` property can be set to False, in which case half the attention heads 
    will be dedicated to the left context, and half to the right context. Note that if this argument is False 
    (the default), the same bias will be applied at a given distance to the right or to the left of a given position.
    
    Also note that these embeddings shouldn't be applied to cross-attention.
    """
    def __init__(self, num_heads: int, shard_id: int = 0, shard_count: int = 1, causal: bool = True, max_bias: int = 8):
        super().__init__()
        self.causal = causal
        self.alibi_slopes = ((2 ** (-max_bias / num_heads)) ** (torch.arange(num_heads) + 1))
        self.num_heads = num_heads // shard_count
        self.alibi_slopes = self.alibi_slopes[shard_id * self.num_heads:(shard_id + 1) * self.num_heads]

    def forward(self, query: Tensor, key: Tensor, offset: Union[LongTensor, int] = 0)  -> Tensor:
        """
        Args:
            query: tensor of shape (B, T, H, D)
            key: tensor of shape (B, S, H, D)
            offset: number of previous tokens (during incremental decoding), can be a tensor of shape (B,) if the
                length of the each sequence in the batch is different (due to different padding)
        
        Returns:
            tensor of shape (B, H, T, S) or (1, H, T, S)
        """
        query_len = query.size(1)
        key_len = key.size(1)

        self.alibi_slopes = self.alibi_slopes.to(query)
        
        query_position = torch.arange(
            query_len,
            dtype=torch.long,
            device=query.device,
        )[None, None, :, None]  # 1x1xTx1
        
        key_position = torch.arange(
            key_len,
            dtype=torch.long,
            device=query.device,
        )[None, None, None, :]  # 1x1x1xS

        if torch.is_tensor(offset) and offset.dim() == 1:
            query_position = query_position + offset[:, None, None, None]  # Bx1xTx1
        else:
            query_position += offset  # 1x1xTx1

        relative_position = key_position - query_position  # Bx1xTxS or 1x1xTxS
        relative_position = -relative_position.abs()  # symmetrical
        relative_position = relative_position.expand(-1, self.num_heads, -1, -1)  # BxHxTxS or 1xHxTxS
        alibi = self.alibi_slopes[None, :, None, None] * relative_position  # BxHxTxS or 1xHxTxS

        if not self.causal:
            # non-causal, break symmetry by allocating half the heads to the left context, and half to the 
            # right context, as suggested here: https://github.com/ofirpress/attention_with_linear_biases/issues/5
            assert key_len == query_len
            head_mask = torch.ones(query_len, query_len, dtype=torch.bool, device=query.device)
            head_mask = torch.triu(head_mask, 1).unsqueeze(0).expand(self.num_heads // 2, -1, -1)
            head_mask = torch.concat([head_mask, ~head_mask], dim=0)  # HxTxS
            alibi = alibi.masked_fill(head_mask[None], float('-inf'))

        return alibi


class T5Embedding(nn.Module):
    """
    T5 positional embeddings as implemented here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L351
    """
    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        causal: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.causal = causal
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    def _relative_position_bucket(self, relative_position):
        """
        Args:
            relative_position: an int32 Tensor

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)

        Copied verbatim from HuggingFace Transformers
        """
        relative_buckets = 0
        if self.causal:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
            num_buckets = self.num_buckets
        else:
            num_buckets = self.num_buckets // 2
            relative_buckets += (relative_position > 0).long() * num_buckets
            relative_position = torch.abs(relative_position)
            
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def forward(self, query: Tensor, key: Tensor, offset: Union[LongTensor, int] = 0)  -> Tensor:
        """
        Args:
            query: tensor of shape (B, T, H, D)
            key: tensor of shape (B, S, H, D)
            offset: number of previous tokens (during incremental decoding), can be a tensor of shape (B,) if the
                length of the each sequence in the batch is different (due to different padding)
        
        Returns:
            tensor of shape (B, H, T, S) or (1, H, T, S)
        """
        query_len = query.size(1)
        key_len = key.size(1)

        query_position = torch.arange(
            query_len,
            dtype=torch.long,
            device=query.device,
        )[None, :, None]  # 1xTx1
        
        key_position = torch.arange(
            key_len,
            dtype=torch.long,
            device=query.device,
        )[None, None, :]  # 1x1xS

        if torch.is_tensor(offset) and offset.dim() == 1:
            query_position = query_position + offset[:, None, None]  # BxTx1
        else:
            query_position += offset  # 1xTx1
        
        relative_position = key_position - query_position  # BxTxS or 1xTxS
        relative_position_bucket = self._relative_position_bucket(relative_position)  # BxTxS or 1xTxS
        values = self.relative_attention_bias(relative_position_bucket)  # BxTxSxH or 1xTxSxH
        values = values.permute([0, 3, 1, 2])  # BxHxTxS or 1xHxTxS        
        return values
