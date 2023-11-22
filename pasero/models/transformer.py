# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import logging
import math
import functools
import regex
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
from torch import Tensor, LongTensor, BoolTensor
from . import modules
from .modules import Embedding, Identity
from pasero import utils
from pasero.tasks import Task
from pasero.utils import defined, SpecialTokens
from pasero.config import DistributedConfig, TransformerConfig


logger = logging.getLogger('models')


class BaseModel(nn.Module):
    """
    Base class for encoders, decoders and encoder-decoder models
    """
    special_tokens: SpecialTokens

    @property
    def padding_idx(self) -> int: return self.special_tokens.padding_idx
    @property
    def bos_idx(self) -> int: return self.special_tokens.bos_idx
    @property
    def eos_idx(self) -> int: return self.special_tokens.eos_idx
    @property
    def unk_idx(self) -> int: return self.special_tokens.unk_idx

    def parallelize(self, devices: list[str]) -> None: pass


class Encoder(BaseModel):
    """
    Base class for all encoders
    """
    task: Task
    embed_tokens: Optional[Embedding]
    cfg: TransformerConfig
    
    @property
    def special_tokens(self) -> SpecialTokens:
        return self.task.special_tokens

    @property
    def max_len(self) -> int:
        return self.cfg.encoder_max_len

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Decoder(BaseModel):
    """
    Base class for all decoders
    """
    task: Task
    embed_tokens: Optional[Embedding]
    cfg: TransformerConfig
    
    @property
    def special_tokens(self) -> SpecialTokens:
        return self.task.special_tokens

    @property
    def max_len(self) -> int:
        return self.cfg.decoder_max_len

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def reorder_state(state: Optional[dict[str, Tensor]], indices: LongTensor) -> None:
        if not state:
            return
        for k, v in state.items():
            if torch.is_tensor(v) and v.dim() > 0:
                indices = indices.to(v.device)
                state[k] = v.index_select(0, indices)


class DummyEncoder(Encoder):
    """
    Encoder which can also be used as an empty encoder to create decoder-only Transformers (e.g., for language modeling)
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.embed_tokens = None

    @property
    def max_len(self) -> int:
        return 0

    def forward(self, *args, **kwargs):
        return None, None, {}


class EncoderDecoder(BaseModel):
    """
    Base class for Transformer and EnsembleModel
    """
    task: Task
    cfg: TransformerConfig
    
    @property
    def special_tokens(self) -> SpecialTokens:
        return self.task.special_tokens

    @property
    def total_param_count(self) -> int:
        return sum(param.numel() for param in self.parameters())


class Transformer(EncoderDecoder):
    """
    Base class for all other models: encoder-decoder Transformer Base model.
    Other Transformer architectures (e.g., Transformer Big, mBART, etc.) are defined in 'config.py'
    
    Defining a variant of the Transformer (i.e., with other default hyper-parameter values) is done by subclassing
    `TransformerConfig`

    Defining new architectures (e.g., with a different encoder or decoder) is done by sublassing this class and 
    overriding `build_encoder` or `build_decoder` (example: `adapters.AdapterTransformer`)
    """

    # parameters to split along the columns when doing tensor parallelism
    col_parallel_regex = r'.*\.(k_proj|q_proj|v_proj|fc1|fc3)\.(weight|bias|lora\.up\.weight)'
    # parameters to split along the rows when doing tensor parallelism (note that the bias parameters are only on
    # the first rank)
    row_parallel_regex = r'.*\.(out_proj|fc2|t5_embed\.relative_attention_bias)\.(weight|bias|lora\.down\.weight)'

    def __init__(
        self,
        cfg: TransformerConfig,
        dist_cfg: DistributedConfig,
        task: Task,
    ):
        """
        Args:
            cfg: dataclass containing the model hyper-parameters. Note that at training, they are defined by command 
                line options and/or YAML configuration. However, at decoding, the model arguments are extracted from
                the checkpoint directly and frozen. The only ways to modify them are to use the '--model-args'
                option (which takes a serialized json dict) or to add new options in `DecodingConfig` and modify 
                `TransformerConfig.setup_for_inference` to take these new options into account.
            dist_cfg: contains the tensor parallelism settings
            task: which task this model will be used for (e.g., an instance of TranslationTask or LanguageModelingTask),
                contains the tokenizers (used for initializing the embeddings), special token ids, set of languages,
                etc.
        """
        super().__init__()
        self.cfg = cfg
        self.dist_cfg = dist_cfg
        self.task = task

        self.find_unused_parameters = False  # used by Trainer to setup `TrainingConfig.find_unused_parameters`
        self.batch_by = None  # used in `train.py` to setup `TrainingDatasetConfig.batch_by`, to group samples into 
        # homegeneous batches (e.g., by language or domain)
        
        self.encoder = self.build_encoder()
        embed = self.encoder.embed_tokens if self.cfg.shared_embeddings else None
        self.decoder = self.build_decoder(embed=embed)

    @property
    def shard_count(self):
        """
        Number of tensor parallelism shards (i.e., model parameters will be split in this many parts, each on a 
        different GPU). This also corresponds to the number of checkpoints that should be saved.
        """
        return self.dist_cfg.tp_size or 1

    @property
    def shard_id(self):
        """
        Tensor parallelism rank of the current process.
        """
        return self.dist_cfg.tp_rank or 0

    @property
    def is_sharded(self):
        """
        Whether this model needs to be saved as multiple checkpoints and has to communicate across ranks during its
        forward pass (which means that it should receive batches of the exact same size on all ranks to avoid deadlocks)
        """
        return self.shard_count > 1

    def build_encoder(
        self,
        embed: Optional[Embedding] = None,
    ) -> Union[DummyEncoder, 'TransformerEncoder']:
        """
        Creates the encoder. Can be a "dummy decoder" (i.e., a fake module whose forward function does nothing) in the 
        case of decoder-only models.
        Override this method in subclasses to implement new architectures.
        """
        if self.cfg.model_type == 'decoder':
            return DummyEncoder()
        else:
            return TransformerEncoder(
                self.cfg,
                self.dist_cfg,
                task=self.task,
                embed=embed,
            )

    def build_decoder(self, embed: Optional[Embedding] = None) -> 'TransformerDecoder':
        """
        Creates the decoder. Override this method in subclasses to implement new architectures.

        `embed` corresponds the encoder's embeddings when --shared-embeddings is set
        """
        return TransformerDecoder(
            self.cfg,
            self.dist_cfg,
            task=self.task,
            embed=embed,
        )

    def forward(
        self,
        source: Optional[Tensor] = None,
        source_length: Optional[LongTensor] = None,
        decoder_input: Optional[LongTensor] = None,
        target: Optional[LongTensor] = None,
        prompt_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, dict]:
        """
        Run a full forward step of the Transformer: taking a batch of source and target sequences,
        1) encode the sources with the encoder and generate sequences of encoder states
        2) input the decoder with these encoder states and decoder inputs (i.e., targets shifted by one position to the 
        right)
        3) generate a distribution over the target vocabulary and compute cross entropy against the target sequences

        Args:
            source: padded sequences of indices or source features
            source_length: real lengths of the sequences in `source`
            decoder_input: target indices shifted by one position to the right (the first token is BOS)
            target: padded sequences of indices ending with EOS
            prompt_mask: mask with 1s at the position of prefix tokens (e.g., language codes or source tokens in the 
                case of decoder-only machine translation)
        Shape:
            source: (B, S) or (B, S, D) with binary inputs (e.g., with '--task speech_translation')
            source_length: (B,)
            decoder_input: (B, T)
            target: (B, T)
            prompt_mask: (B, T)

        Note that `decoder_input` has the same length `target`. The dimension named "T" in the shape notations
        above includes the EOS token in `target` and the BOS token of `decoder_input`.
        
        Example with T=3 and B=1 (here bos_idx = eos_idx = 2):
        ```
        decoder_input = [[2, 43, 134]]
        target = [[43, 134, 2]]
        ```

        Returns: a tuple (loss, logs), the output of `compute_loss`
        """
        encoder_out, encoder_mask, enc_layer_outputs = self.encoder(source, source_length, **kwargs)
        # encoder_out: Tensor of shape (B, S, D) containing the output of the last encoder layer
        # encoder_mask: BoolTensor of shape (B, S) with True at every padding token position
        decoder_out, dec_layer_outputs = self.decoder(
            encoder_out,
            encoder_mask,
            decoder_input,
            prompt_mask=prompt_mask,
            **kwargs,
        )
        # decoder_out: Tensor of shape (B, T, V) containing the output of the output projection (i.e., logits)
        layer_outputs = {**enc_layer_outputs, **dec_layer_outputs}

        if self.cfg.prompt_loss == 0:  # disable loss computation over the prefix tokens
            # mask the prefix, including the separator (we don't want to compute the loss over those tokens)
            target = target.masked_fill(prompt_mask, self.padding_idx)
        
        prompt_loss_scale = self.cfg.prompt_loss

        if prompt_loss_scale == 1.0:  # same loss multiplier is applied on all parts of the sequence
            return self.compute_loss(decoder_out, target, layer_outputs)
        else:
            """
            Typically used in decoder-only machine translation or LM alignment, to apply a different loss on the 
            prompt tokens than on the rest of the sequence.

            Prompt loss is the loss computed on the prompt tokens (multiplier: `prompt_loss_scale`)
            Gen loss is the loss computed on the other tokens (multiplier: 1)

            Content of the `logs` dict:

            - num_tokens: total number of tokens (prompt + gen)
            - num_prompt_tokens: number of prompt tokens
            - loss: gen loss + scaled prompt loss (to normalize by `num_tokens`)
            - nll_loss: gen nll loss (to normalize by `num_tokens - num_prompt_tokens`)
            - prompt_nll_loss: Prompt nll loss (to normalize by `num_prompt_tokens`)
            """
            loss, logs = self.compute_loss(  # Gen loss
                decoder_out,
                target.masked_fill(prompt_mask, self.padding_idx),
                layer_outputs,
            )
            
            if prompt_loss_scale > 0:  # Prompt loss
                prompt_loss, prompt_logs = self.compute_loss(
                    decoder_out,
                    target.masked_fill(~prompt_mask, self.padding_idx),
                    layer_outputs,
                )
                logs['prompt_nll_loss'] = prompt_logs['nll_loss']
                logs['loss'] = logs['loss'] + prompt_loss_scale * prompt_logs['loss']
                logs['num_tokens'] += prompt_logs['num_tokens']
                logs['num_prompt_tokens'] = prompt_logs['num_tokens']
                loss += prompt_loss_scale * prompt_loss

            return loss, logs

    @utils.benchmark('loss')
    def compute_loss(
        self,
        logits: Tensor,
        target: LongTensor,
        layer_outputs: dict,  # unused here, but may be used in subclasses
        *args,     # unused
        **kwargs,  # unused
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute the cross entropy loss for this model.
        Transformer can be subclassed to define other models or other training losses.

        Note about distributed training with data parallelism: the loss should not be reduced here
        (i.e., normalized by the number of tokens or sentences) as reduction by tokens is done in `Trainer`. The 
        returned `logs` dict should contain a `num_tokens` entry that will be used for this.

        Args:
            logits: outputs of the vocabulary projection of a Transformer decoder (i.e., scores for every token in the 
                target vocabulary)
            target: training ground-truth against which to compute cross-entropy (ending with EOS)
            layer_outputs: optional dictionary of layer names and layer outputs, tensors of shape (B, T, ...) not used
                for cross-entropy but potentially useful for other losses
        Shape:
            logits: (B, T, V)
            target: (B, T)
        
        Returns: a tuple (loss, logs) with
            loss: scalar Tensor, aggregated loss term to use for training
            logs: dictionary of other loss terms as floats for logging, and `num_tokens` used for reduction
        """
        batch_size = target.size(0)
        logits = logits.float().view(-1, logits.size(-1))
        target = target.view(-1)
        # activations take (2 * NUM_TGT_TOKENS * VOCAB_SIZE * 4) bytes 
        loss_fn = functools.partial(
            F.cross_entropy,
            logits,
            target,
            ignore_index=self.padding_idx,
            reduction='sum',
        )

        if self.cfg.label_smoothing:
            # activations take (3 * NUM_TGT_TOKENS * VOCAB_SIZE * 4) bytes
            loss = loss_fn(label_smoothing=self.cfg.label_smoothing)
            with torch.no_grad():
                nll_loss = loss_fn()
        else:
            loss = nll_loss = loss_fn()

        logs = {
            'loss': loss.item() / math.log(2),
            'nll_loss': nll_loss.item() / math.log(2),
            'num_tokens': (target != self.padding_idx).sum().item(),
            'num_lines': batch_size,
        }
        return loss, logs

    def remap_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """
        Run various re-mapping operations, like shifting layers or re-mapping embeddings that come from a different 
        vocabulary.

        These operations are not done in `update_state_dict`, as they should be done only once, when finetuning a 
        different model and not when resuming training.

        `state_dict` is modified in-place, no new dict is returned.
        """
        enc_embed_name = 'encoder.embed_tokens.weight'
        dec_embed_name = 'decoder.embed_tokens.weight'
        out_proj_name = 'decoder.output_projection.weight'

        # Remap the embeddings according to the task's configuration (e.g., --old-source-dict and --old-target-dict)
        if enc_embed_name in state_dict:
            state_dict[enc_embed_name] = self.task.remap_encoder_embed(state_dict[enc_embed_name])
        if dec_embed_name in state_dict:
            state_dict[dec_embed_name] = self.task.remap_decoder_embed(state_dict[dec_embed_name])
        if out_proj_name in state_dict:
            state_dict[out_proj_name] = self.task.remap_decoder_embed(state_dict[out_proj_name])
        
        # Shift layers from the checkpoint if needed
        for component in 'encoder', 'decoder':
            shift_layers = getattr(self.cfg, f'shift_{component}_layers')
            if self.training and shift_layers:
                new_state_dict = {}
                for key, value in state_dict.items():
                    r = rf'{component}\.layers\.(\d+)\.'
                    match = regex.match(r, key)
                    if match:
                        layer_id = int(match.group(1)) + shift_layers
                        key = regex.sub(r, f'{component}.layers.{layer_id}.', key)
                    new_state_dict[key] = value
                state_dict.clear()
                state_dict.update(new_state_dict)

    def update_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """
        Modify a model checkpoint to make it compatible with Transformer before attempting to load it.
        
        This includes renaming or re-mapping of fairseq or HuggingFace parameters, adding some shared parameters that
        may have been removed by `clean_state_dict` to save disk space, adding missing adapter parameters (in 
        AdapterTransformer), etc.
        
        Note that this method may be applied several times to the same checkpoint (e.g., when resuming training).
        Operations that should only be run once at the very beginning of training (e.g., embedding re-mapping of an 
        older model) should be implemented in `remap_state_dict`.

        It can also have a different behavior at training and inference (using the `Module.training` property). At  
        inference, the model should be put in evaluation mode (`model.eval()`) before calling this method.
        """

        # remove useless fairseq params
        for k in list(state_dict):
            if k.endswith('.version'):
                state_dict.pop(k)
        
        state_dict.pop('lm_head.weight', None)   # HuggingFace LMs
        
        # Find shared parameters that may be missing from the checkpoint (shared embeddings and tied output projection)
        enc_embed_name = 'encoder.embed_tokens.weight'
        dec_embed_name = 'decoder.embed_tokens.weight'

        # some checkpoints may only contain the encoder or decoder embeddings
        if enc_embed_name in state_dict and dec_embed_name not in state_dict:
            state_dict[dec_embed_name] = state_dict[enc_embed_name]
        if dec_embed_name in state_dict and enc_embed_name not in state_dict:
            state_dict[enc_embed_name] = state_dict[dec_embed_name]
        
        if self.encoder.embed_tokens is None:  # decoder-only or speech-translation model
            state_dict.pop(enc_embed_name, None)

        if self.cfg.tied_output_projection:
            # remove unused param that is sometimes present in old checkpoints
            for name in list(state_dict):
                if name.endswith('.output_projection.weight'):
                    state_dict.pop(name)

        # Remap old fairseq attention parameters
        for name in list(state_dict):
            if name.endswith('.in_proj_weight') or name.endswith('.in_proj_bias'):
                param = state_dict.pop(name)
                dim = param.size(0) // 3
                for i, s in enumerate(['.q_proj.', '.k_proj.', '.v_proj.']):
                    state_dict[name.replace('.in_proj_', s)] = param[dim * i:dim * (i + 1)]
            else:
                new_name = name.replace('decoder.final_layer_norm.', 'decoder.layer_norm.')  # HF checkpoints
                state_dict[new_name] = state_dict.pop(name)

        # Frozen embeddings are a copy of the regular embeddings
        frozen_embed_name = enc_embed_name.replace('.weight', '.frozen_embedding.weight')
        if self.task.freeze_encoder_embed_mask is not None:
            state_dict[frozen_embed_name] = state_dict[enc_embed_name]
        else:
            state_dict.pop(frozen_embed_name, None)

        if self.cfg.lora_rank and self.training:
            # We don't want to throw an error when training new adapters: add the randomly initialized 
            # parameters to the state dict to avoid errors in super().load_state_dict
            modules.add_missing_parameters(self, state_dict, r'.*\.lora\..*')
        
        if not self.training:
            # For efficiency, merge lora weights with linear weights
            for name in list(state_dict):
                if (m := regex.fullmatch(r'(?P<prefix>.*\.)lora\.down\.weight', name)):
                    prefix = m.group('prefix')
                    lora_down = state_dict.pop(prefix + 'lora.down.weight')
                    lora_up = state_dict.pop(prefix + 'lora.up.weight')
                    # do matmul in float32 because float16 matmul does not work on CPU
                    rank = lora_down.size(0)
                    patch = torch.matmul(
                        lora_up.float(),
                        lora_down.float() * self.cfg.lora_alpha / rank,
                    ).to(lora_down.dtype)
                    state_dict[prefix + 'weight'] += patch

    @classmethod
    def shard_state_dict(
        cls,
        state_dict: dict[str, Tensor],
        shard_id: int,
        shard_count: int,
        total_shard_count: int,  # unused here
    ) -> dict:
        """
        Shard given state dict into `shard_count` parts and return the one corresponding to given `shard_id`.
        This is used to load checkpoints with fewer shards than the current model's shard count, for example when
        using tensor parallelism to finetune a model trained without tensor parallelism.

        Args:
            - state_dict: model parameters to shard into `shard_count` parts
            - shard_count: the number of parts `state_dict` will be sharded into
            - shard_id: which part should go to this process (the other will be discarded)
            - total_shard_count: the total number of shards this checkpoint currently has
        """
        state_dict_new = {}
        for key, value in state_dict.items():
            dim = value.dim()
            if regex.fullmatch(cls.col_parallel_regex, key):
                value = value.view(shard_count, value.size(0) // shard_count, -1)[shard_id]
                if dim == 1:
                    value = value.squeeze(-1)
            elif regex.fullmatch(cls.row_parallel_regex, key):
                if dim > 1:
                    value = value.view(-1, shard_count, value.size(1) // shard_count)[:, shard_id].squeeze(-1)
                elif shard_id > 0:
                    continue
            state_dict_new[key] = value
        return state_dict_new

    @classmethod
    def unshard_state_dict(
        cls,
        *state_dicts: dict[str, Tensor],
        total_shard_count: int,  # unused here
    ) -> dict:
        """
        Merge model state dicts that were initially on different ranks but now need to be loaded on the same rank. This
        is used to load checkpoints with more shards than the current model's shard count, for example when decoding
        on a single GPU with a model trained with tensor parallelism.

        Args:
            - state_dicts: list of model shards that need to be merged
            - total_shard_count: the total number of shards this checkpoint currently has
        """
        state_dict_new = {}
        for key, value in state_dicts[-1].items():
            dim = value.dim()
            if regex.fullmatch(cls.col_parallel_regex, key):
                value = torch.cat([state_dict[key] for state_dict in state_dicts])
            elif regex.fullmatch(cls.row_parallel_regex, key) and dim > 1:
                value = torch.cat([state_dict[key] for state_dict in state_dicts], dim=-1)
            state_dict_new[key] = value
        for state_dict in state_dicts[:-1]:
            for key, value in state_dict.items():
                state_dict_new.setdefault(key, value)
        return state_dict_new

    def set_ddp_params_and_buffers_to_ignore(self):
        """
        Defines the model parameters that should be ignored by DDP when reducing gradients
        """
        params_to_ignore = getattr(self, '_ddp_params_and_buffers_to_ignore', [])
        self._ddp_params_and_buffers_to_ignore = params_to_ignore
        if (self.dist_cfg.tp_size or 1) > 1:
            for name, param in self.named_parameters():
                if regex.fullmatch(self.col_parallel_regex, name) or regex.fullmatch(self.row_parallel_regex, name):
                    params_to_ignore.append(name)  # used by DDP to avoid reducing the gradients of this parameter
                    param._is_sharded = True  # used in training.py and optimization.py to normalize gradients 
                    # correctly, depending on whether the parameter is shared or local to this rank (aka sharded)

    def load_state_dict(self, state_dict: dict[str, Tensor], strict: bool = True):
        """ More verbose version of `nn.Module.load_state_dict` """
        status = super().load_state_dict(state_dict, strict)
        if not strict:
            if status.missing_keys:
                logger.warning('missing keys: ' + ' '.join(status.missing_keys))
            if status.unexpected_keys:
                logger.warning('unexpected keys: ' + ' '.join(status.unexpected_keys))
        return status

    def clean_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """
        Remove redundant parameters which do not need to be saved in the checkpoint.
        """
        decoder_embed_key = 'decoder.embed_tokens.weight'
        if decoder_embed_key in state_dict and state_dict[decoder_embed_key].numel() == 0:
            state_dict.pop(decoder_embed_key)
        state_dict.pop('encoder.embed_tokens.frozen_embedding.weight', None)
        state_dict.pop('decoder.embed_tokens.frozen_embedding.weight', None)

    def parallelize(self, devices: list[str]) -> None:
        """
        Activate pipeline parallelism: distribute the model's layers across several GPUs.
        This is only supported at inference.
        """
        assert not self.training
        if len(devices) == 1 or isinstance(self.encoder, DummyEncoder):
            enc_devices = dec_devices = devices
        else:
            n = len(devices) // 2
            enc_devices, dec_devices = devices[:n], devices[n:]
        self.encoder.parallelize(enc_devices)
        self.decoder.parallelize(dec_devices)  # if the embeddings are shared, this will move encoder's embeddings
        # to devices[-1]


class TransformerEncoder(Encoder):
    def __init__(
        self,
        cfg: TransformerConfig,
        dist_cfg: DistributedConfig,
        task: Task,
        embed: Optional[Embedding] = None,
    ):
        """
        Args:
            cfg: model hyper-parameters
            embed: optional existing Embedding to reuse
        """
        super().__init__()
        self.cfg = cfg
        self.dist_cfg = dist_cfg
        self.task = task

        if self.task.encoder_num_embeddings == 0:
            self.embed_tokens = None
        elif embed is None:
            self.embed_tokens = Embedding(
                self.task.encoder_num_embeddings,
                cfg.embed_dim,
                self.padding_idx,
                task.freeze_encoder_embed_mask,
            )
        else:
            self.embed_tokens = embed
        
        self.subsample = self.in_linear = None
        input_dim = cfg.input_dim or cfg.embed_dim
        
        encoder_max_len = cfg.encoder_max_len
        if cfg.conv_kernel_sizes:
            conv_input_dim = cfg.conv_input_dim or input_dim
            conv_channels = cfg.conv_channels or conv_input_dim
            if conv_input_dim != input_dim:
                self.in_linear = nn.Sequential(
                    modules.WrappableLinear(input_dim, cfg.conv_input_dim),
                    nn.ReLU(),
                )
            self.subsample = modules.ConvolutionSubsampler(
                conv_input_dim,
                conv_channels,
                cfg.embed_dim,
                cfg.conv_kernel_sizes,
                cfg.conv_strides,
                cfg.conv_activation,
            )
            # the positional encoding is done after the convolutions: recompute encoder_max_len accordingly
            encoder_max_len = self.subsample.get_new_length(encoder_max_len).item()
        elif input_dim != cfg.embed_dim:
            self.in_linear = modules.WrappableLinear(input_dim, cfg.embed_dim)

        self.embed_positions = modules.PositionalEmbedding(
            cfg.encoder_positional_encoding,
            encoder_max_len, cfg.embed_dim,
            shift=cfg.positional_encoding_shift,
        )
        self.embed_scale = math.sqrt(cfg.embed_dim) if self.cfg.scale_embed else 1
        Norm = (
            modules.WrappableRMSNorm if cfg.rms_norm else
            modules.WrappableLayerNorm if cfg.norm_bias else
            modules.LayerNormWithoutBias
        )
        self.layernorm_embedding = Norm(cfg.embed_dim, eps=cfg.norm_eps) if cfg.encoder_embed_norm else Identity()
        self.dropout = nn.Dropout(defined(cfg.embed_dropout, cfg.dropout))
        self.layers = nn.ModuleList([])
        for layer_id in range(cfg.encoder_layers):
            self.layers.append(self.build_layer(layer_id))
        
        if cfg.encoder_positional_encoding == 't5':  # T5 relative position parameters are tied across all layers
            for layer in self.layers[1:]:
                layer.self_attn.t5_embed = self.layers[0].self_attn.t5_embed
        
        self.layer_norm = Norm(cfg.embed_dim, eps=cfg.norm_eps) if cfg.encoder_prenorm else Identity()
        self.device = None  # used for pipeline parallelism at inference

    def build_layer(self, layer_id: int) -> 'TransformerEncoderLayer':
        """
        Creates a Transformer layer (at a given depth) and returns it. This method can be overriden to implement 
        different architectures.
        """
        layer = TransformerEncoderLayer(self.cfg, self.dist_cfg, layer_id)
        return modules.checkpoint_wrapper(layer, activate=self.cfg.checkpoint_activations)

    @utils.benchmark('encoder')
    def forward(
        self,
        source: Tensor,
        source_length: LongTensor,
        return_layers: list[str] = [],
        meta: dict = {},  # unused here, but may be used in subclasses
        **kwargs,  # unused
    ) -> tuple[Tensor, BoolTensor, dict]:
        """
        Args:
            source: batch of token or feature sequences
            source_length: length of each sequence in the batch
            return_layers: tuple of strings specifying which layer outputs to return (e.g., 'enc_0_self_attn')
            meta: metadata about this batch (source and target languages, domain, etc.)
        Shape:
            source: (B, S) or (B, S, D)
            source_length: (B,)
        
        Returns: a tuple (encoder_out, padding_mask, layer_outputs) with
            encoder_out: a Tensor containing the output of the last encoder layer
            padding_mask: a boolean mask with True at every padding position in encoder_out
            layer_outputs: a dictionary of layer names and layer outputs, tensors of shape (B, S, ...) which can be used
                by `compute_loss`, for logging or as decoding outputs
        """
        return_layers = return_layers or ()

        x = (
            self.embed_tokens(source) if source.ndim == 2
            else source   # input is already binary (e.g., speech features)
        )
        # `embed_tokens` might not be on self.device is embeddings are shared between encoder and decoder
        # in this case, `source` is automatically put on the right device by `embed_tokens`
        
        x = x.to(self.device)
        source_length = source_length.to(self.device)
        if self.in_linear is not None:
            x = self.in_linear(x)
        if self.subsample is not None:
            x, source_length = self.subsample(x, source_length)

        x *= self.embed_scale
        padding_mask = utils.len_to_mask(source_length, size=x.size(1))  # BxS
        x += self.embed_positions(x.size(1))

        x = self.layernorm_embedding(x)
        x = self.dropout(x)
        layer_outputs = {}
        for layer in self.layers:
            x, layer_output = layer(x, padding_mask, return_layers)
            layer_outputs.update(layer_output)
        
        x = x.to(self.device)
        x = self.layer_norm(x)  # BxSxD
        return x, padding_mask, layer_outputs

    def parallelize(self, devices: list[str]) -> None:
        for layer_id, layer in enumerate(self.layers):
            device_id = (layer_id * len(devices)) // len(self.layers)
            device = devices[device_id]
            logger.debug(f'Encoder layer {layer_id} --> {device}')
            utils.move_to_device(layer, device)
        
        self.device = devices[0]
        for name, module in self.named_children():
            if name != 'layers':
                utils.move_to_device(module, self.device)


class TransformerDecoder(Decoder):
    def __init__(
        self,
        cfg: TransformerConfig,
        dist_cfg: DistributedConfig,
        task: Task,
        embed: Optional[Embedding] = None,
    ):
        """
        Args:
            cfg: model hyper-parameters
            embed: optional existing Embedding to reuse
        """
        super().__init__()
        self.cfg = cfg
        self.dist_cfg = dist_cfg
        self.task = task

        if embed is None:
            self.embed_tokens = Embedding(task.decoder_num_embeddings, cfg.embed_dim, self.padding_idx)
        else:
            self.embed_tokens = embed
        
        self.embed_positions = modules.PositionalEmbedding(
            cfg.decoder_positional_encoding,
            cfg.decoder_max_len, cfg.embed_dim,
            shift=cfg.positional_encoding_shift,
        )
        self.embed_scale = math.sqrt(cfg.embed_dim) if self.cfg.scale_embed else 1
        self.dropout = nn.Dropout(defined(cfg.embed_dropout, cfg.decoder_dropout, cfg.dropout))
        
        self.layers = nn.ModuleList([])
        for layer_id in range(cfg.decoder_layers):
            self.layers.append(self.build_layer(layer_id))
        
        if cfg.decoder_positional_encoding == 't5':  # T5 relative position parameters are tied across all layers
            for layer in self.layers[1:]:
                layer.self_attn.t5_embed = self.layers[0].self_attn.t5_embed
        
        Norm = (
            modules.WrappableRMSNorm if cfg.rms_norm else
            modules.WrappableLayerNorm if cfg.norm_bias else
            modules.LayerNormWithoutBias
        )
        self.layernorm_embedding = Norm(cfg.embed_dim, eps=cfg.norm_eps) if cfg.decoder_embed_norm else Identity()
        self.layer_norm = Norm(cfg.embed_dim, eps=cfg.norm_eps) if cfg.decoder_prenorm else Identity()
        if cfg.tied_output_projection:
            self.output_projection = None
        else:
            self.output_projection = modules.WrappableLinear(  # FIXME: LoRA does not apply on this, nor on the embedings
                cfg.embed_dim, self.task.decoder_num_embeddings, bias=False,
            )
            nn.init.xavier_uniform_(self.output_projection.weight)
        self.device = None  # used for pipeline parallelism at inference

    def build_layer(self, layer_id: int) -> 'TransformerDecoderLayer':
        """
        Creates a Transformer layer (at a given depth) and returns it. This method can be overriden to implement 
        different architectures.
        """
        layer = TransformerDecoderLayer(self.cfg, self.dist_cfg, layer_id)
        return modules.checkpoint_wrapper(layer, activate=self.cfg.checkpoint_activations)

    @utils.benchmark('decoder')
    def forward(
        self,
        encoder_out: Tensor,
        encoder_mask: BoolTensor,
        decoder_input: LongTensor,
        prompt_mask: Optional[Tensor] = None,
        state: Optional[dict[str, Tensor]] = None,
        return_layers: list[str] = [],
        meta: dict = {},  # unused here, but may be used in subclasses
        **kwargs,  # unused
    ) -> tuple[Tensor, dict]:
        """
        Args:
            encoder_out: encoder states returned by `TransformerEncoder.forward`
            encoder_mask: padding mask returned by `TransformerEncoder.forward`
            decoder_input: batch of tokens to use as input to the decoder, target sequences shifted by one position 
                when training (with teacher forcing) or previously generated tokens when decoding (generally, S=1)
            state: incremental decoding state (`decoder_input` should only contain the tokens generated at the latest
                timestep)
            return_layers: tuple of strings specifying which layer outputs to return (e.g., 'dec_0_cross_attn')
            meta: metadata about this batch (source and target languages, domain, etc.)
        Shape:
            encoder_out: (B, S, D)
            encoder_mask: (B, S)
            decoder_input: (B, T)
        
        Note that `decoder_input` usually starts with BOS (except at later decoding timesteps).

        Returns: a tuple (decoder_out, layer_outputs) with
            decoder_out: (B, T, D)
            layer_outputs: a dictionary of layer names and layer outputs, tensors of shape (B, T, ...) which can be used
                by `compute_loss`, for logging or as decoding outputs
        """
        return_layers = return_layers or ()
        decoder_input = decoder_input.to(self.device)
        padding_mask = decoder_input.eq(self.padding_idx)
        
        if self.cfg.disable_bos:
            # This is useful for the BLOOM models, which do not use any beginning of sequence (we replace it with 
            # padding). TODO: tried at decoding but not at training. It would probably be good to pad the target as
            # well (we don't want to predict the first token from padding, but rather predict the second token from the
            # first)
            decoder_input[decoder_input == self.bos_idx] = self.padding_idx

        length = decoder_input.size(1)
        pos_offset = state.get('offset', 0) if state else 0
        pos_embed = self.embed_positions(length, offset=pos_offset)
        if state is not None:
            state['offset'] = pos_offset + length

        x = self.embed_tokens(decoder_input)
        x *= self.embed_scale
        x += pos_embed
        x = self.layernorm_embedding(x)
        x = self.dropout(x)

        layer_outputs = {}
        for layer in self.layers:
            x, layer_output = layer(x, encoder_out, encoder_mask, padding_mask, prompt_mask, state, return_layers)
            layer_outputs.update(layer_output)

        x = x.to(self.device)
        x = self.layer_norm(x)
        with utils.benchmark('output_projection'):
            if self.output_projection is None:
                x = self.embed_tokens.projection(x)
            else:
                x = self.output_projection(x)
        
        return x, layer_outputs
    
    def parallelize(self, devices: list[str]) -> None:
        for layer_id, layer in enumerate(self.layers):
            device_id = (layer_id * len(devices)) // len(self.layers)
            device = devices[device_id]
            logger.debug(f'Decoder layer {layer_id} --> {device}')
            utils.move_to_device(layer, devices[device_id])
        
        self.device = devices[-1]
        for name, module in self.named_children():
            if name != 'layers':
                utils.move_to_device(module, self.device)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig, dist_cfg: DistributedConfig, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.dist_cfg = dist_cfg
        self.layer_id = layer_id

        shard_id = dist_cfg.tp_rank or 0
        shard_count = dist_cfg.tp_size or 1
        
        self.self_attn = modules.MultiheadAttention(
            cfg.embed_dim,
            cfg.encoder_attention_heads,
            dropout=cfg.attention_dropout,
            shard_id=shard_id,
            shard_count=shard_count,
            positional_encoding=cfg.encoder_positional_encoding,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            has_bias=cfg.has_bias,
            key_bias=cfg.attention_key_bias,
            layer_id=layer_id,
            scaled=cfg.scale_attn,
            rope_base=cfg.rope_base,
            alibi_max_bias=cfg.alibi_max_bias,
            max_qkv=cfg.max_qkv,
        )

        Norm = (
            modules.RMSNorm if cfg.rms_norm else
            nn.LayerNorm if cfg.norm_bias else
            modules.LayerNormWithoutBias
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.self_attn_layer_norm = Norm(cfg.embed_dim, eps=cfg.norm_eps)
        self.prenorm = cfg.encoder_prenorm
        
        ffn_dim = cfg.encoder_ffn_dim // shard_count
        
        self.fc1 = modules.Linear(
            cfg.embed_dim,
            ffn_dim,
            bias=cfg.has_bias,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
        )
        self.fc2 = modules.Linear(
            ffn_dim,
            cfg.embed_dim,
            bias=cfg.has_bias and shard_id == 0,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
        )
        self.fc3 = modules.Linear(
            cfg.embed_dim,
            ffn_dim,
            bias=cfg.has_bias,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
        ) if cfg.activation_fn in ('swiglu', 'geglu') else None

        self.activation_fn = modules.get_activation_fn(cfg.activation_fn)

        self.activation_dropout = nn.Dropout(cfg.activation_dropout)
        if cfg.shared_norm:
            self.final_layer_norm = lambda x: self.self_attn_layer_norm(x)
        else:
            self.final_layer_norm = Norm(cfg.embed_dim, eps=cfg.norm_eps)
        
        self.set_name(f'enc_{layer_id}')
        self.return_layers = []
        self.layer_outputs = {}

        if shard_count > 1:
            # tensor parallelism like the Megatron-LM paper
            self.enter = modules.MegatronLMEnter.apply
            self.exit = modules.MegatronLMExit.apply
        else:
            self.enter = self.exit = nn.Identity()

        self.device = None

    def set_name(self, name: str):
        self.name = name
        self.self_attn_key = f'{self.name}_self_attn'

    def ffn(
        self,
        x: Tensor,
        residual: Tensor,  # unused here, but may be used in subclasses
        padding_mask: BoolTensor,  # unused here, but may be used in subclasses
    ) -> Tensor:
        """
        Shape:
            x: (B, S, D)
            residual: (B, S, D)
            padding_mask: (B, S)
        """
        x = self.enter(x)
        y = self.fc1(x)
        y = self.activation_fn(y)
        y = self.activation_dropout(y)
        if self.fc3 is not None:
            y = y * self.fc3(x)
        x = self.fc2(y)
        x = self.exit(x)
        return x

    def self_attention(
        self,
        x: Tensor,
        residual: Tensor,  # unused here, but may be used in subclasses
        padding_mask: BoolTensor,
    ) -> Tensor:
        """
        Shape:
            x: (B, S, D)
            residual: (B, S, D)
            padding_mask: (B, S)
        """
        x = self.enter(x)
        padding_mask = self.enter(padding_mask)
        return_attn = self.self_attn_key in self.return_layers
        x, self_attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=padding_mask, return_attn=return_attn)
        x = self.exit(x)
        if return_attn:
            self_attn = self.exit(self_attn)
            self.layer_outputs[self.self_attn_key] = self_attn
        return x
    
    def self_attn_residual(self, x: Tensor, residual: Tensor) -> Tensor:
        return residual + self.dropout(x)
    def self_attn_prenorm(self, x: Tensor) -> Tensor:
        return self.self_attn_layer_norm(x) if self.prenorm else x
    def self_attn_postnorm(self, x: Tensor) -> Tensor:
        return x if self.prenorm else self.self_attn_layer_norm(x)
    def ffn_residual(self, x: Tensor, residual: Tensor) -> Tensor:
        return residual + self.dropout(x)
    def ffn_prenorm(self, x: Tensor) -> Tensor:
        return self.final_layer_norm(x) if self.prenorm else x
    def ffn_postnorm(self, x: Tensor) -> Tensor:
        return x if self.prenorm else self.final_layer_norm(x)

    def forward(
        self,
        x: Tensor,
        padding_mask: BoolTensor,
        return_layers: list[str] = [],
        /,  # all above arguments are positional-only
    ) -> tuple[Tensor, dict]:
        """
        Shape:
            x: (B, S, D)
            padding_mask: (B, S)
        """
        x = x.to(self.device)
        padding_mask = padding_mask.to(self.device)

        self.return_layers = return_layers
        
        residual = x
        x = self.self_attn_prenorm(x)
        x = self.self_attention(x, residual, padding_mask)
        x = self.self_attn_residual(x, residual)
        x = self.self_attn_postnorm(x)
        if self.cfg.check_inf:
            # T5 was trained with this and it doesn't work in float16 without it
            x = modules.clamp(x)

        residual = x
        x = self.ffn_prenorm(x)
        x = self.ffn(x, residual, padding_mask)
        x = self.ffn_residual(x, residual)
        x = self.ffn_postnorm(x)
        if self.cfg.check_inf:
            x = modules.clamp(x)

        layer_outputs = self.layer_outputs
        # These two attributes are only updated and used during this call to forward(), to reduce the verbosity of
        # function definitions and calls
        self.layer_outputs = {}
        self.return_layers = []

        if self.name in return_layers:
            layer_outputs[self.name] = x

        return x, layer_outputs


class TransformerDecoderLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig, dist_cfg: DistributedConfig, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.dist_cfg = dist_cfg
        self.layer_id = layer_id
        
        shard_id = dist_cfg.tp_rank or 0
        shard_count = dist_cfg.tp_size or 1
        
        self.self_attn = modules.MultiheadAttention(
            cfg.embed_dim,
            cfg.decoder_attention_heads,
            kv_heads=cfg.attention_heads_kv,
            sliding_window=cfg.sliding_window,
            dropout=cfg.attention_dropout,
            shard_id=shard_id,
            shard_count=shard_count,
            positional_encoding=cfg.decoder_positional_encoding,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            max_len=cfg.decoder_max_len,
            causal=True,
            has_bias=cfg.has_bias,
            key_bias=cfg.attention_key_bias,
            layer_id=layer_id,
            scaled=cfg.scale_attn,
            rope_base=cfg.rope_base,
            alibi_max_bias=cfg.alibi_max_bias,
            max_qkv=cfg.max_qkv,
        )

        Norm = (
            modules.RMSNorm if cfg.rms_norm else
            nn.LayerNorm if cfg.norm_bias else
            modules.LayerNormWithoutBias
        )
        self.dropout = nn.Dropout(defined(cfg.decoder_dropout, cfg.dropout))
        self.self_attn_layer_norm = Norm(cfg.embed_dim, eps=cfg.norm_eps)
        self.prenorm = cfg.decoder_prenorm

        if self.cfg.model_type == 'decoder':
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = modules.MultiheadAttention(
                cfg.embed_dim,
                cfg.decoder_attention_heads,
                dropout=cfg.attention_dropout,
                shard_id=shard_id,
                shard_count=shard_count,
                lora_rank=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                has_bias=cfg.has_bias,
                key_bias=cfg.attention_key_bias,
                layer_id=layer_id,
                scaled=cfg.scale_attn,
                max_qkv=cfg.max_qkv,
            )
            self.encoder_attn_layer_norm = Norm(cfg.embed_dim, eps=cfg.norm_eps)

        ffn_dim = cfg.decoder_ffn_dim // shard_count
        
        self.fc1 = modules.Linear(
            cfg.embed_dim,
            ffn_dim,
            bias=cfg.has_bias,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
        )
        self.fc2 = modules.Linear(
            ffn_dim,
            cfg.embed_dim,
            bias=cfg.has_bias and shard_id == 0,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
        )
        
        self.fc3 = modules.Linear(  # Llama or T5
            cfg.embed_dim,
            ffn_dim,
            bias=cfg.has_bias,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
        ) if cfg.activation_fn in ('swiglu', 'geglu') else None

        if cfg.activation_fn in ('gelu_tanh', 'geglu'):
            self.activation_fn = nn.GELU(approximate='tanh')   # BLOOM
        elif cfg.activation_fn == 'gelu':
            self.activation_fn = nn.GELU(approximate='none')   # MPT, FalconLLM
        elif cfg.activation_fn == 'swiglu':
            self.activation_fn = nn.SiLU()  # Llama
        else:
            self.activation_fn = nn.ReLU()
        
        if shard_count > 1:  # we don't want tensor parallelism to affect how weights are initialized
            self.fc2.weight.data /= shard_count**0.5
            if self.fc2.bias is not None:
                self.fc2.bias.data /= shard_count**0.5

        self.activation_dropout = nn.Dropout(cfg.activation_dropout)
        if cfg.shared_norm:
            self.final_layer_norm = lambda x: self.self_attn_layer_norm(x)
        else:
            self.final_layer_norm = Norm(cfg.embed_dim, eps=cfg.norm_eps)

        self.set_name(f'dec_{layer_id}')
        self.return_layers = []
        self.layer_outputs = {}
        
        if shard_count > 1:  # Tensor Parallelism as in the Megatron-LM paper: https://arxiv.org/abs/1909.08053
            self.enter = modules.MegatronLMEnter.apply
            self.exit = modules.MegatronLMExit.apply
        else:
            self.enter = self.exit = nn.Identity()

        self.device = None

    def set_name(self, name: str):
        self.name = name
        self.self_attn_key = f'{self.name}_self_attn'
        self.cross_attn_key = f'{self.name}_cross_attn'

    def ffn(
        self,
        x: Tensor,
        residual: Tensor,  # unused here, but may be used in subclasses
        padding_mask: BoolTensor,  # unused here, but may be used in subclasses
    ) -> Tensor:
        """
        Shape:
            x: (B, T, D)
            residual: (B, T, D)
            padding_mask: (B, T)
        """
        x = self.enter(x)
        y = self.fc1(x)
        y = self.activation_fn(y)
        y = self.activation_dropout(y)
        if self.fc3 is not None:
            y = y * self.fc3(x)
        x = self.fc2(y)
        x = self.exit(x)
        return x
    
    def self_attention(
        self,
        x: Tensor,
        residual: Tensor,  # unused here, but may be used in subclasses
        padding_mask: BoolTensor,
        state: Optional[dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Shape:
            x: (B, T, D)
            residual: (B, T, D)
            padding_mask: (B, S)
        """
        x = self.enter(x)
        
        if state is not None:
            self_attn_prefix = f'{self.self_attn_key}_'
            self_attn_state = {
                k.removeprefix(self_attn_prefix): v
                for k, v in state.items()
                if k.startswith(self_attn_prefix)
            }
        else:
            self_attn_state = None
        
        return_attn = self.self_attn_key in self.return_layers

        x, self_attn = self.self_attn(
            query=x, key=x, value=x,
            state=self_attn_state,
            return_attn=return_attn,
        )
        
        x = self.exit(x)

        if return_attn:
            self_attn = self.exit(self_attn)
            self.layer_outputs[self.self_attn_key] = self_attn

        if self_attn_state:
            state.update({f'{self_attn_prefix}{k}': v for k, v in self_attn_state.items()})

        return x
    
    def cross_attention(
        self,
        x: Tensor,
        residual: Tensor,  # unused here, but may be used in subclasses
        encoder_out: Tensor,
        encoder_mask: BoolTensor,
    ) -> Tensor:
        """
        Shape:
            x: (B, T, D)
            residual: (B, T, D)
            encoder_out: (B, S, D)
            encoder_mask: (B, S)
        """
        x = self.enter(x)
        encoder_out = self.enter(encoder_out)
        encoder_mask = self.enter(encoder_mask)
        return_attn = self.cross_attn_key in self.return_layers
        x, cross_attn = self.encoder_attn(
            query=x, key=encoder_out, value=encoder_out,
            key_padding_mask=encoder_mask,
            return_attn=return_attn,
        )
        x = self.exit(x)
        if return_attn:
            cross_attn = self.exit(cross_attn)
            self.layer_outputs[self.cross_attn_key] = cross_attn
        return x

    def self_attn_residual(self, x: Tensor, residual: Tensor) -> Tensor:
        return residual + self.dropout(x)
    def self_attn_prenorm(self, x: Tensor) -> Tensor:
        return self.self_attn_layer_norm(x) if self.prenorm else x
    def self_attn_postnorm(self, x: Tensor) -> Tensor:
        return x if self.prenorm else self.self_attn_layer_norm(x)
    def cross_attn_residual(self, x: Tensor, residual: Tensor) -> Tensor:
        return residual + self.dropout(x)
    def cross_attn_prenorm(self, x: Tensor) -> Tensor:
        return self.encoder_attn_layer_norm(x) if self.prenorm else x
    def cross_attn_postnorm(self, x: Tensor) -> Tensor:
        return x if self.prenorm else self.encoder_attn_layer_norm(x)
    def ffn_residual(self, x: Tensor, residual: Tensor) -> Tensor:
        return residual + self.dropout(x)
    def ffn_prenorm(self, x: Tensor) -> Tensor:
        return self.final_layer_norm(x) if self.prenorm else x
    def ffn_postnorm(self, x: Tensor) -> Tensor:
        return x if self.prenorm else self.final_layer_norm(x)

    def forward(
        self,
        x: Tensor,
        encoder_out: Optional[Tensor],
        encoder_mask: Optional[BoolTensor],
        padding_mask: BoolTensor,
        prompt_mask: Optional[BoolTensor] = None,
        state: Optional[dict[str, Tensor]] = None,
        return_layers: list[str] = [],
        /,  # all above arguments are positional-only
    ) -> tuple[Tensor, dict]:
        """
        Shape:
            x: (B, T, D)
            encoder_out: (B, S, D)
            encoder_mask: (B, S)
            padding_mask: (B, T)
        """
        self.return_layers = return_layers

        x = x.to(self.device)
        padding_mask = padding_mask.to(self.device)

        if encoder_out is not None:
            encoder_out = encoder_out.to(self.device)
        if encoder_mask is not None:
            encoder_mask = encoder_mask.to(self.device)
        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(self.device)

        residual = x
        x = self.self_attn_prenorm(x)
        x = self.self_attention(x, residual, padding_mask, state=state)

        if self.cfg.parallel_attention:
            # FFN and ATTN blocks take the same inputs and can be computed in parallel. The residual connexion
            # is only applied once to the sum of their outputs
            assert self.cfg.model_type == 'decoder'
            assert self.cfg.decoder_prenorm
            y = self.ffn_prenorm(residual)
            y = self.ffn(y, residual, padding_mask)
            x = self.ffn_residual(x + y, residual)
            layer_outputs = self.layer_outputs
            self.layer_outputs = {}
            return x, layer_outputs

        x = self.self_attn_residual(x, residual)
        x = self.self_attn_postnorm(x)
        if self.cfg.check_inf:
            x = modules.clamp(x)
        
        if not self.cfg.model_type == 'decoder':
            residual = x
            x = self.cross_attn_prenorm(x)
            x = self.cross_attention(x, residual, encoder_out, encoder_mask)
            x = self.cross_attn_residual(x, residual)
            x = self.cross_attn_postnorm(x)
            if self.cfg.check_inf:
                x = modules.clamp(x)

        residual = x
        x = self.ffn_prenorm(x)
        x = self.ffn(x, residual, padding_mask)
        x = self.ffn_residual(x, residual)
        x = self.ffn_postnorm(x)
        if self.cfg.check_inf:
            x = modules.clamp(x)

        layer_outputs = self.layer_outputs
        self.layer_outputs = {}

        if self.name in return_layers:
            layer_outputs[self.name] = x

        return x, layer_outputs
