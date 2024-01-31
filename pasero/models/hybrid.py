# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import logging
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor, LongTensor, BoolTensor
from pasero import utils
from . import modules
from .modules import Embedding, Identity
from pasero.tasks import Task
from pasero.models.transformer import Transformer, Decoder
from pasero.models.adapters import AdapterTransformer
from pasero.utils import defined
from pasero.config import register_model, AdapterHybridTransformerConfig, DistributedConfig, HybridTransformerConfig


logger = logging.getLogger('models')


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, kdim: int):
        super().__init__()
        self.key_proj = modules.Linear(kdim, embed_dim)
        self.query_proj = modules.Linear(embed_dim, embed_dim)
        self.inner_proj = modules.Linear(embed_dim, 1)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: BoolTensor,
        state: Optional[dict] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Shape:
            query: (B, T, D)
            key: (B, S, D)
            value: (B, S, D)
            key_padding_mask: (B, S)
        """
        key = state['attn_key'] if state else self.key_proj(key).unsqueeze(dim=1)
        if state is not None:
            state['attn_key'] = key
        query = self.query_proj(query).unsqueeze(dim=2)
        scores = self.inner_proj(torch.tanh(key + query)).squeeze(-1)
        scores = (
            scores.float()
            .masked_fill_(key_padding_mask.unsqueeze(1), float("-inf"))
            .type_as(scores)
        )
        scores = F.softmax(scores, dim=2)
        attn = torch.bmm(scores, value)
        scores = scores.unsqueeze(2)  # BxTx1xS
        return attn, scores


@register_model('hybrid_transformer')
class HybridTransformer(Transformer):
    """
    Hybrid model with a Transformer encoder and LSTM decoder like in our "Efficient Inference for Multilingual NMT" 
    paper. The LSTM uses additive single-head attention over the last encoder state like in the
    "Learning to Align and Translate" paper, only at the first LSTM layer.
    """
    cfg: HybridTransformerConfig

    def build_decoder(self, embed: Optional[Embedding] = None) -> 'LSTMDecoder':
        return LSTMDecoder(self.cfg, self.dist_cfg, self.task, embed=embed)


@register_model('adapter_hybrid_transformer')
class AdapterHybridTransformer(AdapterTransformer):
    def build_decoder(self, embed: Optional[Embedding] = None) -> 'LSTMDecoder':
        return LSTMDecoder(self.cfg, self.dist_cfg, self.task, embed=embed)


class LSTMDecoderWithoutEmbed(Decoder):
    def __init__(
        self,
        cfg: AdapterHybridTransformerConfig,
        dist_cfg: DistributedConfig,
        task: Task,
    ):
        super().__init__()
        self.task = task
        self.cfg = cfg
        self.hidden_size = cfg.decoder_hidden_size
        self.embed_proj = (
            modules.Linear(cfg.embed_dim, cfg.embed_dim, bias=True) if cfg.decoder_embed_proj
            else Identity()
        )
        self.dropout = nn.Dropout(defined(cfg.decoder_dropout, cfg.dropout))
        self.layers = nn.ModuleList([
            nn.LSTM(
                input_size=cfg.embed_dim if layer == 0 else cfg.embed_dim + self.hidden_size,
                hidden_size=self.hidden_size,
                batch_first=True,
            )
            for layer in range(cfg.decoder_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(cfg.embed_dim if layer == 0 else cfg.embed_dim + self.hidden_size, eps=cfg.norm_eps)
            for layer in range(cfg.decoder_layers)
        ])
        self.attention = AttentionLayer(embed_dim=self.hidden_size, kdim=cfg.embed_dim)
        self.out_proj = (
            modules.Linear(self.hidden_size, cfg.embed_dim, bias=True) if self.hidden_size != cfg.embed_dim
            else Identity()
        )
        self.cross_attn_key = f'dec_{cfg.decoder_layers - 1}_cross_attn'

    @utils.benchmark('decoder')
    def forward(
        self,
        encoder_out: Tensor,
        encoder_mask: BoolTensor,
        decoder_input: Tensor,
        prompt_mask: Optional[BoolTensor] = None,
        state: Optional[dict[str, Tensor]] = None,
        return_layers: list[str] = [],
        **kwargs,  # unused
    ) -> tuple[Tensor, dict]:
        """
        Shape:
            encoder_out: (B, S, D)
            encoder_mask: (B, S)
            decoder_input: (B, T, D)
        
        Returns: a tuple (decoder_out, layer_outputs) with
            decoder_out: (B, T, D)
            layer_outputs: a dictionary of layer names and layer outputs, tensors of shape (B, T, ...)
        """
        decoder_input = self.embed_proj(decoder_input)

        if state:
            prev_hiddens = list(state['prev_hiddens'].unsqueeze(0).unbind(dim=2))
            prev_cells = list(state['prev_cells'].unsqueeze(0).unbind(dim=2))
        else:
            prev_hiddens = [None] * len(self.layers)
            prev_cells = [None] * len(self.layers)

        x = self.dropout(decoder_input)

        layer_outputs = {}
        attn = None
        for i, layer in enumerate(self.layers):
            residual = x
            hx = None if prev_hiddens[i] is None else (prev_hiddens[i].contiguous(), prev_cells[i].contiguous())
            input_ = x if attn is None else torch.cat([x, attn], dim=-1)
            input_ = self.layer_norms[i](input_)
            x, (h, c) = layer(input_, hx=hx)

            if i == 0:
                x = self.dropout(x)
                attn, attn_weights = self.attention(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_mask,
                    state=state,
                )
                if self.cross_attn_key in return_layers:
                    layer_outputs[self.cross_attn_key] = attn_weights
            else:
                x = self.dropout(x)
                x = residual + x
            
            prev_hiddens[i] = h.squeeze(0)
            prev_cells[i] = c.squeeze(0)

        if state is not None:
            state['prev_hiddens'] = torch.stack(prev_hiddens, dim=1)
            state['prev_cells'] = torch.stack(prev_cells, dim=1)

        x = self.out_proj(x) + attn
        return x, layer_outputs


class LSTMDecoder(LSTMDecoderWithoutEmbed):
    def __init__(
        self,
        cfg: AdapterHybridTransformerConfig,
        dist_cfg: DistributedConfig,
        task: Task,
        embed: Optional[Embedding] = None,
    ):
        super().__init__(cfg, dist_cfg, task)
        if embed is None:
            self.embed_tokens = Embedding(self.task.decoder_num_embeddings, cfg.embed_dim, self.padding_idx)
        else:
            self.embed_tokens = embed
        if cfg.tied_output_projection:
            self.output_projection = None
        else:
            self.output_projection = modules.Linear(cfg.embed_dim, self.task.decoder_num_embeddings, bias=False)
            nn.init.xavier_uniform_(self.output_projection.weight)

    @utils.benchmark('decoder')
    def forward(
        self,
        encoder_out: Tensor,
        encoder_mask: BoolTensor,
        decoder_input: LongTensor,
        state: Optional[dict[str, Tensor]] = None,
        return_layers: list[str] = [],
        meta: dict = {},
        **kwargs,  # unused
    ) -> tuple[Tensor, dict]:
        """
        Shape:
            encoder_out: (B, S, D)
            encoder_mask: (B, S)
            decoder_input: (B, T)
        
        Returns: a tuple (decoder_out, layer_outputs) with
            decoder_out: (B, T, D)
            layer_outputs: a dictionary of layer names and layer outputs, tensors of shape (B, T, ...)
        """
        x = self.embed_tokens(decoder_input)
        x, layer_outputs = super().forward(encoder_out, encoder_mask, x, state=state,
                                           return_layers=return_layers, **kwargs)
        with utils.benchmark('output_projection'):
            if self.output_projection is None:
                x = self.embed_tokens.projection(x)
            else:
                x = self.output_projection(x)
        return x, layer_outputs
