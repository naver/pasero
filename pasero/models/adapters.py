# adapters.py Copyright (C) 2023 Naver Corporation
#
# This file is licensed under the Creative Commons BY-NC-SA 
# (Attribution-NonCommercial-ShareAlike) 4.0 license ("License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License (i.e., CC BY-NC-SA 4.0) at:
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# You may obtain a summary of the License (i.e., CC BY-NC-SA 4.0) at:
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/
#

import logging
import torch.nn as nn

from torch import Tensor
from typing import Union, Optional
from pasero import utils
from pasero.tasks import Task
from pasero.config import register_model, AdapterTransformerConfig, DistributedConfig
from . import modules
from .modules import Embedding, AdapterLayer
from .transformer import Transformer, DummyEncoder
from .transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer


logger = logging.getLogger('adapters')


_LANG_PREFIX = 'lang:'
_DOMAIN_PREFIX = 'domain:'


@register_model('adapter_transformer')
class AdapterTransformer(Transformer):
    """
    Transformer with bottleneck adapters after each of its layers, with the same architecture as 
    'Simple, Scalable Adaptation for Neural Machine Translation', Bapna et al., 2019
    
    By default, adapters named "default" are added after all Transformer layers.

    Several sets of adapters may be included with the `--encoder-adapters-by` and `--decoder-adapters-by` options (and
    trained according to the languages or domain of the current batch).
    For instance, to train an MT model with language adapters:
    `--encoder-adapters-by source_lang --decoder-adapters-by target_lang`

    Some layers may be skipped with the `--encoder-adapter-layer-ids` and `--decoder-adapter-layer-ids` options.

    By default, the Transformer parameters are frozen and the adapter parameters are trained (usual setup for
    adapter training). The option `--train-all-params` can be used to also train the Transformer parameters.
    The option `--freeze-params-regex` overrides these two behaviors and lets the user define precisely which 
    parameters to train (e.g., the Transformer parameters and not the adapters).

    The `--encoder-adapters-by` and `--decoder-adapters-by` options are stored in the model configuration and 
    applied by default at inference. However, regular adapters are disabled by default and need to be activated
    manually with the `--encoder-adapters ADAPTER_NAME` and `--decoder-adapters ADAPTER_NAME` options.
    """

    def __init__(
        self,
        cfg: AdapterTransformerConfig,
        dist_cfg: DistributedConfig,
        task: Task,
    ):

        batch_by = set()
        find_unused_parameters = False

        if cfg.encoder_adapters is not None:
            self.encoder_adapter_names = cfg.encoder_adapters
        elif cfg.encoder_adapters_by:
            self.encoder_adapter_names = []
            for key in cfg.encoder_adapters_by:
                prefix = _DOMAIN_PREFIX if key == 'domain' else _LANG_PREFIX
                for value in sorted(task.get_langs_or_domains(key)):  # this sorting is very important, because 
                    # sets have a non-deterministic order, which can cause issues in distributed settings
                    adapter_name = f'{prefix}{value}'
                    self.encoder_adapter_names.append(adapter_name)
                batch_by.add(key)
                find_unused_parameters = True
        else:
            self.encoder_adapter_names = ['default']

        if cfg.decoder_adapters is not None:
            self.decoder_adapter_names = cfg.decoder_adapters
        elif cfg.decoder_adapters_by:
            self.decoder_adapter_names = []
            for key in cfg.decoder_adapters_by:
                prefix = _DOMAIN_PREFIX if key == 'domain' else _LANG_PREFIX
                for value in sorted(task.get_langs_or_domains(key)):
                    adapter_name = f'{prefix}{value}'
                    self.decoder_adapter_names.append(adapter_name)
                batch_by.add(key)
                find_unused_parameters = True
        else:
            self.decoder_adapter_names = ['default']

        batch_by = sorted(batch_by)

        # eliminate duplicate names
        # we do not do list(set(...)) because sets have no guaranteed order. It is important for
        # parallel training for modules to be created in the exact same order
        self.encoder_adapter_names = list(dict.fromkeys(self.encoder_adapter_names))
        self.decoder_adapter_names = list(dict.fromkeys(self.decoder_adapter_names))

        super().__init__(cfg, dist_cfg, task)
        
        # this should be done after super().__init__ (which would overwrite these values)
        self.batch_by = batch_by
        self.find_unused_parameters = find_unused_parameters

        if not cfg.train_all_params:
            # freeze all parameters that are not adapters
            for name, param in self.named_parameters():
                if 'adapters' not in name.split('.'):
                    param.requires_grad = False

        self.extra_adapters = {}  # holds adapters from the checkpoint that are not part of the model

    def build_encoder(
        self,
        embed: Optional[Embedding] = None,
    ) -> Union[DummyEncoder, 'AdapterTransformerEncoder']:
        if self.cfg.model_type == 'decoder':  # e.g., for language-modeling (see the 'OPT*' architectures)
            return DummyEncoder()
        else:
            return AdapterTransformerEncoder(
                self.cfg, self.dist_cfg, self.task, embed=embed,
                adapter_names=self.encoder_adapter_names,
            )
    
    def build_decoder(
        self,
        embed: Optional[Embedding] = None,
    ) -> 'AdapterTransformerDecoder':

        return AdapterTransformerDecoder(
            self.cfg, self.dist_cfg, self.task, embed=embed,
            adapter_names=self.decoder_adapter_names,
        )

    def update_state_dict(self, state_dict: dict) -> None:
        super().update_state_dict(state_dict)

        adapter_regex = r'(?P<module>encoder|decoder)\..*\.adapters\.(?P<uid>.*?)\.'
        
        # We don't want to throw an error when training new adapters: add the randomly initialized 
        # parameters to the state dict to avoid errors in super().load_state_dict
        if self.training:
            modules.add_missing_parameters(self, state_dict, adapter_regex)

        # If state_dict contains unused adapters, we don't want to raise an error as this is likely to happen in
        # incremental learning settings and at decoding time.
        # Instead, display a helpful message, store the unused adapters and include them in the future checkpoints.
        extra_adapters = modules.remove_unused_parameters(self, state_dict, adapter_regex)
        self.extra_adapters.update(extra_adapters)

    def clean_state_dict(self, state_dict: dict) -> None:
        super().clean_state_dict(state_dict)
        # Keep the adapters that were not used in this particular training instance. This is useful in continual
        # learning settings (e.g., when training several domain adapters or language pair adapters separately)
        state_dict.update(self.extra_adapters)


class AdapterTransformerEncoder(TransformerEncoder):
    cfg: AdapterTransformerConfig

    def __init__(self, *args, adapter_names: Optional[list[str]] = None, **kwargs):
        self.adapter_names = adapter_names
        super().__init__(*args, **kwargs)
    
    def build_layer(self, layer_id: int) -> nn.Module:
        if (
            (self.cfg.encoder_adapter_layer_ids is None or layer_id in self.cfg.encoder_adapter_layer_ids) and
            self.cfg.encoder_adapter_dim
        ):
            layer = AdapterTransformerEncoderLayer(self.cfg, self.dist_cfg, layer_id, self.adapter_names)
        else:
            layer = TransformerEncoderLayer(self.cfg, self.dist_cfg, layer_id)
        return modules.checkpoint_wrapper(layer, activate=self.cfg.checkpoint_activations)

    @utils.benchmark('encoder')
    def forward(self, *args, **kwargs):
        adapters_in_use = []
        for key in self.cfg.encoder_adapters_by:
            prefix = _DOMAIN_PREFIX if key == 'domain' else _LANG_PREFIX
            value = kwargs['meta'][key]
            adapter_name = f'{prefix}{value}'
            adapters_in_use.append(adapter_name)
        
        if adapters_in_use:
            for layer in self.layers:
                layer.adapters_in_use = adapters_in_use
        return super().forward(*args, **kwargs)


class AdapterTransformerDecoder(TransformerDecoder):
    cfg: AdapterTransformerConfig
    
    def __init__(self, *args, adapter_names: Optional[list[str]] = None, **kwargs):
        self.adapter_names = adapter_names
        super().__init__(*args, **kwargs)
    
    def build_layer(self, layer_id: int) -> nn.Module:
        if (
            (self.cfg.decoder_adapter_layer_ids is None or layer_id in self.cfg.decoder_adapter_layer_ids) and
            self.cfg.decoder_adapter_dim
        ):
            layer = AdapterTransformerDecoderLayer(self.cfg, self.dist_cfg, layer_id, self.adapter_names)
        else:
            layer = TransformerDecoderLayer(self.cfg, self.dist_cfg, layer_id)
        return modules.checkpoint_wrapper(layer, activate=self.cfg.checkpoint_activations)

    @utils.benchmark('decoder')
    def forward(self, *args, **kwargs):
        adapters_in_use = []
        for key in self.cfg.decoder_adapters_by:
            prefix = _DOMAIN_PREFIX if key == 'domain' else _LANG_PREFIX
            value = kwargs['meta'][key]
            adapter_name = f'{prefix}{value}'
            adapters_in_use.append(adapter_name)
        
        if adapters_in_use:
            for layer in self.layers:
                layer.adapters_in_use = adapters_in_use
        return super().forward(*args, **kwargs)


class AdapterTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(
        self,
        cfg: AdapterTransformerConfig,
        dist_cfg: DistributedConfig,
        layer_id: int,
        adapter_names: list[str],
    ):
        super().__init__(cfg, dist_cfg, layer_id)
        adapter_names = adapter_names or []
        self.adapters = nn.ModuleDict({
            uid: AdapterLayer(
                cfg.embed_dim,
                cfg.encoder_adapter_dim,
                zero_init=cfg.adapter_zero_init,
                activation_fn='relu',
            )
            for uid in adapter_names
        })
        self.adapters_in_use = adapter_names

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Shape:
            x: (B, S, D)
        
        Returns: tuple (adapter_out, layer_outputs) with
            adapter_out: tensor of shape (B, S, D)
            layer_outputs: a dictionary of layer names and layer outputs
        """
        x, layer_outputs = super().forward(x, *args, **kwargs)
        for adapter_uid in self.adapters_in_use:
            x = self.adapters[adapter_uid](x)
        return x, layer_outputs


class AdapterTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self,
        cfg: AdapterTransformerConfig,
        dist_cfg: DistributedConfig,
        layer_id: int,
        adapter_names: list[str],
    ):
        super().__init__(cfg, dist_cfg, layer_id)
        adapter_names = adapter_names or []
        self.adapters = nn.ModuleDict({
            uid: AdapterLayer(
                cfg.embed_dim,
                cfg.decoder_adapter_dim,
                zero_init=cfg.adapter_zero_init,
                activation_fn='relu',
            )
            for uid in adapter_names
        })
        self.adapters_in_use = adapter_names

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Shape:
            x: (B, T, D)
        
        Returns: tuple (adapter_out, layer_outputs) with
            adapter_out: tensor of shape (B, T, D)
            layer_outputs: a dictionary of layer names and layer outputs
        """
        x, layer_outputs = super().forward(x, *args, **kwargs)
        for adapter_uid in self.adapters_in_use:
            x = self.adapters[adapter_uid](x)
        return x, layer_outputs
