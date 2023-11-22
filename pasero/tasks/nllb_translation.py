import os
import torch
import logging
import json
import itertools
import re
from torch import nn
from typing import Optional
from pasero.config import NLLBTranslationTaskConfig, TransformerConfig
from pasero.tasks import TranslationTask

logger = logging.getLogger('nllb_translation')


class NLLBTranslationTask(TranslationTask):
    cfg: NLLBTranslationTaskConfig

    gate_regex = r'(?P<component>encoder|decoder)\.layers\.(?P<layer_id>\d+)\.moe_layer\.gate\.wg\.weight'

    def __init__(self, data_dir: str, cfg: NLLBTranslationTaskConfig):
        super().__init__(data_dir, cfg)
        self.prev_expert_ckpt = None
        self.gate_state_dict = None
        self.rank = self.world_size = None
    
    @staticmethod
    def expert_id_to_ckpt(expert_id: int) -> str:
        """
        Converts an expert id (integer between 0 and 1535) to a checkpoint name
        """
        layer_id = expert_id // 128 * 4 + 3
        expert_id = expert_id % 128
        if layer_id < 24:
            module = 'encoder'
        else:
            module = 'decoder'
            layer_id -= 24
        return f'{module}-{layer_id}-expert-{expert_id}.bin'

    def get_experts_for_lang_pair(self, source_lang: str, target_lang: str) -> list[str]:
        # an explicit list of expert checkpoints can be given via --expert-ckpt
        if self.cfg.expert_ckpt is not None:
            expert_ckpt = self.cfg.expert_ckpt
        # or a path to a JSON file with the list of expert checkpoints per language pair
        elif self.cfg.expert_json:
            assert self.cfg.expert_dir, '--expert-json requires --expert-dir (path to the expert checkpoints)'
            lang_pair = f'{source_lang}-{target_lang}'
            expert_index = json.load(open(self.cfg.expert_json))
            if lang_pair in expert_index:
                expert_ids = expert_index[lang_pair]
            else:
                expert_ids = (
                    [i for i in expert_index[source_lang] if i < 768] +
                    [i for i in expert_index[target_lang] if i >= 768]
                )
            expert_ckpt = [self.expert_id_to_ckpt(i) for i in expert_ids]
        else:
            return None
        
        if self.cfg.expert_dir:  # paths specified above are relative to --expert-dir
            expert_ckpt = [os.path.join(self.cfg.expert_dir, path) for path in expert_ckpt]
        
        return expert_ckpt

    def load_checkpoint_for_inference(
        self,
        *ckpt_paths: str,
        rank: int = 0,
        world_size: int = 1,
        arch: Optional[str] = None,
    ) -> tuple[dict, TransformerConfig]:
        
        if arch is None and (self.cfg.expert_ckpt or self.cfg.expert_json):
            arch = 'moe_transformer'

        model_state, model_cfg = super().load_checkpoint_for_inference(
            *ckpt_paths,
            rank=0,
            world_size=1,  # disable other types of sharding (TP or Tutel) since NLLB-200 has its own per-expert
            # sharding
            arch=arch,
        )
        self.gate_state_dict = {}
        for k in list(model_state):
            if re.fullmatch(self.gate_regex, k):
                self.gate_state_dict[k] = model_state.pop(k)
                # used in subsequent calls to "load_nllb_experts"
        self.rank = rank
        self.world_size = world_size

        expert_ckpt = self.get_experts_for_lang_pair(self.default_source_lang, self.default_target_lang)
        
        if expert_ckpt:
            expert_params, expert_args = self.load_nllb_experts(
                gate_state_dict=self.gate_state_dict,
                paths=expert_ckpt,
                rank=rank,
                world_size=world_size,
            )
            model_state.update(expert_params)
            model_cfg.parse_dict({**expert_args, 'moe_impl': 'tutel'})
            # FIXME: tutel is very verbose (it messes off with the logger, so all ranks end up logging)
            self.expert_args = expert_args
            self.prev_expert_ckpt = expert_ckpt
        
        return model_state, model_cfg

    def prepare_model_for_inference(self, model: nn.Module, meta: dict) -> None:
        source_lang = meta.get('source_lang')
        target_lang = meta.get('target_lang')

        assert source_lang is not None, 'source language is missing'
        assert target_lang is not None, 'target language is missing'

        assert self.gate_state_dict is not None, 'load_checkpoint_for_inference should be called before ' \
            'prepare_model_for_inference'

        expert_ckpt = self.get_experts_for_lang_pair(source_lang, target_lang)
        if expert_ckpt != self.prev_expert_ckpt:
            logger.info(f'updating the model experts for language pair: {source_lang}-{target_lang}')

            assert len(expert_ckpt) == len(self.prev_expert_ckpt), 'cannot update the model with a different expert ' \
                'count'
            expert_params, expert_args = self.load_nllb_experts(
                gate_state_dict=self.gate_state_dict,
                paths=expert_ckpt,
                rank=self.rank,
                world_size=self.world_size,
            )
            assert expert_args == self.expert_args, 'cannot update the model with a different expert count per layer'
            model.update_state_dict(expert_params)
            model.load_state_dict(expert_params, strict=False)  # update the model with the new experts
            self.prev_expert_ckpt = expert_ckpt

    @classmethod
    def load_nllb_experts(
        cls,
        gate_state_dict: dict,
        paths: list[dict],
        rank: int = 0,
        world_size: int = 1,
    ) -> tuple[dict, dict]:
        """
        Loads multiple NLLB-200 expert checkpoints and returns the new arguments that need to be added to the model
        config.

        `paths` should follow specific naming conventions: 'expert-EXPERT_ID.bin',
        or '(encoder|decoder)-expert-EXPERT_ID.bin' or '(encoder|decoder)-LAYER_ID-expert-EXPERT_ID.bin'

        The expert counts per layer are automatically updated (in `model_cfg.encoder_expert_count` and
        `model_cfg.decoder_expert_count` as dicts)

        If `world_size > 1`, experts in each layer are divided equally across all GPUs: i.e., this rank will load
        `1/world_size` of all experts.

        This is different from Tutel sharding/unsharding implemented in `mixtures_of_experts.py` or
        `scripts/merge-tutel-ckpt.py`, where each model shard contains all the dense parameters and 1/Nth of the expert
        parameters.
        """
        expert_ckpt = paths
        matches = [
            re.fullmatch(
                r'((?P<component>encoder|decoder)-((?P<layer_id>\d+)-)?)?expert-(?P<expert_id>\d+)\.bin',
                os.path.basename(path)
            )
            for path in expert_ckpt
        ]
        assert all(matches)
        experts = [
            (
                m.group('component'),
                (i := m.group('layer_id')) and int(i),   # layer_id can be None
                int(m.group('expert_id')),
            ) for m in matches
        ]
        experts = sorted(zip(experts, expert_ckpt))
        expert_keys = [key for key, _ in experts]   # can contain duplicates
        expert_paths = dict(experts)                # maps expert keys to checkpoint paths

        encoder_expert_ids = {}
        decoder_expert_ids = {}
        params = {}
        args = {}

        for (component, layer_id), keys in itertools.groupby(expert_keys, key=lambda p: p[:2]):
            keys = list(keys)
            expert_ids = [expert_id for _, _, expert_id in keys]
            if len(expert_ids) == 1:   # same expert across all nodes
                expert_ids = expert_ids * world_size
            if component == 'encoder' or component is None:
                encoder_expert_ids[layer_id] = expert_ids
            if component == 'decoder' or component is None:
                decoder_expert_ids[layer_id] = expert_ids
            
            ckpt_paths = [expert_paths[k] for k in keys]   # list of checkpoint names
            if len(ckpt_paths) == 1:
                ckpt_paths = ckpt_paths * world_size
            assert len(ckpt_paths) % world_size == 0
            experts_per_node = len(ckpt_paths) // world_size
            # identify which expert checkpoints should go to this rank
            ckpt_paths = ckpt_paths[rank * experts_per_node:(rank + 1) * experts_per_node]
            for ckpt_path in ckpt_paths:
                logger.info(f"loading checkpoint {ckpt_path}")
                params_ = torch.load(ckpt_path, map_location='cpu')
                if 'model' in params_:   # for back-compatibility with old checkpoints
                    params_ = params_['model']
                for key, value in params_.items():
                    params.setdefault(key, []).append(value)
        
        # encoder.layers.2.moe_layer.experts.0.fc1.weight * EXPERT_COUNT ->
        # encoder.layers.2.moe_layer.experts.EXPERT_ID.fc1.weight
        params = {
            name.replace('experts.0', f'experts.{expert_id}'): expert_weight
            for name, weights in params.items()
            for expert_id, expert_weight in enumerate(weights)
        }  # in the NLLB-200 expert checkpoints, all experts have the id "0", regardless of their rank

        if None in encoder_expert_ids:   # same expert ids for all layers
            args['encoder_expert_count'] = len(encoder_expert_ids[None])
        else:
            args['encoder_expert_count'] = {
                layer_id: len(expert_ids) for layer_id, expert_ids in encoder_expert_ids.items()
            }
        if None in decoder_expert_ids:
            args['decoder_expert_count'] = len(decoder_expert_ids[None])
        else:
            args['decoder_expert_count'] = {
                layer_id: len(expert_ids) for layer_id, expert_ids in decoder_expert_ids.items()
            }

        for name, value in gate_state_dict.items():
            if (m := re.fullmatch(cls.gate_regex, name)):
                component = m.group('component')
                layer_id = int(m.group('layer_id'))
                if component == 'encoder':
                    # if encoder_expert_ids contains None, then the same expert ids are shared across all encoder
                    # layers
                    expert_ids = encoder_expert_ids.get(None, []) or encoder_expert_ids.get(layer_id, [])
                else:
                    expert_ids = decoder_expert_ids.get(None, []) or decoder_expert_ids.get(layer_id, [])
                
                if expert_ids:
                    new_name = name.replace('gate.wg.weight', 'gate.weight')
                    params[new_name] = value[expert_ids]
        
        return params, args
