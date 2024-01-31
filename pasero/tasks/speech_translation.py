# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import numpy as np
import logging
import os
from typing import Any
from pasero.config import register_task, TranslationTaskConfig, TransformerConfig
from pasero.tasks import Task, TranslationTask, ParallelCorpus, InferenceParallelCorpus

logger = logging.getLogger('speech_translation')


@register_task('speech_translation')
class SpeechTranslationTask(TranslationTask):
    cfg: TranslationTaskConfig

    def setup_for_model(self, model_cfg: TransformerConfig) -> None:
        assert model_cfg.model_type == 'encoder_decoder'
        super().setup_for_model(model_cfg)

    @property
    def inference_options(self) -> dict:
        return {**super().inference_options, 'task': 'speech_translation'}

    def input_to_sample(self, input: np.ndarray, meta: dict = {}) -> dict:
        return {'source': input, 'target': None, 'meta': meta}

    @property
    def encoder_num_embeddings(self) -> int:
        return 0

    @property
    def preprocessors(self):
        return {'target': self.tgt_preprocessor}

    def log_sample(self, sample_bin: dict) -> None:
        # log with TranslationTask, but remove 'encoder_input' which is not text (TranslationTask handles this 
        # edge case for decoder-only translation)
        super().log_sample({k: v for k, v in sample_bin.items() if k != 'encoder_input'})

    def preprocess(
        self,
        sample: dict[str, Any],
        truncate: bool = False,
        tokenize: bool = True,
        append_eos: bool = False,
    ) -> dict[str, Any]:
        source, target = sample['source'], sample['target']
        meta = sample['meta']
        source_cutoff = self.max_source_len if truncate else None
        target_cutoff = self.max_target_len if truncate else None

        # find the target-side tags and prompt length by tokenizing an empty line
        tags = self.get_target_tags(meta)
        prompt_len = len(tags)

        encoder_input = source[:source_cutoff]

        target_tok = list(tags)
        if target is None:
            pass
        elif tokenize:
            target_tok += self.tgt_preprocessor.tokenize(target)
        else:
            target_tok += target.split()
        
        decoder_input = self.tgt_preprocessor.binarize(
            target_tok,
            max_len=target_cutoff,
            prepend_bos=self.prepend_bos,
            append_eos=append_eos,
        )
        prompt_mask = np.zeros_like(decoder_input, dtype=bool)
        prompt_mask[:prompt_len + int(self.prepend_bos)] = True

        if self.should_skip(len(encoder_input), len(decoder_input)):
            assert not truncate  # this shouldn't happen since we truncate
            return {}
        else:
            return {
                'encoder_input': encoder_input,
                'decoder_input': decoder_input,
                'prompt_mask': prompt_mask,
                'meta': meta,
            }

    def postprocess(
        self,
        sample_bin: dict[str, Any],
        hypothesis: dict[str, Any],
        detokenize: bool = True,
    ) -> None:
        Task.postprocess(self, sample_bin, hypothesis, detokenize=detokenize)

    @classmethod
    def _get_corpus(cls, *args, **kwargs) -> ParallelCorpus:
        corpus = super()._get_corpus(*args, **kwargs)
        corpus.file_formats = ['numpy', 'txt']
        return corpus

    @classmethod
    def get_inference_corpus(cls, *args, **kwargs) -> InferenceParallelCorpus:
        corpus = super().get_inference_corpus(*args, **kwargs)
        corpus.file_formats = ['numpy', 'txt']
        return corpus
