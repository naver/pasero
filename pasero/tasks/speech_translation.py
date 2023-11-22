# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import numpy as np
import logging
from typing import Any
from pasero.config import TranslationTaskConfig
from pasero.tasks import Task, TranslationTask, ParallelCorpus, InferenceParallelCorpus

logger = logging.getLogger('speech_translation')


class SpeechTranslationTask(TranslationTask):

    def __init__(self, data_dir: str, cfg: TranslationTaskConfig):
        super().__init__(data_dir, cfg)
        self.src_tokenizer = None  # FIXME: it is unnecessarily initialized by the call above

    def set_model_type(self, model_type: str) -> None:
        assert model_type == 'encoder_decoder'
        super().set_model_type(model_type)

    @property
    def inference_options(self) -> dict:
        return {**super().inference_options, 'task': 'speech_translation'}

    def input_to_sample(self, input: np.ndarray, meta: dict) -> dict:
        return {'source': input, 'target': None, 'meta': meta}

    @property
    def encoder_num_embeddings(self):
        return 0

    @property
    def decoder_num_embeddings(self):
        return self.tgt_preprocessor.num_symbols

    @property
    def preprocessors(self):
        return {'target': self.tgt_preprocessor}

    def log_sample(self, sample_bin: dict) -> None:
        target_tok = self.tgt_preprocessor.debinarize(sample_bin['target'])
        corpus_id = sample_bin['meta']['corpus_id']
        logger.debug(f'{corpus_id} | line example: {target_tok}')

    def preprocess(
        self,
        sample: dict[str, Any],
        truncate: bool = False,
        tokenize: bool = True,
        inference: bool = False,
    ) -> dict[str, Any]:
        source, target = sample['source'], sample['target']
        meta = sample['meta']
        source_cutoff = self.max_source_len if truncate else None
        target_cutoff = self.max_target_len if truncate else None

        # find the target-side tags and prompt length by tokenizing an empty line
        tags = self.get_target_tags(meta)
        prompt_len = len(tags)

        source_bin = source[:source_cutoff]

        target_tok = list(tags)
        if target is None:
            pass
        elif tokenize:
            target_tok.append(self.tgt_preprocessor.tokenize(target))
        else:
            target_tok.append(target)
        target_tok = ' '.join(target_tok)
        
        target_bin = self.tgt_preprocessor.binarize(target_tok, max_len=target_cutoff)
        prompt_mask = np.zeros_like(target_bin, dtype=bool)
        prompt_mask[:prompt_len] = True

        if self.should_skip(len(source_bin), len(target_bin)):
            assert not truncate  # this shouldn't happen since we truncate
            return {}
        else:
            return {
                'source': source_bin,
                'target': target_bin,
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
