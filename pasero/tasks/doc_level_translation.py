# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import numpy as np
from itertools import zip_longest
from typing import Any, Optional, Union
from pasero.config import DocumentLevelTranslationTaskConfig
from pasero.tasks import TranslationTask, ParallelCorpus
from pasero.preprocessing import split_tags


class DocumentLevelTranslationTask(TranslationTask):
    def __init__(
        self,
        data_dir,
        cfg: DocumentLevelTranslationTaskConfig,
    ):
        super().__init__(data_dir, cfg)
        self.sent_merge_prob = cfg.sent_merge_prob
        self.max_doc_size = cfg.max_doc_size
        self.trailing_sent_sep = cfg.trailing_sent_sep
        self.sent_sep = cfg.sent_sep

        if self.sent_sep:
            sep_tokens = self.sent_sep.split()
            for dictionary in self.src_preprocessor.dictionary, self.tgt_preprocessor.dictionary:
                assert all(token in dictionary for token in sep_tokens), f'{self.sent_sep} is OOV'

    @property
    def inference_options(self) -> dict:
        return {
            **super().inference_options,
            'sent_sep': self.sent_sep,
            'task': 'doc_level_translation',
        }

    def get_reference(self, sample: dict[str, Any]):
        target_sents = self.split_sentences(sample['target'])
        return target_sents[-1]

    def input_to_sample(self, input: str, meta: dict) -> dict:
        source, *target = input.rsplit('|||', maxsplit=1)  # '|||' is used a separator between the source and an 
        # optional prompt
        source = self.split_sentences(source)
        target = self.split_sentences(target[0]) if target else None
        return {'source': source, 'target': target, 'meta': meta}
    
    def compute_score(
        self,
        metric: str,
        hypotheses: list[dict[str, Any]],
        references: list[str],
        merge_bpe: bool = False,
        **eval_opts,
    ) -> Optional[float]:
        # Only scores the last sentence in each hypothesis. References should already be single sentences if they
        # are obtained through `get_reference`
        hypotheses = list(hypotheses)
        for hyp in hypotheses:
            if self.sent_sep:
                detok = hyp['detok']
                tok = self.tgt_preprocessor.tokenize(detok)  # assumes that tokenization is reversible
                tok = tok.split(self.sent_sep)[-1]  # we need to tokenize before splitting because the separator
                # may be different once detokenized
                detok = self.tgt_preprocessor.detokenize(tok)
                hyp['detok'] = detok
                # we could use hyp['tokens'] directly instead of re-tokenizing hyp['detok'], but it may contain prompt
                # tokens that have already been removed in `postprocess`
        
        return super().compute_score(metric, hypotheses, references, merge_bpe=merge_bpe, **eval_opts)

    def add_separators(self, sents: list[str]):
        if not self.sent_sep:
            return sents
        sents_with_sep = []
        for i, sent in enumerate(sents):
            is_last_sent = (i == len(sents) - 1)
            if self.trailing_sent_sep or not is_last_sent:
                sent = f'{sent} {self.sent_sep}'
            sents_with_sep.append(sent)
        return sents_with_sep

    @classmethod
    def get_train_corpora(
        cls,
        cfg: DocumentLevelTranslationTaskConfig,
        data_dir: str,
        corpus_definitions: list[dict],
    ) -> list[ParallelCorpus]:
        corpora: list[ParallelCorpus] = []
        for corpus_definition in corpus_definitions:
            corpora += cls._get_corpora(
                data_dir,
                cfg.source_langs,
                cfg.target_langs,
                cfg.lang_pairs,
                corpus_definition,
                cfg.allow_monolingual,
                cfg.source_tags,
                cfg.target_tags,
            )
            # Validation corpora have max_doc_size=1. It is up to the user to make sure that they already have one
            # document per line, with a "<sep>" delimiter between sentences.
            # We do this because the "max_doc_size" approach is approximative and not fit for evaluation: it merges a 
            # random number of consecutive lines, regardless of whether or not they actually belong to the same
            # document.
            for corpus in corpora:
                corpus.max_doc_size = corpus_definition.get('max_doc_size', cfg.max_doc_size)
        assert len(set(corpus.corpus_id for corpus in corpora)) == len(corpora), 'there are duplicate corpus ' \
            'definitions'
        return corpora

    def split_sentences(self, doc_or_sent: Union[str, list[str]]) -> list[str]:
        if isinstance(doc_or_sent, str):
            sents = doc_or_sent.split('<sep>')  # we don't use --sent-sep here, which is only used 
            # as a separator *after* preprocessing
            # inputs should either be split already (by TrainingDataset), or contain "<sep>" delimiters
            return [sent.strip() for sent in sents]  # separator may have whitespaces around it ("sent1 <sep> sent2")
        else:
            return list(doc_or_sent)  # already split

    def preprocess(
        self,
        sample: dict[str, Any],
        truncate: bool = False,
        tokenize: bool = True,
        inference: bool = False,
    ) -> dict[str, Any]:
        source = sample.get('source')
        target = sample.get('target')
        meta = sample.get('meta')

        source_cutoff = self.max_source_len if truncate else None
        target_cutoff = self.max_target_len if truncate else None

        # The given pair may be a document of several consecutive sentences
        src_sents = self.split_sentences(source)
        tgt_sents = self.split_sentences(target) if target else []

        if self.training:
            assert len(src_sents) == len(tgt_sents)
        else:
            # at inference, no target prefix may be provided, or a variable number of target sentences to continue
            # decoding from
            assert len(src_sents) >= len(tgt_sents)

        # extract potential pre-existing prefix tags (can happen with datasets or user inputs that already contain
        # language codes)
        *src_tags, src_sents[0] = split_tags(src_sents[0])
        if tgt_sents:
            *tgt_tags, tgt_sents[0] = split_tags(tgt_sents[0])
        else:
            tgt_tags = []

        # prefix tokens and tags (e.g., language codes)
        src_tags += self.get_source_tags(meta) if tokenize else []
        tgt_tags += self.get_target_tags(meta) if tokenize else []
        prompt_len = len(tgt_tags)

        emojis = []
        if self.cfg.escape_emojis and not self.training:
            for i in range(len(src_sents)):
                src_sents[i], emojis_ = self.src_preprocessor.escape_emojis(src_sents[i])
                emojis += emojis_

        if self.sent_merge_prob and len(src_sents) > 1 and self.training:
            # randomly merge consecutive sentences before preprocessing them, effectively dropping separators or 
            # language codes between them
            merges = np.random.randint(0, 2, size=len(src_sents) - 1)
            for i, merge in enumerate(merges):
                if merge:
                    src_sents[i:i+2] = [None, ' '.join(src_sents[i:i+2])]
                    tgt_sents[i:i+2] = [None, ' '.join(tgt_sents[i:i+2])]
            src_sents = [sent for sent in src_sents if sent is not None]
            tgt_sents = [sent for sent in tgt_sents if sent is not None]
            assert len(src_sents) == len(tgt_sents)

        # tokenize all sentences
        src_sents = [self.src_preprocessor.tokenize(sent) for sent in src_sents]
        tgt_sents = [self.tgt_preprocessor.tokenize(sent) for sent in tgt_sents]
        # FIXME: check how whitespaces are handled, we may need to add a prefix whitespace when i > 0
        # FIXME: with some tokenizers (e.g., Llama's), this gives a different tokenization than when tokenizing
        # the concatenation of those sentences. For instance:
        # ['Hello', 'World'] with sep='<0x0A>' -> '▁Hello <0x0A> ▁World' -> 'Hello\n World'
        # The whitespace before "World" is impossible to avoid with the current solution, while we may wish to do:
        # 'Hello\nWorld' -> '▁Hello <0x0A> World' -> 'Hello\nWorld'
        # The same is true with BLOOM's tokenizer with --hf-add-prefix-space

        # prefixes will be added only to the first source or target sentence, for instance:
        # "Input: src1 <sep> src2" + "Output: tgt1 <sep> tgt2"
        src_sents[0] = ' '.join(src_tags + [src_sents[0]]).rstrip(' ')
        if tgt_sents:
            tgt_sents[0] = ' '.join(tgt_tags + [tgt_sents[0]]).rstrip(' ')
        else:
            tgt_sents = [' '.join(tgt_tags)]

        if self.cfg.copy_placeholder and self.training:
            for i in range(len(src_sents)):
                src_sents[i], tgt_sents[i] = self.copy_placeholder(src_sents[i], tgt_sents[i])

        sent_sep_len = len(self.sent_sep.split()) if self.sent_sep else 0

        # Truncate documents to respect max source and target length constraints
        # We always to this, regardless of the value of `truncate`
        assert sent_sep_len + 1 < self.max_source_len
        assert sent_sep_len + 1 < self.max_target_len
        if self.model_type == 'decoder':
            # make sure that there is always some space left for target tokens
            assert sent_sep_len + 1 + self.max_source_len < self.max_target_len, '--max-target-len should be ' \
                'higher than --max-source-len (with some margin)'

        src_sents_truncated, tgt_sents_truncated = [], []
        src_length = 0 if self.model_type == 'decoder' else 1  # EOS is not added to source when concatenating
        # source and target
        tgt_length = 1  # EOS

        for i, (src_sent, tgt_sent) in enumerate(zip_longest(src_sents, tgt_sents)):
            src_tokens = src_sent.split()
            tgt_tokens = tgt_sent.split() if tgt_sent is not None else []

            if i == 0:  # truncate the first sentence if it is too long
                max_src_tokens = self.max_source_len - sent_sep_len - src_length
                src_tokens_truncated = src_tokens[:max(0, max_src_tokens)]
                src_length += len(src_tokens_truncated) + sent_sep_len
                
                max_tgt_tokens = self.max_target_len - sent_sep_len - tgt_length  # includes sep that will be added
                # later and EOS
                if self.model_type == 'decoder':
                    max_tgt_tokens -= len(src_tokens_truncated)
                tgt_tokens_truncated = tgt_tokens[:max(0, max_tgt_tokens)]
                tgt_length += len(tgt_tokens_truncated) + sent_sep_len

                # if `truncate` is False, we keep the full sentence and it will be skipped in `TrainingDataset`
                src_sents_truncated.append(' '.join(src_tokens_truncated) if truncate else src_sent)
                tgt_sents_truncated.append(' '.join(tgt_tokens_truncated) if truncate else tgt_sent)
                
                truncated = (
                    len(src_tokens_truncated) < len(src_tokens) or
                    len(tgt_tokens_truncated) < len(tgt_tokens)
                )
                if truncated:  # if the first sentence was already too long, we drop the next sentences
                    break
            else:  # drop the next sentences if the document it too long
                src_length += len(src_tokens) + sent_sep_len
                tgt_length += len(tgt_tokens) + sent_sep_len
                tgt_len_concat = src_length + tgt_length if self.model_type == 'decoder' else tgt_length
                if src_length <= self.max_source_len and tgt_len_concat <= self.max_target_len:
                    src_sents_truncated.append(src_sent)
                    if tgt_sent is not None:
                        tgt_sents_truncated.append(tgt_sent)
                else:
                    break
            
        src_sents = src_sents_truncated
        tgt_sents = tgt_sents_truncated

        if self.model_type == 'decoder':  # source and target will be concatenated into a single 'target' sequence
            # Add the sentence separators. This is done after truncating, because we don't want to accidentally 
            # remove the separators.
            src_sents = self.add_separators(src_sents)
            # it is recommended to use --target-tags to have a separator between the last source sentence and the 
            # first target sentence
            tgt_sents = self.add_separators(tgt_sents)

            sources = []
            targets = []
            source_masks = []
            target_masks = []

            for i, src_sent in enumerate(src_sents):
                src_bin = self.src_preprocessor.binarize(src_sent, append_eos=False)
                sources.append(src_bin)
                src_mask = np.ones_like(src_bin, dtype=bool)
                source_masks.append(src_mask)

            for i, tgt_sent in enumerate(tgt_sents):
                is_last = (i == len(tgt_sents) - 1)
                tgt_bin = self.tgt_preprocessor.binarize(tgt_sent, append_eos=is_last)
                targets.append(tgt_bin)
                tgt_mask = np.zeros_like(tgt_bin, dtype=bool)
                if i == 0:  # mask the target prefixes
                    tgt_mask[:prompt_len] = True
                target_masks.append(tgt_mask)

            source_bin = np.concatenate(sources)
            target_bin = np.concatenate(targets)
            source_mask = np.concatenate(source_masks)
            target_mask = np.concatenate(target_masks)

            if self.should_skip(len(source_bin), len(source_bin) + len(target_bin)):
                assert not truncate  # this shouldn't happen since we truncate
                return {}
            else:
                return {
                    'target': np.concatenate([source_bin, target_bin]),
                    'prompt_mask': np.concatenate([source_mask, target_mask]),
                    'meta': meta,
                }
        else:
            src_sents = self.add_separators(src_sents)
            tgt_sents = self.add_separators(tgt_sents)
            source = ' '.join(src_sents)
            target = ' '.join(tgt_sents)
            source_bin = self.src_preprocessor.binarize(source, max_len=source_cutoff)
            target_bin = self.tgt_preprocessor.binarize(target, max_len=target_cutoff)
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
