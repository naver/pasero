# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import os
import itertools
import numpy as np
import logging
import torch
import regex
from typing import Optional, Any
from torch import Tensor
from pasero import utils
from pasero.utils import defined, tokens_as_tensor, SpecialTokens, warn_once
from pasero.config import TranslationTaskConfig, PreprocessingConfig, NoiseConfig, TransformerConfig
from pasero.preprocessing import Dictionary, TextPreprocessor, copy_tag, get_domain_tag, get_lang_code, split_tags
from pasero.tasks import Task, Corpus, InferenceCorpus

logger = logging.getLogger('translation')


class ParallelCorpus(Corpus):

    def __init__(
        self,
        source_path: str,
        target_path: str,
        source_lang: str,
        target_lang: str,
        source_tags: Optional[list[str]] = None,
        target_tags: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(
            paths=[source_path, target_path],
            langs=[source_lang, target_lang],
            **kwargs,
        )
        self.source_tags = source_tags
        self.target_tags = target_tags

    @property
    def source_path(self) -> str: return self.paths[0]
    @property
    def target_path(self) -> str: return self.paths[1]
    @property
    def source_lang(self) -> str: return self.langs[0]
    @property
    def target_lang(self) -> str: return self.langs[1]
    @property
    def src_format(self) -> str: return self.file_formats[0]
    @property
    def tgt_format(self) -> str: return self.file_formats[1]

    @classmethod
    def infer_domain(cls, path: str, langs: list[str]) -> str:
        name = super().infer_domain(path, langs)
        source_lang, target_lang = langs
        return name.removesuffix(f'.{source_lang}-{target_lang}').removesuffix(f'.{target_lang}-{source_lang}')

    @property
    def corpus_id(self) -> str:
        """
        Infer a unique corpus identifier from its path and language pair:
        "data/newstest2019.de" (de, en) -> "newtest2019.de-en"
        "data/newstest2019.de-en.de" (de, en) -> "newtest2019.de-en"
        "data/newstest2019.en-de.de" (de, en) -> "newtest2019.en-de.de-en"
        """
        suffix = '.' + '-'.join(self.langs)
        if not self.paths[0]:
            return f'stdin{suffix}'
        name = os.path.basename(self.paths[0])
        for lang in self.langs:
            name = name.removesuffix(f'.{lang}')
        return name.removesuffix(suffix) + suffix

    @property
    def meta(self):
        return {
            **super().meta,
            'source_lang': self.source_lang,
            'target_lang': self.target_lang,
            'source_tags': self.source_tags,
            'target_tags': self.target_tags,
        }

    def tuple_to_dict(self, tuple_: tuple) -> dict[str, Any]:
        assert len(tuple_) == 2
        source, target = tuple_
        return {'source': source, 'target': target, 'meta': self.meta}


class InferenceParallelCorpus(InferenceCorpus, ParallelCorpus):
    def __init__(
        self,
        source_path: str,
        source_lang: str,
        target_lang: str,
        ref_path: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            source_path=source_path,
            target_path=ref_path,
            source_lang=source_lang,
            target_lang=target_lang,
            **kwargs,
        )
        self.output_path = output_path
    
    @property
    def ref_path(self) -> str: return self.target_path


class TranslationTask(Task):
    cfg: TranslationTaskConfig

    def __init__(self, data_dir: str, cfg: TranslationTaskConfig):
        super().__init__(data_dir, cfg)

        src_preprocessor_cfg = PreprocessingConfig(cfg)
        self.src_preprocessor = TextPreprocessor(src_preprocessor_cfg, data_dir)
        # TranslationTaskConfig defines --target-* arguments to specify different parameters for the target tokenizer. 
        # Modify PreprocessingConfig to use these parameters for the target tokenizer.
        tgt_preprocessor_cfg = PreprocessingConfig(cfg)
        for key in tgt_preprocessor_cfg.as_dict():
            tgt_key = f'target_{key}'
            tgt_val = getattr(cfg, tgt_key, None)
            if tgt_val is not None:
                setattr(tgt_preprocessor_cfg, key, tgt_val)
        default_noise_cfg = NoiseConfig()
        for key, value in default_noise_cfg.as_dict().items():  # disable target-side noise by overwriting the noise
            # configuration with the default values
            setattr(tgt_preprocessor_cfg, key, value)
        self.tgt_preprocessor = TextPreprocessor(tgt_preprocessor_cfg, data_dir)

        # check that all tokenizers share the same special tokens
        assert self.src_preprocessor.special_tokens == self.tgt_preprocessor.special_tokens, (
            'source and target preprocessors should have the same special tokens'
        )
        
        if cfg.freeze_source_embed_regex:
            # used in Transformer to find which source embeddings to freeze: any source token that matches this regex
            # will have its source embedding frozen
            self.freeze_encoder_embed_mask = torch.tensor([
                bool(regex.match(cfg.freeze_source_embed_regex, token))
                for token in self.src_preprocessor.dictionary
            ])
            frozen_count = self.freeze_encoder_embed_mask.sum()
            logger.info(f'{frozen_count}/{len(self.freeze_encoder_embed_mask)} source embeddings will be frozen')

        self.min_len_ratio = cfg.min_len_ratio
        self.max_len_ratio = cfg.max_len_ratio

        # by default, the supported languages or domains are specified thanks to the --domains, --source-langs and 
        # --target-langs options (which can be defined in 'inference.yaml'), but other languages or domains can be
        # added later by `register_corpora`, or inferred from the language codes in the model's dictionary when loading
        # the model.
        self.source_langs = set()
        self.target_langs = set()
        if cfg.lang_pairs:  # has priority over --source-langs and --target-langs
            for lang_pair in cfg.lang_pairs:
                src, tgt = lang_pair.split('-')
                self.source_langs.add(src)
                self.target_langs.add(tgt)
        elif cfg.source_langs and cfg.target_langs:
            self.source_langs.update(cfg.source_langs)
            self.target_langs.update(cfg.target_langs)

        # A default source and target language can be defined in 'inference.yaml'. These are different from
        # `source_langs` and `target_langs`, which define the full list of covered languages.
        # These are used in the decoding API or with `pasero-serve` to define a default language pair when the user 
        # forgets to specify one.
        self.default_source_lang = cfg.source_lang
        self.default_target_lang = cfg.target_lang

        self.domains = set(cfg.domains or [])
        self.check_tags()

    # These can be updated after creating the task (e.g., when loading a model at inference)
    @property
    def max_source_len(self):
        return self.cfg.max_source_len
    @property
    def max_target_len(self):
        return self.cfg.max_target_len

    def register_corpora(self, *corpora: ParallelCorpus) -> None:
        """
        At training or inference, the global --source-langs and --target-langs may be underspecified: update the 
        full set of covered languages by looking at the languages of the corpora (specified through corpus attributes
        at training or file extensions at inference)
        """
        for corpus in corpora:
            meta = corpus.meta
            self.source_langs.add(meta['source_lang'])
            self.target_langs.add(meta['target_lang'])
            self.domains.add(meta['domain'])
        self.check_tags()
    
    def make_meta(
        self,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        domain: Optional[str] = None,
        source_tags: Optional[list[str]] = None,
        target_tags: Optional[list[str]] = None,
        corpus_id: Optional[str] = None,
    ) -> dict:
        return {
            'source_lang': source_lang or self.default_source_lang,
            'target_lang': target_lang or self.default_target_lang,
            'domain': domain,
            'source_tags': source_tags,
            'target_tags': target_tags,
            'corpus_id': corpus_id,
        }

    def check_meta(self, meta: dict) -> None:
        """ Check whether the languages and domains specified in the dictionary are supported by the current task """
        if meta.get('source_lang') is not None:
            assert meta['source_lang'] in self.source_langs, 'this source language is not covered by the model'
        if meta.get('target_lang') is not None:
            assert meta['target_lang'] in self.target_langs, 'this target language is not covered by the model'
        if meta.get('domain') is not None:
            assert meta['domain'] in self.domains, 'this domain is not covered by the model'

    def set_model_type(self, model_type: str) -> None:
        if model_type == 'decoder':
            assert self.max_target_len > self.max_source_len
            if not self.cfg.target_tags and not self.cfg.target_lang_code:
                warn_once(
                    logger,
                    'attempting decoder-only MT without a separator between source and target (it is recommended '
                    'to set --target-tags SOME_SPECIAL_TOKENS or --target-lang-code)'
                )
        super().set_model_type(model_type)

    @property
    def task_info(self) -> dict:
        return {
            **super().task_info,
            'source_langs': sorted(self.source_langs),
            'target_langs': sorted(self.target_langs),
            'default_source_lang': self.default_source_lang,
            'default_target_lang': self.default_target_lang,
            'domains': sorted(self.domains),
        }

    def get_langs_or_domains(self, key: str) -> set[str]:
        if key == 'source_lang':
            return self.source_langs
        elif key == 'target_lang':
            return self.target_langs
        elif key == 'domain':
            return self.domains
        else:
            raise NotImplementedError

    @property
    def inference_options(self) -> dict:
        options = dict(self.src_preprocessor.inference_options)
        
        for name, value in self.tgt_preprocessor.inference_options.items():
            if options.get(name) != value:
                options[f'target_{name}'] = value  # e.g., dict and target_dict
        
        option_names = [
            'lang_code', 'source_lang_code', 'target_lang_code', 'domain_tag', 'max_source_len', 'max_target_len',
            'source_tags', 'target_tags',
        ]

        for name in option_names:
            value = getattr(self.cfg, name)
            if value:  # boolean options above are all False by default and lists are empty by default
                options[name] = value

        options['source_langs'] = sorted(self.source_langs)
        options['target_langs'] = sorted(self.target_langs)

        return options

    def input_to_sample(self, input: str, meta: dict) -> dict:
        source, *target = input.rsplit('|||', maxsplit=1)  # '|||' is used a separator between the source and an 
        # optional prompt
        target = target[0] if target else None
        return {'source': source, 'target': target, 'meta': meta}

    @property
    def encoder_num_embeddings(self):
        return 0 if self.model_type == 'decoder' else self.src_preprocessor.num_symbols

    @property
    def decoder_num_embeddings(self):
        return self.tgt_preprocessor.num_symbols

    @property
    def preprocessors(self):
        return {'source': self.src_preprocessor, 'target': self.tgt_preprocessor}

    def log_sample(self, sample_bin: dict) -> None:
        corpus_id = sample_bin['meta']['corpus_id']

        if 'source' in sample_bin:
            # decoder-only models don't have sources (source and target are concatenated)
            source_tok = self.src_preprocessor.debinarize(sample_bin['source'])
            logger.debug(f'{corpus_id} | source line example: {source_tok}')

        target_tok = self.tgt_preprocessor.debinarize(sample_bin['target'])
        logger.debug(f'{corpus_id} | target line example: {target_tok}')

    def get_reference(self, sample: dict[str, Any]):
        return sample['target']

    def should_skip(self, source_len: int, target_len: int) -> bool:
        ratio = source_len / max(1, target_len)
        return (
            source_len == 0 or
            source_len > self.max_source_len or
            target_len > self.max_target_len or
            self.min_len_ratio and ratio < self.min_len_ratio or
            self.max_len_ratio and ratio > self.max_len_ratio
        )

    def copy_placeholder(self, source_tok: str, target_tok: str) -> tuple[str, str]:
        # Replace OOV symbols with the same source and target count (typically emojis) with a copy
        # placeholder.
        # OOV characters whose count doesn't match are just removed.
        src_counts, src_oov = self.src_preprocessor.get_oov(source_tok)
        tgt_counts, tgt_oov = self.tgt_preprocessor.get_oov(target_tok)
        to_copy = {w for w in src_oov | tgt_oov if src_counts[w] == tgt_counts[w]}
        to_del = {w for w in src_oov | tgt_oov if len(w) == 1 and src_counts[w] != tgt_counts[w]}
        if to_copy or to_del:
            source_tok = ' '.join(copy_tag if w in to_copy else w for w in source_tok.split() if w not in to_del)
            target_tok = ' '.join(copy_tag if w in to_copy else w for w in target_tok.split() if w not in to_del)
        return source_tok, target_tok

    def check_tags(self):
        """ Check that the dictionaries contain all the necessary lang codes and domain tags """
        src_dict = self.src_preprocessor.dictionary
        tgt_dict = self.tgt_preprocessor.dictionary
        
        if self.cfg.domain_tag:
            for domain in self.domains:
                domain_tag = get_domain_tag(domain)
                assert domain_tag in src_dict, f'{domain_tag} is OOV'
        
        if self.cfg.source_lang_code:
            for source_lang in self.source_langs:
                source_lang_code = get_lang_code(source_lang)
                assert source_lang_code in src_dict, f'{source_lang_code} is OOV'
        
        if self.cfg.lang_code:
            for target_lang in self.target_langs:
                target_lang_code = get_lang_code(target_lang)
                assert target_lang_code in src_dict, f'{target_lang_code} is OOV'
            
        if self.cfg.target_lang_code:
            for target_lang in self.target_langs:
                target_lang_code = get_lang_code(target_lang)
                assert target_lang_code in tgt_dict, f'{target_lang_code} is OOV'

    def get_source_tags(self, meta: dict) -> list[str]:
        """
        Return the tags (language codes, domain tags) that should be prepended to source lines according to given 
        metadata (which should specify this line's languages and domain)
        """
        source_lang = meta.get('source_lang')
        target_lang = meta.get('target_lang')
        domain = meta.get('domain')
        source_tags = meta.get('source_tags') or self.cfg.source_tags
        tags = []
        if source_tags:
            tags += source_tags
        if self.cfg.lang_code:
            assert target_lang, 'missing target language information'
            tags.append(get_lang_code(target_lang))
        if self.cfg.source_lang_code:
            assert source_lang, 'missing source language information'
            tags.append(get_lang_code(source_lang))
        if self.cfg.domain_tag:
            tags.append(get_domain_tag(domain))
        return list(filter(None, tags))

    def get_target_tags(self, meta: dict) -> list[str]:
        """
        Return the tags (language codes, domain tags) that should be prepended to target lines according to given 
        metadata (which should specify this line's languages and domain)
        """
        target_lang = meta.get('target_lang')
        target_tags = meta.get('target_tags') or self.cfg.target_tags
        tags = []
        if target_tags:
            tags += target_tags
        if self.cfg.target_lang_code:
            assert target_lang, 'missing target language information'
            tags.append(get_lang_code(target_lang))
        return list(filter(None, tags))

    def preprocess(
        self,
        sample: dict[str, Any],
        truncate: bool = False,
        tokenize: bool = True,
        inference: bool = False,
    ) -> dict[str, Any]:
        """
        Pre-process given sample pair with this task's preprocessors. The inputs are un-tokenized text and the outputs
        are tokenized (potentially with extra prepended tags, random noise, etc.) and binarized as numpy arrays (with
        the id of each token in the source and target dictionaries).

        Args:
            sample: dictionary with a "source" string, a "target" string and a "meta" dictionary indicating the 
                source lang, target lang and domain
            truncate: whether to truncate sources and targets that are too long
            tokenize: whether the sample should be tokenized (set to False if it is already tokenized)
            inference: True when this method is called with inference inputs (by TextGenerator)

        Returns: a dict with keys 'source', 'target', 'prompt_mask' and 'emojis'
            source: tokenized and binarized source sentence (as a numpy array)
            target: tokenized and binarized target sentence (as a numpy array)
            prompt_mask: boolean mask identifying which parts of `target` are not being predicted (their training 
                loss may be disabled with --prompt-loss 0)
            emojis: list of emojis found in the source string and replaced with placeholders
        """
        source = sample.get('source')
        target = sample.get('target')
        meta = sample.get('meta')

        source_cutoff = self.max_source_len if truncate else None
        target_cutoff = self.max_target_len if truncate else None

        # extract potential pre-existing prefix tags (can happen with datasets or user inputs that already contain
        # language codes)
        *src_tags, source = split_tags(source)
        
        if target is None:
            tgt_tags = []
        else:
            *tgt_tags, target = split_tags(target)

        # prefix tokens and tags (e.g., language codes)
        if tokenize:
            src_tags += self.get_source_tags(meta)
            tgt_tags += self.get_target_tags(meta)
        
        prompt_len = len(tgt_tags)

        if self.cfg.escape_emojis and not self.training:
            source, emojis = self.src_preprocessor.escape_emojis(source)
        else:
            emojis = []
        
        source_tok = self.src_preprocessor.tokenize(source) if tokenize else source
        source_tok = ' '.join(src_tags + [source_tok])

        target_tok = list(tgt_tags)
        if not target:
            pass
        elif tokenize:
            target_tok.append(self.tgt_preprocessor.tokenize(target))
        else:
            target_tok.append(target)
        target_tok = ' '.join(target_tok)

        if self.cfg.copy_placeholder and self.training:
            source_tok, target_tok = self.copy_placeholder(source_tok, target_tok)

        if self.model_type == 'decoder':  # concatenate
            
            source_bin = self.src_preprocessor.binarize(source_tok, max_len=source_cutoff, append_eos=False)
            if target_cutoff is not None:
                target_cutoff -= len(source_bin)
            
            target_bin = self.tgt_preprocessor.binarize(
                target_tok,
                max_len=target_cutoff,
            )
            
            source_mask = np.ones_like(source_bin, dtype=bool)
            target_mask = np.zeros_like(target_bin, dtype=bool)
            target_mask[:prompt_len] = True  # mask lang codes, etc.
            
            if self.should_skip(len(source_bin), len(source_bin) + len(target_bin)):  # max target length applies to the
                # concatenation
                assert not truncate  # this shouldn't happen since we truncate
                return {}
            else:                
                return {
                    'target': np.concatenate([source_bin, target_bin]),
                    'prompt_mask': np.concatenate([source_mask, target_mask]),
                    'emojis': emojis,
                    'meta': meta,
                }
        else:
            source_bin = self.src_preprocessor.binarize(source_tok, max_len=source_cutoff)
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
                    'emojis': emojis,
                    'meta': meta,
                }

    def postprocess(
        self,
        sample_bin: dict[str, Any],
        hypothesis: dict[str, Any],
        detokenize: bool = True,
    ) -> None:
        super().postprocess(sample_bin, hypothesis, detokenize=detokenize)
        
        if self.cfg.escape_emojis:
            hypothesis['detok'] = self.tgt_preprocessor.deescape_emojis(hypothesis['detok'], sample_bin['emojis'])
        
        if 'source' in sample_bin:
            # decoder-only models don't have sources (source and target are concatenated)
            hypothesis['src_tokens'] = self.src_preprocessor.debinarize(sample_bin['source'])
        
    @classmethod
    def _get_corpus(cls, *args, **kwargs) -> ParallelCorpus:
        """
        Used in `_get_corpora` to get corpora for this task and overriden by SpeechTranslationTask to modify the 
        type of the corpus' source file without having to rewrite `_get_corpora`
        """
        return ParallelCorpus(*args, **kwargs)

    @classmethod
    def _get_corpora(
        cls,
        data_dir: str,
        source_langs: Optional[list[str]],
        target_langs: Optional[list[str]],
        lang_pairs: Optional[list[str]],
        corpus_definition: dict,
        allow_monolingual: bool = False,
        source_tags: Optional[list[str]] = None,
        target_tags: Optional[list[str]] = None,
    ) -> list[ParallelCorpus]:
        """
        Called by `get_train_corpora` and `get_valid_corpora` to make corpora from the task's configuration and 
        YAML corpus specifications. This is method is also used by TranslationTask's subclasses (e.g., 
        SpeechTranslationTask) and should remain generic enough. The method `_get_corpus` can be used to define 
        task-specific behavior in each subclass (e.g., different file formats)
        """
        corpora: dict[str, ParallelCorpus] = {}
        
        bidir = not source_langs and not target_langs and not lang_pairs and corpus_definition.get('bidir')   # bidir is 
        # disabled by manual specification of language pairs

        domain = corpus_definition.get('domain')
        multiplier = corpus_definition.get('multiplier', 1)
        probability = corpus_definition.get('probability')
        early_stopping = corpus_definition.get('early_stopping', True)
        source_langs = corpus_definition.get('source_langs', source_langs) or []
        target_langs = corpus_definition.get('target_langs', target_langs) or []
        lang_pairs = corpus_definition.get('lang_pairs', lang_pairs) or []
        flexible = corpus_definition.get('flexible')
        source_tags = corpus_definition.get('source_tags', source_tags)
        target_tags = corpus_definition.get('target_tags', target_tags)
        paths = corpus_definition.get('paths')
        source_paths = corpus_definition.get('source_paths') or paths
        target_paths = corpus_definition.get('target_paths') or paths
        assert isinstance(source_paths, list), 'corpus definition does not contain a valid list of source paths'
        assert isinstance(target_paths, list), 'corpus definition does not contain a valid list of target paths'

        if lang_pairs:
            lang_pairs = [tuple(pair.split('-')) for pair in lang_pairs]
        elif source_langs and target_langs:
            lang_pairs = [
                (source_lang, target_lang) for source_lang, target_lang in itertools.product(source_langs, target_langs)
                if allow_monolingual or source_lang != target_lang
            ]
            # lang pairs where the source and target language are identical can still be defined with the
            # 'lang_pairs' attribute or --lang-pairs
        
        assert lang_pairs, 'no language pair is defined'
        assert all(len(pair) == 2 for pair in lang_pairs)

        if bidir:
            lang_pairs += [(target_lang, source_lang) for source_lang, target_lang in lang_pairs]
        # removes duplicates while keeping order of insertion (contrary to set)
        lang_pairs = list(dict.fromkeys(lang_pairs))

        for source_path, target_path in zip(source_paths, target_paths):
            for source_lang, target_lang in lang_pairs:
                """
                The list of corpora is the product of all language pairs and given paths for this corpus. For instance:
                
                paths: ['europarl.{pair}', 'news-commentary.{pair}']
                lang_pairs: ['fr-en', 'en-fr']

                Results in:
                
                [
                    ParallelCorpus('europarl.fr-en.fr', 'europarl.fr-en.en'),
                    ParallelCorpus('europarl.en-fr.en', 'europarl.en-fr.fr'),
                    ParallelCorpus('news-commentary.fr-en.fr', 'news-commentary.fr-en.en'),
                    ParallelCorpus('news-commentary.en-fr.en', 'news-commentary.en-fr.fr'),
                ]
                """

                # generate a list of possible candidates for this corpus by reversing the language pair (i.e., if 
                # europarl.fr-en.* don't exist, look for europarl.en-fr.*) and by looking for files relative to
                # data_dir or to the working directory
                candidates: list[ParallelCorpus] = []
                for pair in f'{source_lang}-{target_lang}', f'{target_lang}-{source_lang}':
                    for root_dir in data_dir, '.':
                        src_path = source_path.format(src=source_lang, tgt=target_lang, pair=pair)
                        tgt_path = target_path.format(src=source_lang, tgt=target_lang, pair=pair)
                        src_path = os.path.join(root_dir, src_path)
                        tgt_path = os.path.join(root_dir, tgt_path)

                        if len(lang_pairs) == 1 and os.path.exists(src_path) and os.path.exists(tgt_path):
                            # do not add lang code suffixes if the paths exist: this lets us specify either full paths
                            # or corpus prefixes
                            pass
                        else:
                            src_suffix = f'.{source_lang}'
                            tgt_suffix = f'.{target_lang}'
                            src_path = src_path.removesuffix(src_suffix) + src_suffix
                            tgt_path = tgt_path.removesuffix(tgt_suffix) + tgt_suffix

                        domain_ = defined(domain, ParallelCorpus.infer_domain(src_path, [source_lang, target_lang]))
                        corpus = cls._get_corpus(
                            src_path, tgt_path,
                            source_lang=source_lang, target_lang=target_lang,
                            source_tags=source_tags, target_tags=target_tags,
                            domain=domain_,
                            multiplier=multiplier,
                            probability=probability,
                            early_stopping=early_stopping,
                        )
                        candidates.append(corpus)

                # find the first candidate corpus whose files exist
                corpus = next((corpus for corpus in candidates if corpus.exists()), None)
                if corpus is None:
                    if flexible:
                        continue
                    else:
                        raise FileNotFoundError(f"corpus ({', '.join(candidates[0].paths)}) does not exist")
                elif corpus.corpus_id not in corpora:  # avoids corpora with duplicate corpus ids
                    corpora[corpus.corpus_id] = corpus
        
        corpus_list = list(corpora.values())

        for corpus in corpus_list:
            if corpus.probability:
                corpus.probability /= len(corpus_list)
        
        return corpus_list

    @classmethod
    def get_valid_corpora(
        cls,
        cfg: TranslationTaskConfig,
        data_dir: str,
        corpus_definitions: list[dict],
    ) -> list[ParallelCorpus]:
        corpora: list[ParallelCorpus] = []
        for corpus_definition in corpus_definitions:
            corpora += cls._get_corpora(
                data_dir,
                cfg.valid_source_langs or cfg.source_langs,
                cfg.valid_target_langs or cfg.target_langs,
                cfg.valid_lang_pairs or cfg.lang_pairs,
                corpus_definition,
                cfg.allow_monolingual,
                cfg.source_tags,
                cfg.target_tags,
            )
        assert len(set(corpus.corpus_id for corpus in corpora)) == len(corpora), 'there are duplicate corpus ' \
            'definitions'
        return corpora

    @classmethod
    def get_train_corpora(
        cls,
        cfg: TranslationTaskConfig,
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
        assert len(set(corpus.corpus_id for corpus in corpora)) == len(corpora), 'there are duplicate corpus ' \
            'definitions'
        return corpora

    @classmethod
    def get_inference_corpus(
        cls,
        source_path: str,
        source_lang: str,
        target_lang: str,
        ref_path: Optional[str] = None,
        output_path: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> InferenceParallelCorpus:
        
        return InferenceParallelCorpus(
            source_path,
            source_lang,
            target_lang,
            ref_path=ref_path,
            output_path=output_path,
            domain=domain,
        )

    @classmethod
    def get_inference_corpora(
        cls,
        cfg: TranslationTaskConfig,
        input_paths: Optional[list[str]] = None,
        output_paths: Optional[list[str]] = None,
        ref_paths: Optional[list[str]] = None,
        corpus_prefix: Optional[str] = None,
    ) -> list[InferenceParallelCorpus]:
        if cfg.lang_pairs:
            source_langs = []
            target_langs = []
            for pair in cfg.lang_pairs:
                src, tgt = pair.split('-')
                source_langs.append(src)
                target_langs.append(tgt)
        else:
            source_langs = [cfg.source_lang] if cfg.source_lang else None
            target_langs = [cfg.target_lang] if cfg.target_lang else None

        if corpus_prefix is not None:
            assert input_paths is None and ref_paths is None, '-e/--eval-corpus is exclusive with -i/--input and ' \
                '-r/--reference'
            assert source_langs and target_langs, '-e/--eval-corpus requires -s/--source-lang and -t/--target-lang ' \
                'or -l/--lang-pairs'
            input_paths = [f'{corpus_prefix}.{{src}}']
            ref_paths = [f'{corpus_prefix}.{{tgt}}']

        if source_langs and target_langs:
            if len(source_langs) == 1:   # "-s en -t de fr" -> "-s en en -t de fr"
                source_langs *= len(target_langs)
            if len(target_langs) == 1:   # "-s de fr -t en" -> "-s de fr -t en en"
                target_langs *= len(source_langs)
            
            if not cfg.allow_monolingual and len(source_langs) > 1:  # "-s en en en -t en de fr" -> "-s en en -t de fr"
                source_langs, target_langs = zip(
                    *[(src, tgt) for src, tgt in zip(source_langs, target_langs) if src != tgt]
                )

        if not source_langs:
            # "-i test.de test.fr" -> "-s de fr"
            source_langs = (
                [src_file.split('.')[-1] for src_file in input_paths]
                if input_paths else [None]
            )
            assert all(source_langs), 'source languages cannot be inferred, please provide -s/--source-lang or -l/--lang-pairs'
        
        if not target_langs:
            # "-r test.de test.fr" -> "-r de fr"
            target_langs = (
                [tgt_file.split('.')[-1] for tgt_file in ref_paths]
                if ref_paths else [None]
            )
            assert all(target_langs), 'target languages cannot be inferred, please provide -t/--target-lang or -l/--lang-pairs'

        # if a single value is given for -o/-r/-i, replace placeholders ({pair}, {src}, {tgt}) with the given source and 
        # target langs. For instance "-s de fr -t en -o test.{pair}.out" is equivalent to:
        # "-s de fr -t en en -o test.de-en.out test.fr-en.out"
        for paths in input_paths, output_paths, ref_paths:
            if paths and len(paths) == 1:
                path = paths[0]
                paths[:] = [
                    TranslationTaskConfig.format_path(path, src, tgt)
                    for src, tgt in zip(source_langs, target_langs)
                ]

        # Because we will later iterate over input, output and reference files, along with their language and domain
        # properties, we must make sure these lists have the same length

        input_paths = input_paths or [None]    # None will read from standard input
        ref_paths = ref_paths or [None]        # None will not perform any evaluation
        output_paths = output_paths or [None]  # None will write to standard output
        
        domains = cfg.domains or [None]
        if len(input_paths) == 1 and len(ref_paths) == 1:
            input_paths *= len(domains)
            ref_paths *= len(domains)

        if len(ref_paths) == 1:
            ref_paths *= len(input_paths)
        if len(input_paths) == 1:
            input_paths *= len(ref_paths)
        
        # --outputs, --source-langs, --target-langs, and --domains must have the same number of arguments as --input,
        # or just 1 (in which case, the same output file / source lang / target lang / domain is used for all input files)
        if len(source_langs) == 1:
            source_langs *= len(input_paths)
        if len(target_langs) == 1:
            target_langs *= len(input_paths)
        if len(output_paths) == 1:
            output_paths *= len(input_paths)
        if len(domains) == 1:
            domains *= len(input_paths)

        assert len(input_paths) == len(output_paths), '-i/--input and -r/--reference must have the same number of arguments'
        assert len(input_paths) == len(ref_paths), '-i/--input and -o/--output must have the same number of arguments'
        assert len(input_paths) == len(source_langs), '-i/--input and -l/--lang-pairs must have the same number of arguments'
        assert len(input_paths) == len(domains), '-i/--input and --domains must have the same number of arguments'

        # update the task's potentially incomplete configuration with the inferred languages and domains
        cfg.source_langs = source_langs
        cfg.target_langs = target_langs
        cfg.domains = domains

        corpora: list[InferenceParallelCorpus] = []
        for input_path, output_path, ref_path, source_lang, target_lang, domain in zip(
            input_paths,
            output_paths,
            ref_paths,
            source_langs,
            target_langs,
            domains,
        ):
            corpus = cls.get_inference_corpus(
                input_path,
                source_lang=source_lang,
                target_lang=target_lang,
                ref_path=ref_path,
                output_path=output_path,
                domain=domain,
            )
            corpora.append(corpus)

        return corpora

    @classmethod
    def collate(
        cls,
        batch: list[dict],
        special_tokens: SpecialTokens,
        dtype: torch.dtype,
        model_type: str,
    ) -> dict:
        if not batch:
            return None

        batched = super().collate(batch, special_tokens, dtype, model_type)

        if model_type == 'encoder_decoder':
            sources = [sample['source'] for sample in batch]
            source_batch, source_length = tokens_as_tensor(sources, special_tokens, dtype=dtype)
            batched.update({
                'source': source_batch,
                'source_length': source_length,
            })
            if all('emojis' in sample for sample in batch):
                batched['emojis'] = [sample['emojis'] for sample in batch]

        return batched

    def count_oov(self, sample_bin: dict) -> tuple[int, int]:
        oov, total = super().count_oov(sample_bin)
        if 'source' in sample_bin:
            total += (sample_bin['source'] != self.padding_idx).sum()
            if self.unk_idx != self.padding_idx:
                oov += (sample_bin['source'] == self.unk_idx).sum()
        return oov, total

    def remap_encoder_embed(self, embed: Optional[Tensor]) -> Optional[Tensor]:
        if self.cfg.old_source_dict and embed is not None:
            old_source_dict_path = utils.find_file(self.cfg.old_source_dict, dirs=[self.data_dir, '.'])
            old_source_dict = Dictionary(old_source_dict_path)
            embed = self.src_preprocessor.dictionary.remap_embed(
                old_embed=embed,
                old_dict=old_source_dict,
                default=self.cfg.default_embed,
            )
        return embed
    
    def remap_decoder_embed(self, embed: Optional[Tensor]) -> Optional[Tensor]:
        if self.cfg.old_target_dict and embed is not None:
            old_target_dict_path = utils.find_file(self.cfg.old_target_dict, dirs=[self.data_dir, '.'])
            old_target_dict = Dictionary(old_target_dict_path)
            embed = self.tgt_preprocessor.dictionary.remap_embed(
                old_embed=embed,
                old_dict=old_target_dict,
                default=self.cfg.default_embed,
            )
        return embed
    
    def load_checkpoint_for_inference(
        self,
        *ckpt_paths: str,
        rank: int = 0,
        world_size: int = 1,
        arch: Optional[str] = None,
    ) -> tuple[dict, TransformerConfig]:

        # If the list of covered languages is not explicitely set, try to infer it from the language codes in the 
        # dictionary. We don't do that in `__init__`, because `pasero-decode` tries to infer languages from the 
        # filenames after initializing the task.
        if not self.source_langs:
            self.source_langs = self.src_preprocessor.infer_langs()
        if not self.target_langs:
            self.target_langs = self.tgt_preprocessor.infer_langs()

        model_state, model_cfg = super().load_checkpoint_for_inference(
            *ckpt_paths,
            rank=rank,
            world_size=world_size,
            arch=arch,
        )
        
        if self.cfg.old_source_dict or self.cfg.old_target_dict:
            # test-time vocabulary filtering: source and target embeddings are now different
            model_cfg.shared_embeddings = False

        return model_state, model_cfg
