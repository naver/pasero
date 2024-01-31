# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import os
import numpy as np
import logging
from typing import Optional, Any
from pasero.utils import defined
from pasero.config import register_task, LanguageModelingTaskConfig, TransformerConfig
from pasero.preprocessing import TextPreprocessor, get_domain_tag, get_lang_code
from pasero.tasks import Task, Corpus, InferenceCorpus

logger = logging.getLogger('language_modeling')


class MonolingualCorpus(Corpus):

    def __init__(
        self,
        path: str,
        lang: str,
        tags: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(
            paths=[path],
            langs=[lang],
            **kwargs,
        )
        self.tags = tags
        
    @property
    def path(self) -> str: return self.paths[0]
    @property
    def lang(self) -> str: return self.langs[0]
    @property
    def format(self) -> str: return self.file_formats[0]

    @property
    def corpus_id(self) -> str:
        if not self.paths[0]:
            return 'stdin'
        else:
            return os.path.basename(self.paths[0])

    @property
    def meta(self):
        return {
            **super().meta,
            'lang': self.lang,
            'tags': self.tags,
        }

    def tuple_to_dict(self, tuple_: tuple) -> dict[str, Any]:
        assert len(tuple_) == 1
        return {'target': tuple_[0], 'meta': self.meta}


class InferenceMonolingualCorpus(InferenceCorpus, MonolingualCorpus):
    def __init__(
        self,
        path: str,
        lang: str,
        ref_path: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(path=path, lang=lang, **kwargs)
        self.ref_path = ref_path
        self.output_path = output_path
    

@register_task('language_modeling')
class LanguageModelingTask(Task):
    cfg: LanguageModelingTaskConfig

    def __init__(
        self,
        data_dir: str,
        cfg: LanguageModelingTaskConfig,
    ):
        super().__init__(data_dir, cfg)
        self.preprocessor = TextPreprocessor(cfg, data_dir)
        self.langs = set(cfg.langs or [])
        self.domains = set(cfg.domains or [])
        self.check_tags()

    # This can be updated after creating the task (e.g., when loading a model at inference)
    @property
    def max_len(self):
        return self.cfg.max_len

    def register_corpora(self, *corpora: MonolingualCorpus) -> None:
        """ Add the languages or domains of the given corpora to the task """
        for corpus in corpora:
            meta = corpus.meta
            self.langs.add(meta['lang'])
            self.domains.add(meta['domain'])
        self.check_tags()
    
    def make_meta(
        self,
        lang: Optional[str] = None,
        domain: Optional[str] = None,
        tags: Optional[list[str]] = None,
        corpus_id: Optional[str] = None,
    ) -> dict:
        return {
            'lang': lang,
            'domain': domain,
            'tags': tags,
            'corpus_id': corpus_id,
        }

    def check_meta(self, meta: dict) -> None:
        """ Check whether the languages and domains specified in the dictionary are supported by the current task """
        if meta.get('lang') is not None:
            assert meta['lang'] in self.langs, 'this language is not covered by the model'
        if meta.get('domain') is not None:
            assert meta['domain'] in self.domains, 'this domain is not covered by the model'
    
    def setup_for_model(self, model_cfg: TransformerConfig) -> None:
        assert model_cfg.model_type == 'decoder'
        super().setup_for_model(model_cfg)

    @property
    def task_info(self) -> dict:
        return {
            **super().task_info,
            'langs': sorted(self.langs),
            'domains': sorted(self.domains),
        }

    def get_langs_or_domains(self, key: str) -> set[str]:
        if key == 'lang':
            return self.langs
        elif key == 'domain':
            return self.domains

    @property
    def inference_options(self) -> dict:
        options = {**self.preprocessor.inference_options, 'task': 'language_modeling'}
        
        for name in 'lang_code', 'domain_tag', 'max_len', 'tags':
            value = getattr(self.cfg, name)
            if value:  # lang_code and domain_tag are False by default, tags is empty by default
                options[name] = value
        
        if self.langs:
            options['langs'] = sorted(self.langs)

        return options

    def input_to_sample(self, input: str, meta: dict = {}) -> dict:
        return {'target': input, 'meta': meta}

    @property
    def encoder_num_embeddings(self) -> int:
        return 0

    @property
    def decoder_num_embeddings(self) -> int:
        return self.preprocessor.num_symbols

    @property
    def preprocessors(self) -> dict[str, TextPreprocessor]:
        return {'target': self.preprocessor}

    def log_sample(self, sample_bin: dict) -> None:
        decoder_input = self.preprocessor.debinarize(sample_bin['decoder_input'])
        decoder_input = ' '.join(decoder_input)
        corpus_id = sample_bin['meta']['corpus_id']
        logger.debug(f'{corpus_id} | line example: {decoder_input}')

    def get_reference(self, sample: dict[str, Any]):
        return None

    def check_tags(self):
        """ Check that the dictionaries contain all the necessary lang codes and domain tags """
        dict = self.preprocessor.dictionary
        
        if self.cfg.domain_tag:
            for domain in self.domains:
                domain_tag = get_domain_tag(domain)
                assert domain_tag in dict, f'{domain_tag} is OOV'
        
        if self.cfg.lang_code:
            for lang in self.langs:
                lang_code = get_lang_code(lang)
                assert lang_code in dict, f'{lang_code} is OOV'

    def get_tags(self, meta: dict) -> list[str]:
        lang = meta.get('lang')
        domain = meta.get('domain')
        tags = meta.get('tags') or self.cfg.tags
        tags = list(tags) if tags else []
        if self.cfg.lang_code:
            assert lang, 'missing language information'
            tags.append(get_lang_code(lang))
        if self.cfg.domain_tag:
            tags.append(get_domain_tag(domain))
        return list(filter(None, tags))

    def preprocess(
        self,
        sample: dict[str, Any],
        truncate: bool = False,
        tokenize: bool = True,
        append_eos: bool = False,
    ) -> dict[str, Any]:
        target = sample['target']
        meta = sample['meta']
        cutoff = self.max_len if truncate else None

        tags = self.get_tags(meta)
        prompt_len = len(tags)

        target_tok = list(tags)
        if not target:
            pass
        elif tokenize:
            target_tok += self.preprocessor.tokenize(target)
        else:
            target_tok += target.split()
        
        decoder_input = self.preprocessor.binarize(
            target_tok,
            max_len=cutoff,
            truncate_left=True,
            prepend_bos=self.prepend_bos,
            append_eos=append_eos,
        )
        
        prompt_mask = np.zeros_like(decoder_input, dtype=bool)
        prompt_mask[:prompt_len + int(self.prepend_bos)] = True

        if len(decoder_input) > self.max_len:
            assert not truncate  # this shouldn't happen since we truncate
            return {}
        else:
            return {
                'decoder_input': decoder_input,
                'prompt_mask': prompt_mask,
                'meta': meta,
            }

    @classmethod
    def _get_corpus(cls, *args, **kwargs) -> MonolingualCorpus:
        """
        Used in `_get_corpora` to create corpora for this task and overriden by DialogueTask to modify the 
        type of the corpus' source file without having to rewrite `_get_corpora`
        """
        return MonolingualCorpus(*args, **kwargs)

    @classmethod
    def _get_corpora(
        cls,
        data_dir: str,
        langs: Optional[list[str]],
        corpus_definition: dict,
        tags: Optional[list[str]] = None,
    ) -> list[MonolingualCorpus]:
        """
        Called by `get_train_corpora` and `get_valid_corpora` to make corpora from the task's configuration and 
        YAML corpus specifications. This is method is also used by LanguageModelingTask's subclasses (e.g., 
        DialogueTask) and should remain generic enough. The method `_get_corpus` can be used to define task-specific 
        behavior in each subclass (e.g., different file formats)
        """
        corpora: dict[str, MonolingualCorpus] = {}
        
        domain = corpus_definition.get('domain')
        multiplier = corpus_definition.get('multiplier', 1)
        probability = corpus_definition.get('probability')
        early_stopping = corpus_definition.get('early_stopping', True)
        langs = langs or corpus_definition.get('langs') or []
        flexible = corpus_definition.get('flexible')
        tags = corpus_definition.get('tags', tags)
        paths = corpus_definition['paths']
        assert isinstance(paths, list)

        langs = langs or ['any']

        # removes duplicates while keeping order of insertion (contrary to set)
        langs = list(dict.fromkeys(langs))
        for path in paths:
            for lang in langs:

                for root_dir in data_dir, '.':  # look for files relative to data_dir or to the working directory
                    path_ = path.format(lang=lang)  # replace all '{lang}' placeholders
                    path_ = os.path.join(root_dir, path_)

                    if len(langs) == 1 and os.path.exists(path_):
                        # do not add a lang code suffix if the path exists
                        pass
                    else:
                        suffix = f'.{lang}'
                        path_ = path_.removesuffix(suffix) + suffix
                    
                    domain_ = defined(domain, MonolingualCorpus.infer_domain(path_, [lang]))
                    
                    corpus = cls._get_corpus(
                        path_,
                        lang=lang,
                        domain=domain_,
                        multiplier=multiplier,
                        probability=probability,
                        early_stopping=early_stopping,
                        tags=tags,
                    )
                    if corpus.exists():
                        break
                    
                corpus_id = corpus.corpus_id
                if corpus_id in corpora:
                    pass
                elif corpus.exists():
                    corpora[corpus_id] = corpus
                elif not flexible:
                    raise FileNotFoundError(f"corpus '{corpus.path}' does not exist")
        
        corpus_list = list(corpora.values())

        for corpus in corpus_list:
            if corpus.probability:
                corpus.probability /= len(corpus_list)

        return corpus_list

    @classmethod
    def get_valid_corpora(
        cls,
        cfg: LanguageModelingTaskConfig,
        data_dir: str,
        corpus_definitions: list[dict],
    ) -> list[MonolingualCorpus]:
        corpora: list[MonolingualCorpus] = []
        for corpus_definition in corpus_definitions:
            corpora += cls._get_corpora(
                data_dir,
                cfg.valid_langs or cfg.langs,
                corpus_definition,
                cfg.tags,
            )
        assert len(set(corpus.corpus_id for corpus in corpora)) == len(corpora), 'there are duplicate corpus ' \
            'definitions'
        return corpora

    @classmethod
    def get_train_corpora(
        cls,
        cfg: LanguageModelingTaskConfig,
        data_dir: str,
        corpus_definitions: list[dict],
    ) -> list[MonolingualCorpus]:
        corpora: list[MonolingualCorpus] = []
        for corpus_definition in corpus_definitions:
            corpora += cls._get_corpora(
                data_dir,
                cfg.langs,
                corpus_definition,
                cfg.tags,
            )
        assert len(set(corpus.corpus_id for corpus in corpora)) == len(corpora), 'there are duplicate corpus ' \
            'definitions'
        return corpora

    @classmethod
    def get_inference_corpus(
        cls,
        path: str,
        lang: Optional[str] = None,
        ref_path: Optional[str] = None,
        output_path: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> InferenceMonolingualCorpus:
        
        return InferenceMonolingualCorpus(
            path,
            lang=lang,
            ref_path=ref_path,
            output_path=output_path,
            domain=domain,
        )
    
    @classmethod
    def get_inference_corpora(
        cls,
        cfg: LanguageModelingTaskConfig,
        input_paths: Optional[list[str]] = None,
        output_paths: Optional[list[str]] = None,
        ref_paths: Optional[list[str]] = None,
        corpus_prefix: Optional[str] = None,
    ) -> list[InferenceMonolingualCorpus]:
        langs = cfg.langs

        if corpus_prefix is not None:
            assert input_paths is None, '-e/--eval-corpus is exclusive with -i/--input'
            assert langs, '-e/--eval-corpus requires -l/--langs'
            input_paths = [f'{corpus_prefix}.{{lang}}']

        if not langs:
            # "-i test.de test.fr" -> "-l de fr"
            langs = (
                [path.split('.')[-1] for path in input_paths]
                if input_paths else [None]
            )
        
        # if a single value is given for -o/-r/-i, replace {lang} placeholders with the given langs.
        # For instance "-l de en -o test.{lang}.out" is equivalent to:
        # "-l de en -o test.de.out test.en.out"
        for paths in input_paths, output_paths, ref_paths:
            if paths and len(paths) == 1:
                path = paths[0]
                paths[:] = [LanguageModelingTaskConfig.format_path(path, lang) for lang in langs]

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
        
        # --outputs, --langs, and --domains must have the same number of arguments as --input,
        # or just 1 (in which case, the same output file / lang / domain is used for all input files)
        if len(langs) == 1:
            langs *= len(input_paths)
        if len(output_paths) == 1:
            output_paths *= len(input_paths)
        if len(domains) == 1:
            domains *= len(input_paths)

        assert len(input_paths) == len(output_paths), '-i/--input and -r/--reference must have the same number of arguments'
        assert len(input_paths) == len(ref_paths), '-i/--input and -o/--output must have the same number of arguments'
        assert len(input_paths) == len(langs), '-i/--input and -l/--langs must have the same number of arguments'
        assert len(input_paths) == len(domains), '-i/--input and --domains must have the same number of arguments'

        # modify the tasks's incomplete configuration with the inferred languages and domains
        cfg.langs = langs
        cfg.domains = domains

        corpora = []
        
        for input_path, output_path, ref_path, lang, domain in zip(
            input_paths,
            output_paths,
            ref_paths,
            langs,
            domains,
        ):
            corpus = cls.get_inference_corpus(
                input_path,
                lang=lang,
                ref_path=ref_path,
                output_path=output_path,
                domain=domain,
            )
            corpora.append(corpus)

        return corpora
