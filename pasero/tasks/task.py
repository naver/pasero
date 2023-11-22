# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import os
import numpy as np
import torch
import functools
import logging
import argparse
from torch import nn, Tensor, LongTensor
from typing import Optional, Any, Iterable, Iterator
from pasero import utils, evaluation
from pasero.utils import mask_to_len, tokens_as_tensor, SpecialTokens
from pasero.config import TaskConfig, TransformerConfig, get_model_config
from pasero.preprocessing import TextPreprocessor
from pasero.files import File
from pasero.tokenizers import detokenize


logger = logging.getLogger('translation')


class Corpus:
    """
    Generic corpus that can consist of one or several files.
    Such files are typically accessed in parallel (i.e., they are assumed to have the same number of lines or entries).

    The corpus only specifies the paths and metadata associated with these files (e.g., languages, domains, corpus id),
    as well as a method for opening the files. It is up to datasets (TrainingDataset and ValidationDataset) to open the
    files, read line tuples from them and send those to `Task.preprocess` and `Task.collate`.

    The `tuple_to_dict` method converts line tuples as read from those files into a dictionary (aka "sample") that can
    be understood by the right task.

    Each type of task will typically have its own corpus subclass (e.g., TranslationTask -> ParallelCorpus, 
    LanguageModelingTask -> MonolingualCorpus, etc.)
    It is the responsibility of the task's `get_train_corpora` and `get_valid_corpora` methods to generate lists
    of corpora based on the task's configuration.
    """
    def __init__(
        self,
        paths: list[Optional[str]],
        *,
        langs: Optional[list[str]] = None,
        file_formats: Optional[list[str]] = None,
        domain: Optional[str] = None,
        multiplier: float = 1.0,
        probability: Optional[float] = None,
        early_stopping: bool = True,
        max_doc_size: int = 1,
    ):
        self.paths = paths
        self.langs = langs or [path.split('.')[-1] for path in paths]
        self.file_formats = file_formats or ['txt'] * len(paths)
        assert len(self.langs) == len(self.paths)
        assert len(self.file_formats) == len(self.paths)
        assert len(self.paths) >= 1
        self.domain = domain
        self.multiplier = multiplier
        self.probability = probability
        self.early_stopping = early_stopping   # whether this corpus should be used for early stopping
        self.max_doc_size = max_doc_size

    @property
    def realpaths(self) -> list[str]:
        return [os.path.realpath(path) for path in self.paths]

    def open_files(self, store_files_under: Optional[int] = None) -> list[File]:
        """
        Used in `datasets` to open the files described by this corpus. If `store_files_under` is set, the content of 
        files whose size in bytes does not exceed this value is saved in memory.
        """
        return [
            File.open(path, format=format, store_files_under=store_files_under)
            for path, format in zip(self.paths, self.file_formats)
        ]

    @property
    def meta(self):
        return {
            'domain': self.domain,
            'corpus_id': self.corpus_id,
        }

    def __str__(self) -> str:
        return self.corpus_id

    def __repr__(self) -> str:
        defaults = Corpus(paths=self.paths).__dict__
        args_str = [repr(self.paths)]
        for name, default in defaults.items():
            # create a concise representation that can be used to create a new instance of this corpus, but does not
            # include unnecessary default values
            value = getattr(self, name)
            if name != 'paths' and value != default:
                args_str.append(f'{name}={repr(value)}')
        return f"Corpus({', '.join(args_str)})"

    def exists(self):
        """ Check whether all files described by this corpus exist and are not empty. """
        return all(os.path.exists(path) and os.path.getsize(path) > 0 for path in self.paths)

    @property
    def corpus_id(self) -> str:
        """
        Unique string identifier for this corpus. It should be interpretable (and prettty) because it will be used 
        in logs. To implement by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def infer_domain(path: str, langs: list[str]) -> str:
        """
        Similar to `infer_corpus_id` but strips language codes:
        "data/newstest2019.de" (de, en) -> "newtest2019"
        """
        if not path:
            return 'default'
        name = os.path.basename(path)
        for lang in langs:
            name = name.removesuffix(f'.{lang}')
        return name

    def getmtime(self) -> float:
        """ Last modification time of this corpus' files """
        return max(os.path.getmtime(path) for path in self.paths)

    def getsize(self) -> int:
        """ Size in bytes of this corpus on the disk """
        return sum(os.path.getsize(path) for path in self.paths)

    @classmethod
    def tuple_to_dict(self, tuple_: tuple) -> dict[str, Any]:
        """
        Convert an example represented as a tuple, such as obtained by reading this corpus' files in parallel (with zip)
        into a dictionary example that can be understood by the relevant task.
        For instance, for ParallelCorpus/TranslationTask, this would be do:

        `(first, second)` => `{'source':first, 'target': second}`
        """
        raise NotImplementedError


class InferenceCorpus(Corpus):
    """
    Contrary to training corpora, inference corpora have a single "input" file, which can be interpreted as an input to 
    the encoder, decoder or both by the task (through its `input_to_sample` method).
    This input file can be the standard input.
    
    They may also specify a reference file to evaluate against (different to training corpora's target, which is used
    as input to the decoder at training) and/or an output file where to write the decoding outputs.
    """
    output_path: str
    ref_path: str

    @property
    def input_path(self):
        return self.paths[0]

    @property
    def binary_input(self):
        return self.file_formats[0] == 'numpy'

    def input_file(self, rank: int = 0, world_size: int = 1) -> File:
        def convert_line_breaks(iterator):
            return (line.replace('\\n', '\n') for line in iterator)

        if self.input_path is None:
            # Will understand backspace and the likes.
            # If world_size > 1, only the first rank reads from standard input and broadcasts its result to the
            # other ranks.
            assert not self.binary_input
            file = utils.read_stdin(interactive=True, rank=rank, world_size=world_size)
            # convert escaped line breaks to real line breaks, since many LLMs need those and real line breaks are used
            # as a separator between inputs.
            # InferenceCorpus could also be subclassed to support other types of files that do not use line breaks as a 
            # separator (e.g., JSON files).
            return convert_line_breaks(file)
        else:
            return File.open(format=self.file_formats[0], path=self.input_path)

    def ref_file(self) -> Optional[File]:
        if self.ref_path is None:
            return None
        else:
            return File.open(path=self.ref_path)


class Task:
    """
    Generic text generation class that should be subclassed.

    Handles the corpora and pre-processing (tokenization, batching), evaluation and logging for a given task.
    """
    # Subclasses should define those as attributes or properties:
    tgt_preprocessor: TextPreprocessor  # text generation tasks should all have a target preprocessor (and optionally 
    # a source one)
    preprocessors: dict[str, TextPreprocessor]
    model_type: str  # encoder_decoder or decoder

    def __init__(self, data_dir: str, cfg: TaskConfig):
        self.cfg = cfg
        self.model_type = None  # encoder_decoder or decoder, impacts how preprocessing is done (source and 
        # target need to be concatenated for decoder-only models)
        self.training = False
        self.data_dir = data_dir
        self.freeze_encoder_embed_mask = None  # used in Transformer & defined in TranslationTask

    def register_corpora(self, *corpora: Corpus) -> None:
        """ Add the languages or domains of the given corpora to the task. """
        raise NotImplementedError
    
    def make_meta(self, **kwargs: dict) -> dict:
        """
        Build a dictionary of metadata for this task (languages and domains) using given user options or default 
        values.
        This should cover all of the 'meta' keys that are returned by `Corpus` (or the relevant Corpus subclass).
        Raise an error if any option is unrecognized.
        """
        raise NotImplementedError

    def check_meta(self, meta: dict) -> None:
        """ Check whether the languages and domains specified in the dictionary are supported by the current task. """
        raise NotImplementedError

    def set_model_type(self, model_type: str) -> None:
        # This is not done in __init__ because the type of model might not be known yet when creating the task
        # (at inference, model checkpoints are loaded later)
        self.model_type = model_type
    
    @property
    def task_info(self) -> dict:
        """
        Used in `pasero-serve` to retrieve info about a model (e.g., the languages it covers). This should be consistent 
        with `get_langs_or_domains`.
        """
        return {'model_type': self.model_type}

    def get_langs_or_domains(self, key: str) -> set[str]:
        """
        Used by models (e.g., `AdapterTransformer`) to find available values for given metadata `key`. Keys are data 
        properties that can be used to define lang/domain-specific model parameters (with `--encoder-adapters-by` and 
        `--decoder-adapters-by`) and/or modify how the batching is done (with `--batch-by`)
        """
        raise NotImplementedError

    def input_to_sample(self, input: str, meta: dict) -> dict:
        """
        Called at inference to convert user inputs to samples that can be preprocessed and batched under this 
        task. Some tasks may interpret string inputs as a source to feed an encoder with, or as a prompt for a decoder.
        """
        raise NotImplementedError

    @property
    def special_tokens(self) -> SpecialTokens:
        return self.tgt_preprocessor.special_tokens
    @property
    def eos_idx(self) -> int:
        return self.special_tokens.eos_idx
    @property
    def padding_idx(self) -> int:
        return self.special_tokens.padding_idx
    @property
    def bos_idx(self) -> int:
        return self.special_tokens.bos_idx
    @property
    def unk_idx(self) -> int:
        return self.special_tokens.unk_idx
    @property
    def blacklist(self) -> list[int]:
        return self.tgt_preprocessor.blacklist
    @property
    def stop_sequences(self) -> list[LongTensor]:
        return self.tgt_preprocessor.stop_sequences

    @property
    def encoder_num_embeddings(self) -> int:
        """
        Used in models (e.g., `TransformerModel`) to find out the size of their encoder embedding matrix (if any),
        which depends on the task's vocabulary size.
        Should be implemented in subclasses.
        """
        raise NotImplementedError
    
    @property
    def decoder_num_embeddings(self) -> int:
        """
        Used in models (e.g., `TransformerModel`) to find out the size of their decoder embedding matrix (if any),
        which depends on the task's vocabulary size.
        Should be implemented in subclasses.
        """
        raise NotImplementedError

    def remap_encoder_embed(self, embed: Optional[Tensor]) -> Optional[Tensor]:
        """
        When loading a Transformer checkpoint (either at training or inference), this method is called to let the 
        task (optionally) modify the encoder embeddings.

        For instance, this can be used to remap embeddings after a change of dictionary (see TranslationTask's 
        `--old-source-dict` and `--old-target-dict` options).
        """
        return embed
    
    def remap_decoder_embed(self, embed: Optional[Tensor]) -> Optional[Tensor]:
        """
        When loading a Transformer checkpoint (either at training or inference), this method is called to let the 
        task (optionally) modify the decoder embeddings.
        """
        return embed

    def get_reference(self, sample: dict[str, Any]) -> Optional[str]:
        """
        Given a sample returned by `Corpus.tuple_to_dict`, return the 'reference' (AKA groundtruth) if any. Typically,
        this is `sample['target']`.
        These references will be used by `compute_score` to evaluate decoding hypotheses in the validation stage.
        """
        raise NotImplementedError
    
    def log_sample(self, sample_bin: dict) -> None:
        """ Used in `datasets` to periodically log some training and validation examples. """
        raise NotImplementedError
    
    def count_oov(self, sample_bin: dict) -> tuple[int, int]:
        """ Count the number of OOV and total tokens in given sample. """
        if 'target' in sample_bin:
            total = (sample_bin['target'] != self.padding_idx).sum()
            oov = (sample_bin['target'] == self.unk_idx).sum() if self.unk_idx != self.padding_idx else 0
            return oov, total
        else:
            return 0, 0

    def preprocess(
        self,
        sample: dict[str, Any],
        truncate: bool = False,
        tokenize: bool = True,
        inference: bool = False,
    ) -> dict[str, Any]:
        """
        Takes a raw sample (e.g., a line pair in TranslationTask), tokenizes and binarizes it using the task's
        tokenizers and dictionaries, then returns a dictionary that can be understood by `collate` (which builds a
        batch of several samples).
        `preprocess` can be called at training (by ValidationDataset and TrainingDataset), and at inference
        (by TextGenerator). This function has to take into account that the sample may contain less information at
        inference (e.g., missing target).

        Args:
            sample: dictionary corresponding to a single example that should be processed. Should contain at least 
                'target' (groundtruth for the decoder) and 'meta' (information about this example, like its language).
            truncate: whether to truncate examples that are too long
            tokenize: whether the text data should be tokenized
            inference: True when this method is called with inference inputs (by TextGenerator). The target may be 
                processed differently when it is intended for prompting rather than teacher forcing (e.g., no end of
                sequence symbols).

        Returns: a dict with binary data (numpy arrays) that can be given as input, along with other dicts, to
            `collate` to build batches of several examples. Should have at least 'target', 'prompt_mask' and 'meta'
            fields, and optionally a 'source' field (and more if the subclass's collater supports it).
            
            Note that the default collater will build the decoder input by shifting the target by one position to the 
            right and dropping its last token. So the last target token should either be EOS, or not important. This 
            is particularly important at inference, where the decoder input will be used for prompting.
        """
        raise NotImplementedError

    def postprocess(
        self,
        sample_bin: dict[str, Any],
        hypothesis: dict[str, Any],
        detokenize: bool = True,
    ) -> None:
        """
        Takes a binary sample as generated by `preprocess` as well as a decoding output as generated by 
        `decoding.search` and post-processes this decoding hypothesis by:
        1) debinarizing the generated token ids (i.e., array of integers -> sequence of text tokens)
        2) detokenizing (i.e., sequence of text tokens -> sequence of words)
        3) adding any other useful information (e.g., tokenized source) extracted from `sample_bin`
        """
        prompt_len = utils.mask_to_len(sample_bin['decoder_input'] != self.padding_idx) - 1  # remove 1 because 
        # the output does not include decoder_input's BOS

        tokens = hypothesis['tokens'].tolist()
        
        prompt_tokens = tokens[:prompt_len]
        prompt_tokens = self.tgt_preprocessor.debinarize(prompt_tokens, keep_padding=True)  # keeps EOS and tokens that
        # are after it (some chat templates may contain EOS in their prompt and we don't want to cut their outputs
        # in the middle of the prompt)
        
        new_tokens = tokens[prompt_len:]
        new_tokens = self.tgt_preprocessor.debinarize(new_tokens)
        
        tokens = ' '.join([prompt_tokens, new_tokens]).strip(' ')
        hypothesis['tokens'] = tokens

        if self.cfg.strip_prompt:  # remove the prompt tokens from the output before detokenizing
            tokens = new_tokens
        
        hypothesis['detok'] = (
            self.tgt_preprocessor.detokenize(tokens) if detokenize else
            hypothesis['tokens']
        )

    def train(self) -> None:
        """
        Set this task in training mode (preprocessors may behave differently between training and inference).
        See `TextPreprocessor.train`.
        """
        self.training = True
        for preprocessor in self.preprocessors.values():
            preprocessor.train()
    
    def eval(self) -> None:
        """
        Set this task in evaluation/inference mode (preprocessors may behave differently between training and
        inference). See `TextPreprocessor.eval`.
        """
        self.training = False
        for preprocessor in self.preprocessors.values():
            preprocessor.eval()

    @property
    def preprocessor_files(self) -> set[str]:
        """
        List of files used by this task's preprocessors. Copied to the model directory by `pasero-train` to ensure
        reproducibility.
        """
        paths = set()
        for preprocessor in self.preprocessors.values():
            paths.update(preprocessor.files)
        return paths
    
    @property
    def inference_options(self) -> dict:
        """
        Used in `pasero-train` to generate a YAML inference config file in the model directory ("inference.yaml").
        Should return all the relevant options (whose values are different from the default ones) for running the model 
        at inference (e.g., preprocessing options, task name, etc.)
        """
        raise NotImplementedError

    @classmethod
    def get_train_corpora(cls, cfg: TaskConfig, data_dir: str, corpus_definitions: list) -> list[Corpus]:
        """
        Uses given corpus definitions and task configuration to generate a list of training corpora for this task.

        Each task can have its own corpus definition format, but it has to be a dictionary with at least a 'paths'
        field. For instance:

        ```
        train_corpora:
            - paths: [some_corpus_names]
              langs: [some_langs]
              domain: some_domain
              ...
        ```   

        Note that corpora do not actually contain data, but paths and metadata that let datasets (
        `datasets.ValidationDataset` and `datasets.TrainingDataset`) find the relevant files, open them and read
        samples from them.

        This method has to be overriden in subclasses.
        """
        raise NotImplementedError

    @classmethod
    def get_valid_corpora(cls, cfg: TaskConfig, data_dir: str, corpus_definitions: list) -> list[Corpus]:
        """
        Uses given corpus definitions and task configuration to generate a list of validation corpora for this task.

        See `get_train_corpora` for more information.
        """
        raise NotImplementedError

    @classmethod
    def get_inference_corpora(
        cls,
        cfg: TaskConfig,
        input_paths: Optional[list[str]] = None,
        output_paths: Optional[list[str]] = None,
        ref_paths: Optional[list[str]] = None,
        corpus_prefix: Optional[str] = None,
    ) -> list[InferenceCorpus]:
        """
        Uses given paths and task configuration to generate a list of inference corpora for this task.

        Inference corpora are different than regular corpora. They have an optional reference and output path,
        and they are used by `TextGenerator.decode_corpus`. This function is called by `decode.py` to infer paths 
        and languages from a partial set of decoding options and initialize these corpora.
        
        For instance, input (`--input`) and reference paths (`--reference`) may be given, but not languages, which need
        to be inferred from the file extensions. As such, this function is also allowed to modify the task's
        configuration (e.g., its languages and domains).
        
        Or conversely, a corpus prefix may be given (`--eval-corpus`) and the paths may need to be inferred from the 
        task's languages.
        """
        raise NotImplementedError

    def get_collate_fn(self, dtype: str):
        return functools.partial(
            self.collate,
            special_tokens=self.special_tokens,
            dtype=getattr(torch, dtype),
            model_type=self.model_type,
        )

    @classmethod
    def collate(
        cls,
        batch: list[dict],
        special_tokens: SpecialTokens,
        dtype: torch.dtype,
        model_type: str,
    ) -> dict:
        """
        Create a padded batch from given a list of (binary) samples with variable sequence lengths.
        
        Also adds a "decoder_input" field, which is the target sequence shifted by one position (EOS is removed and BOS
        is added).
        """
        if not batch:
            return None

        targets = [sample['target'] for sample in batch]
        target_batch, target_length = tokens_as_tensor(targets, special_tokens, dtype=dtype)
        
        prompt_masks = [sample['prompt_mask'] for sample in batch]
        
        decoder_input, _ = tokens_as_tensor(targets, special_tokens, shift=True, dtype=dtype)
        # TODO: if special_tokens.bos_idx is None, shift decoder_input and target by one position
        prompt_mask, _ = tokens_as_tensor(prompt_masks, special_tokens, dtype=dtype)
        # map prompt mask to prompt length, examples:
        # [1, 1, 1, 0, 0] -> 3
        # [1, 0, 1, 1, 0] -> 4
        # 0s in the middle are considered as part of the prompt. For instance, when doing dialogue, the entire dialogue
        # except the last answer of the assistant would be the prompt. Note that this prompt length is used at 
        # evaluation, where it wouldn't make sense to ask for the model to generate user tokens. For computing the 
        # training loss, the more finegrained "prompt_mask" is used.
        prompt_length = mask_to_len(prompt_mask) + 1    # add one because prompt_mask doesn't account for BOS
        # prompt_mask[:,1:] = prompt_mask[:,:-1].clone()  # shift like 'decoder_input'
        # TODO: assert that prompt_length >= 1 (we need something to feed the decoder with)

        meta = batch[0]['meta']
        # Only keep info that is common to all samples in the batch, which should be the case if --batch-by is set 
        # correctly. We do this to avoid silent errors later, where samples could be attributed to the wrong language or 
        # domain
        meta = {k: v for k, v in meta.items() if all(sample['meta'].get(k) == v for sample in batch[1:])}
        return {
            'target': target_batch,
            'decoder_input': decoder_input,
            'target_length': target_length,
            # assumes all samples in the batch share the same metadata (use --batch-by to ensure this is the case)
            'meta': meta,
            'prompt_mask': prompt_mask,
            'prompt_length': prompt_length,
        }

    def build_batches(
        self,
        data: list[dict],
        shuffle: bool = True,
        sort: bool = True,
    ) -> list[list[int]]:
        """
        Sort given samples by length and generate batches under given constraints. The returned batches are only lists
        of indices in `data`.
        """
        if self.model_type == 'encoder_decoder':
            source_length = np.array([len(sample['source']) for sample in data])
            target_length = np.array([len(sample['target']) for sample in data])
            length = np.maximum(source_length, target_length)
            indices = np.random.permutation(len(data)) if shuffle else np.arange(len(data))
            if sort:
                indices = indices[np.argsort(target_length[indices], kind='stable')]
                indices = indices[np.argsort(source_length[indices], kind='stable')]
        elif self.model_type == 'decoder':
            length = np.array([len(sample['target']) for sample in data])
            indices = np.random.permutation(len(data)) if shuffle else np.arange(len(data))
            if sort:
                indices = indices[np.argsort(length[indices], kind='stable')]
        else:
            raise NotImplementedError
        return utils.build_batches(
            indices,
            length.__getitem__,
            self.cfg.batch_size,
            self.cfg.batch_size_multiple,
            self.cfg.lines_per_batch,
        )

    @classmethod
    def shard_batch(cls, batch: dict, shard_id: int = 0, shard_count: int = 1):
        """
        Used to distribute given batch across different workers in a data-parallel setup (typically used with
        mixtures of experts at inference). This method ensures that all workers have a (non-empty) batch by allowing
        "dummy" batches.

        Note that it is very similar to `datasets.shard_batch`. The main difference is that this allows for batch sizes 
        that are not multiples of `shard_count`
        """
        dummy_batch = {'dummy': True}
        sharded_batch = {'dummy': False}
        dummy = False
        
        for k, v in batch.items():
            if k == 'meta':  # all items are tensors whose first dimension is the batch size, except this one
                dummy_batch[k] = v
                sharded_batch[k] = v
                continue

            bsz = len(v)
            shard_size = bsz // shard_count
            shard_start = shard_id * shard_size
            shard_end = (shard_id + 1) * shard_size if shard_id < shard_count - 1 else None

            dummy_batch[k] = v[:1]
            sharded_batch[k] = v[shard_start:shard_end]
            
            if len(sharded_batch[k]) == 0:  # make sure all GPUs are receiving the same number of batches
                # (some architectures, like MoEs, or tensor parallelism require GPUs to sync with each other. If
                # a GPU stops computation too early, we'll have a deadlock.
                dummy = True

        return dummy_batch if dummy else sharded_batch

    def debinarize_on_the_fly(self, token_ids: Iterable[int]) -> Iterator[str]:
        """ See `TextPreprocessor.debinarize` """
        for token_id in token_ids:
            yield self.tgt_preprocessor.debinarize([token_id], keep_padding=True)
        
    def detokenize_on_the_fly(self, tokens: Iterable[str]) -> Iterator[tuple[str, list[str]]]:
        """ See `TextPreprocessor.detokenize_on_the_fly` """
        yield from self.tgt_preprocessor.detokenize_on_the_fly(tokens)

    def compute_score(
        self,
        metric: str,
        hypotheses: list[dict[str, Any]],
        references: list[str],
        merge_bpe: bool = False,
        **eval_opts,
    ) -> Optional[float]:
        """
        Used at training and inference to compute given metric for given set of decoding hypotheses and references.
        While the metric computation may seem straightforward, each task may have a different post-processing or
        normalization to apply to the hypotheses or references before computing the metric (in addition to the 
        post-processing done in `task.postprocess`).
        For instance, `DocumentLevelTranslationTask` computes its metrics on the last sentence of each generated 
        document.
        """
        hypotheses = [hyp['detok'] for hyp in hypotheses]
        if merge_bpe:  # assumes there is no target tokenizer: the evaluation references and training targets are 
            # pre-tokenized
            hypotheses = [detokenize(hyp) for hyp in hypotheses]
            references = [detokenize(ref) for ref in references]
        return evaluation.safe_score(
            metric=metric,
            hyps=hypotheses,
            refs=references,
            **eval_opts,
        )
    
    def hypothesis_to_str(self, hypothesis: dict, verbose: bool = False) -> str:
        """
        Converts a decoding hypothesis represented as a dict, into a string that can be printed on the screen or 
        written into a text file.
        Such hypothesis is obtained by running `postprocess` on decoding outputs.

        `Task.hypothesis_to_str` is generic and works with TranslationTask, SpeechTranslationTask and 
        LanguageModelingTask, but new tasks may have to override this method.
        """
        if verbose:
            s = []

            line_id = hypothesis.get('idx', 0)
            src_tok = hypothesis.get('src_tokens')  # absent with decoder-only models
            hyp_tok = hypothesis['tokens']
            hyp_detok = hypothesis['detok']
            global_score = hypothesis['score']
            pos_scores = hypothesis.get('pos_scores')
            cross_attn = [
                v.mean(axis=1) for k, v in hypothesis.items() if k.startswith('dec_') and k.endswith('_cross_attn')
            ]
            
            if isinstance(src_tok, str):  # not str in SpeechTranslationTask
                s.append(f'S-{line_id}\t{src_tok}')
                
                if cross_attn:
                    cross_attn = sum(cross_attn) / len(cross_attn)
                    tag_mask = np.array(
                        [token[0] != '<' or token[-1] != '>' for token in src_tok.split(' ')]
                    )
                    # alignment picks the top source token which is not a tag, per target position (we do this
                    # because some tags like EOS can concentrate a large part of the attention mass, and are 
                    # not very informative)     
                    tag_mask = tag_mask[:cross_attn.shape[1]]
                    
                    alignment = (cross_attn * np.expand_dims(tag_mask, 0)).argmax(axis=1)
                    alignment = ' '.join(map(str, alignment))
                    s.append(f'A-{line_id}\t{alignment}')

            s.append(f'H-{line_id}\t{hyp_tok}')
            s.append(f'D-{line_id}\t{hyp_detok}')
            if pos_scores is not None:
                s.append(f"P-{line_id}\t{global_score:.3f}\t" + ' '.join(f'{score:.3f}' for score in pos_scores))

            return '\n'.join(s)
        else:
            return hypothesis['detok']

    def load_checkpoint_for_inference(
        self,
        *ckpt_paths: str,
        rank: int = 0,
        world_size: int = 1,
        arch: Optional[str] = None,
    ) -> tuple[dict, TransformerConfig]:
        """
        Used at inference by `decoding.TextGenerator` to load a model's checkpoint and hyper-parameters; and optionally
        parallelize it with given settings.
        One key different to training is that the model hyper-parameters are not specified, but loaded from the model's
        checkpoint. So we do not know the model architecture in advance (unless it explicitely specified with `arch`)

        For now, this does not support checkpoint resharding: i.e., if the model is sharded, the checkpoint has to 
        be sharded too and have the same number of shards (and conversely)

        Args:
            - ckpt_paths: checkpoints to load
            - rank: distributed rank. If the model is sharded, this corresponds to the model shard id
            - world_size: distributed world size (i.e., number of GPUs). If the model is sharded this corresponds to 
                number of model shards
            - arch: override the architecture defined in the checkpoint (if any) with this one
        Returns:
            tuple (checkpoint, model_cfg) where checkpoint is a dict containing all the model weights for this rank,
                and model_cfg a model configuration containing the hyper-parameters
        """
        assert len(ckpt_paths) >= 1
        main_ckpt_path, *other_ckpt_paths = ckpt_paths
        # This is also different from the type of sharding implemented in NLLBTranslationTask, which requires one 
        # checkpoint per expert (for loading the NLLB-200 MoE model)
        shard_paths = utils.find_checkpoint_shards(main_ckpt_path)
        assert len(shard_paths) == 1 or len(shard_paths) == world_size, ('the number of checkpoint shards does not '
            'match the number of GPUs, use `scripts/merge-tp-ckpt.py` or `scripts/merge-tutel-ckpt.py`')
        # Load the shard corresponding to this rank:
        main_ckpt_path = shard_paths[0] if len(shard_paths) == 1 else shard_paths[rank]
        logger.info(f"loading checkpoint {main_ckpt_path}")
        checkpoint = utils.load_checkpoint(main_ckpt_path, *other_ckpt_paths)
        model_steps = checkpoint.get('steps', 0)
        logger.info(f"loaded checkpoint {main_ckpt_path} @{model_steps}")
        # TODO: it is very hard to reshard the model like in Trainer, because we don't know the architecture in advance
        # (i.e., before loading the checkpoint). Should we attempt to read this architecture from 'training.yaml' and
        # 'inference.yaml' instead of the model args? Or just assume it is specified in 'inference.yaml' if it is not 
        # the default "transformer" arch?
        # In particular, we cannot know if the model is sharded before knowing what type of model this is. If it is 
        # a regular model (no Tutel and no MoE), `shard_count` and `shard_id` should be ignored.

        # read model hyper-parameters from model checkpoint
        model_args = {}
        if checkpoint.get('args') is not None:
            model_args = checkpoint['args']
        elif checkpoint.get('cfg') is not None:  # recent versions of fairseq
            model_args = checkpoint['cfg']['model']
        if isinstance(model_args, argparse.Namespace):  # older versions of fairseq
            model_args = vars(model_args)
        if model_args:
            utils.convert_from_fairseq(model_args)

        arch = arch or model_args.get('arch')
        assert arch is not None, 'could not find model architecture in checkpoint, use --arch'
        model_cfg = get_model_config(arch)
        model_cfg.parse_dict(model_args)
        model_state = checkpoint['model']
        return model_state, model_cfg

    def prepare_model_for_inference(self, model: nn.Module, meta: dict) -> None:
        """
        Called by TextGenerator to modify the given model and adapt it to given languages and domains. For instance, 
        NLLBTranslationTask can load a different set of experts depending on the language pair (AKA language-specific
        expert pruning)
        """
        pass  # this method does nothing by default

    @classmethod
    def get_inference_corpus(cls, *args, **kwargs) -> InferenceCorpus:
        """
        Should be implemented by subclasses to let the user create an inference corpus for this task, given an input
        file and optional reference and output files. The returned corpus can be used as input to
        `TextGenerator.decode_corpus`
        """
        raise NotImplementedError
