# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import os
import sys
import io
import itertools
import logging
import torch
import math
import numpy as np
import regex
import tempfile
import time
import torch.distributed as dist
import torch.nn as nn
from typing import Optional, Union, Any, Iterator
from torch import Tensor, LongTensor, BoolTensor

from pasero import utils, models
from pasero.tasks import Task, InferenceCorpus
from pasero.models import Encoder, Decoder, EncoderDecoder, Transformer, fast_init
from pasero.config import get_architecture, get_task_class, DecodingConfig, DecodingAPIConfig


logger = logging.getLogger('decoding')


class TextGenerator:

    @classmethod
    def build(cls, model_path: str, start: bool = True, **kwargs) -> 'TextGenerator':
        """
        API-like interface: loads a model for inference using its path and keyword arguments
        (rather than an already parsed `DecodingConfig`)

        `model_path` can be the path to a model directory or to a model checkpoint

        For example:
        
        ```
        generator = TextGenerator.build('models/TED/en-de', batch_size=1000, beam_size=3, max_output_len=50)
        output = generator.decode('Hello , world !', beam_size=1)  # note that some arguments can be specified
        # again here
        print(output[0][0]['detok'])
        ```
        
        The list of languages and domains needs to be known in advance to load potential language-specific
        dicts, tokenizers, and model parameters (e.g., adapters).
        It can be defined in the model's decoding config file ('inference.yaml') or as keyword arguments to this
        function ('source_langs', 'target_langs', 'langs', 'domains')
        """
        cfg = DecodingAPIConfig(model=model_path, **kwargs)
        return cls(cfg, start=start)

    def sync_seed(self):
        """
        Called by `pasero-decode` to make sure that all GPUs have the same seed
        """
        self.cfg.seed = utils.broadcast(self.cfg, self.cfg.seed, dtype=torch.int64)
        utils.set_random_seed(self.cfg.seed)

    def __init__(self, cfg: DecodingAPIConfig, start: bool = True):
        cls_name = self.__class__.__name__
        assert isinstance(cfg, DecodingAPIConfig), (
            f"{cls_name}(...) takes a DecodingAPIConfig instance. To initialize it with a model path, use "
            f"{cls_name}.build(...)"
        )

        # Note: `cfg` contains decoding arguments but not the model hyper-parameters. The later are loaded from
        # the model checkpoint in self.load_model(...), and stored under self.model_cfg.
        # To modify model hyper-parameters at decoding time, use the --model-args option.
        # Declaring a new decoding option with the same name as a model option is not enough,
        # unless modifying `TransformerConfig.setup_for_inference`
        logger.info(f'decoding arguments: {cfg.as_dict()}')

        if cfg.seed is not None:
            utils.set_random_seed(cfg.seed)  # sets the PyTorch, numpy and Python seeds

        self.dtype = getattr(torch, cfg.dtype)
        if torch.cuda.is_available():
            self.devices = cfg.devices or ['cuda']
        else:
            self.devices = ['cpu']

        cfg.sequence_parallel = False  # not supported at inference
        self.data_parallel = cfg.dp_size > 1  # only the first rank will read data and batches will be sharded and 
        # scattered to all ranks
        self.tensor_parallel = cfg.tp_size > 1
        self.pipeline_parallel = len(self.devices) > 1
        self.rank = max(cfg.tp_rank, cfg.dp_rank)
        self.world_size = max(cfg.tp_size, cfg.dp_size)
        cfg.task_cfg.batch_size_multiple = cfg.dp_size  # to be able to shard the batches

        for device in self.devices:
            if device == 'cpu':
                assert self.dtype == torch.float32, \
                    'half-precision decoding is not supported on CPU: use --dtype float32'
            else:
                assert self.dtype != torch.bfloat16 or torch.cuda.get_device_capability(device)[0] >= 8, \
                    'the bfloat16 data type is not supported by your device: use --dtype float16'

        at_most_one_true = lambda *conditions: sum(conditions) <= 1
        assert at_most_one_true(
            self.data_parallel,
            self.tensor_parallel,
            self.pipeline_parallel,
        ), 'combining different types of parallelism is not allowed at inference'
        
        if self.data_parallel or self.tensor_parallel:
            assert dist.is_initialized(), "data and model parallelism should be used only from 'pasero-decode' " \
                "the decoding API only supports pipeline parallelism"

        if self.tensor_parallel or self.pipeline_parallel:
            assert not cfg.encoder_decoder_swapping, 'CPU offloading is incompatible with model parallelism'

        if cfg.benchmark and utils.is_master(cfg):
            utils.benchmark.enable()
        
        task_cls = get_task_class(cfg.task)
        self.task: Task = task_cls(cfg.model_dir, cfg.task_cfg)

        self.cfg = cfg
        self.metrics = utils.Metrics(history_size=-1)
        self.model = None
        
        if start:
            self.start_model()

    @property
    def model_info(self):
        """
        Used in `pasero-serve` to retrieve information about a model (its size, type, task, languages, default,
        decoding options, etc.)
        """
        param_count = 0 if self.model is None else self.model.total_param_count
        decoding_opts = DecodingConfig(self.cfg).as_dict()
        defaults = DecodingConfig().as_dict()
        decoding_opts = {k: v for k, v in decoding_opts.items() if v != defaults.get(k)}
        return {
            'name': os.path.basename(self.cfg.model_dir),
            'decoding_options': decoding_opts,
            **self.task.task_info,
            'task': self.cfg.task,
            'max_len': self.model_cfg.decoder_max_len,
            'param_count': param_count,
        }

    @property
    def encoder(self) -> Encoder:
        return self.model.encoder
    
    @property
    def decoder(self) -> Decoder:
        return self.model.decoder

    @torch.inference_mode()
    def load_model(self, main_ckpt_path: str, *other_ckpt_paths: str) -> Transformer:
        """
        Load model configuration and parameters from the given checkpoint, initialize a model with use and return 
        this model.
        The model parameters loaded from the other given checkpoints are merged into the main checkpoint: this is 
        useful to load trained adapters (the main checkpoint) and the base model's parameters (other checkpoint).
        In case of sharded checkpoint (e.g., for tensor parallelism), only the first shard should be given and the other
        shards will automatically be found.
        """
        model_state, model_cfg = self.task.load_checkpoint_for_inference(
            main_ckpt_path,
            *other_ckpt_paths,  # these will be merged into the main checkpoint
            rank=self.rank,
            world_size=self.world_size,
            arch=self.cfg.arch,
        )

        # FIXME: these 3 things should all be done in a single call
        model_cfg.setup_for_inference(self.cfg)
        # The maximum source and target lengths are None by default and set automatically from the model's settings
        # when loading it:
        self.task.cfg.set_max_length(model_cfg)  # task.max_len / task.max_source_len / task.max_target_len are 
        # properties, so this should work
        self.task.setup_for_model(model_cfg)

        logger.debug(f"model arguments: {model_cfg.as_dict()}")

        arch_cls = get_architecture(model_cfg)
        
        # Save time and CPU/GPU memory by directly creating parameters of the right type and on the right device
        device = None if self.pipeline_parallel or self.cfg.encoder_decoder_swapping else self.devices[0]
        with fast_init(device=device, dtype=self.dtype):
             model: Transformer = arch_cls(model_cfg, dist_cfg=self.cfg, task=self.task)

        logger.debug(model)
        for name, param in model.named_parameters():
            shape = 'x'.join(map(str, param.shape))
            logger.debug(f'{name} ({shape})')
        logger.info('finished building model')
        
        model.eval()   # used by model.load_state_dict
        model = model.to(self.dtype)
        
        with fast_init(device=device, dtype=self.dtype):
            model.update_state_dict(model_state)
            model.remap_state_dict(model_state)  # for --old-source-dict and --old-target-dict
            model.load_state_dict(model_state, strict=not self.cfg.flexible)
        
        return model

    @torch.inference_mode()
    def start_model(self) -> None:
        """
        Load the model and put it into GPU memory (if applicable). This is not done automatically when building a
        `TextGenerator` (unless `start=True`) to save time when the model is actually not needed (e.g., decoding 
        empty files)
        """
        if self.model is not None:  # do not start the model if it's already running
            return
        models: list[Transformer] = []
        for ckpt in self.cfg.ckpt, *self.cfg.ensemble_ckpt:
            models.append(self.load_model(ckpt, *self.cfg.other_ckpt))
        self.model = models[0] if len(models) == 1 else EnsembleModel(models)
        self.model_cfg = models[0].cfg

        logger.info(f'total params {self.model.total_param_count:,}')
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        if encoder_params:
            logger.info(f'encoder params {encoder_params:,}')
        if decoder_params:
            logger.info(f'decoder params {decoder_params:,}')
    
        if self.cfg.encoder_decoder_swapping:
            # the encoder and decoder are put on the GPU alternately (once the encoder is finished encoding
            # a batch, it is moved to the CPU and the decoder is moved to the GPU)
            self.encoder.to(self.devices[0])
        else:
            # move the model to given device(s): does pipeline parallelism if more than one device
            self.model.parallelize(self.devices)

    def preprocess(
        self,
        samples: list[dict],
        *,  # below = keyword-only arguments
        append_eos: bool = False,
        tokenize: bool = True,
        sort: bool = True,
        **kwargs,  # unused
    ) -> list[dict]:
        """
        Preprocess given samples and build batches with this model's task. The preprocessing and batching configuration
        are handled by the task.
        
        If using data parallelism, the same batches are read on all replica, but each replica only keeps its own shard 
        of each batch. For example:
        ```
        batch = [a, b, c, d]
        GPU 0 -> [a, b]
        GPU 1 -> [c, d]
        ```
        """
        samples = [
            self.task.preprocess(
                sample,
                truncate=True,
                tokenize=tokenize,
                append_eos=append_eos,
            )
            for sample in samples
        ]

        assert all(len(sample['decoder_input']) > 0 for sample in samples), 'there is nothing to prompt the decoder ' \
            'with, please set --bos-idx to a positive value'

        batches = self.task.build_batches(samples, shuffle=False, sort=sort)

        collate_fn = self.task.get_collate_fn(self.dtype)
        for i, batch in enumerate(batches):
            batch = collate_fn(batch)
            if self.data_parallel:
                # in case of data parallelism, find out which part of the current batch will go on this GPU
                batch = self.task.shard_batch(batch, shard_id=self.cfg.dp_rank, shard_count=self.cfg.dp_size)
            batches[i] = batch

        return batches
    
    def postprocess(
        self,
        batches: list[dict],
        *,
        detokenize: bool = True,
    ) -> list[list[dict]]:
        """
        Called at the end of decoding to reorder the decoding hypotheses and postprocess them with `self.task`.
        This can include: stripping special tokens or prompts, debinarization and detokenization.
        This also converts torch tensors into numpy arrays.
        
        Each batch should contain an 'outputs' key, of type `list[list[dict]]`, whose first dimension is the batch size,
        second dimension is the beam size (or 1 if beam search is not used), and whose dicts contain the binary
        outputs and scores.
        An object of the same shape is returned, except that its dictionaries correspond to post-processed hypotheses.

        Batches should also have an 'indices' key, indicating the index of each element in the original dataset (since
        sorting by length is done to reduce padding).
        """
        n = sum(len(batch['indices']) for batch in batches)
        hyps_reordered = [None] * n

        # Re-order the hypotheses and add source info
        for batch in batches:
            outputs = batch.pop('outputs')  # B * K
            real_indices = batch.pop('indices')
            meta = batch.pop('meta')
            dummy = batch.pop('dummy', False)

            if dummy:
                continue

            for i, (real_idx, nbest) in enumerate(zip(real_indices, outputs)):
                sample_bin = {
                    k: v[i] for k, v in batch.items()
                }
                sample_bin['meta'] = meta
                # debinarizes and detokenizes the hypotheses and adds other 'informational' fields
                # (e.g., tokenized source, etc.)
                for hyp in nbest:
                    self.task.postprocess(sample_bin, hyp, detokenize=detokenize)

                hyps_reordered[real_idx] = utils.tensor_to_array(nbest)

        # len(sources) * beam_size * {'tokens': str, **layer_outputs}
        return [nbest for nbest in hyps_reordered if nbest is not None]  # filter out None, which correspond to dummy 
        # batches (happens with data parallelism when the number of batches is not a multiple of the number of GPUs)

    @torch.inference_mode()
    def stream(
        self,
        input: Union[str, np.ndarray],
        *,
        tokenize: bool = True,
        detokenize: bool = True,
        stop_regex: Optional[str] = None,
        **decoding_opts,
    ) -> Iterator[dict[str, Any]]:
        """
        Unlike `decode`, this methods takes a single input and generates the model output token by token or word
        by word. Beam-search decoding is not supported.
        
        Args:
            - input: single source sentence or prompt, or numpy array containing audio features
            - tokenize: tokenize the input text. If False, assumes it is already tokenized
            - detokenize: detokenize the output token. If False, tokens are generated at a lower granularity (subwords
                instead of words, and the output dictionaries have no 'detok' key
            - stop_regex: generation will be stopped if the detokenized output matches this regex
            - decoding_opts: dictionary containing optional decoding options that will override the default ones (e.g.,
                'sampling', 'beam_size', 'max_output_len', etc.), can also contain task-specific metadata
                (e.g., 'lang', 'source_lang', 'target_lang', 'domain')
        
        Returns: iterator of dictionaries of
            `{'detok': word, 'tokens': [subwords], 'scores': [probs]}`  if detokenize is True;
            `{'tokens': [subword], 'scores': [prob]}`                   otherwise
        """
        # FIXME: is CUDA memory automatically freed when interrupting generation? It seems like when the thread is 
        # deleted, so is the generator and everything that was allocated in it.
        # This seems to work somewhat magically with multi-threading: the generation is slower but happens in parallel.
        
        assert not self.cfg.encoder_decoder_swapping, 'CPU offloading is incompatible with on-the-fly generation'
        assert not self.data_parallel and not self.tensor_parallel, 'Data and tensor parallelism are incompatible ' \
            'with on-the-fly generation'

        if stop_regex and not detokenize:
            utils.warn_once(
                logger,
                "the 'stop_regex' option operates on detokenized outputs, but 'detokenize' is False"
            )
        
        # Make a copy of the model's configuration and modify it to change decoding options (e.g., beam size, 
        # max output length, etc.)
        cfg = DecodingConfig(self.cfg)   # FIXME: this type of instantiation doesn't work with the "defaults" attribute,
        # "set_defaults" should be called
        unknown_opts = cfg.parse_dict(decoding_opts)
        meta = self.task.make_meta(**unknown_opts)  # options that are not in DecodingConfig are assumed to be task 
        # metadata (e.g., lang or domain)
        self.task.check_meta(meta)  # check that the given languages and domains are supported by the model
        
        if not cfg.sampling:
            cfg.sampling_temperature = 0

            if cfg.beam_size > 1:
                utils.warn_once(logger, 'beam search does not implement on-the-fly generation, disabling beam search')
                cfg.beam_size = 1

        self.start_model()

        self.task.prepare_model_for_inference(self.model, meta)

        batch = self.preprocess(
            samples=[self.task.input_to_sample(input, meta)],
            tokenize=tokenize,
            sort=False,
        )[0]   # single batch of a single example

        batch = utils.move_to_device(batch, self.devices[0])
        encoder_outputs, encoder_masks, _ = self.encoder(
            **batch,
        )

        opts = dict(
            encoder_outputs=encoder_outputs,
            encoder_masks=encoder_masks,
            **batch,
            device=self.devices[-1],
            return_scores=True,
            blacklist=self.task.blacklist,
            stop_sequences=self.task.stop_sequences,
            **cfg.as_dict(),
        )

        output = []
        
        def token_iterator():
            for out in sample_on_the_fly(self.decoder, **opts):
                if out and 'tokens' in out:
                    output.append(out)
                    yield out['tokens'][0, -1].item()  # `sampling` supports batching but here we don't
        
        iterator = token_iterator()  # iterator over generated tokens, which also saves the entire outputs (pos scores,
        # etc.) in a list as a side effect 

        iterator = self.task.debinarize_on_the_fly(iterator)
        if detokenize:
            iterator = self.task.detokenize_on_the_fly(iterator)
        else:
            iterator = ((None, [token]) for token in iterator)

        count = 0
        text = ''
        while True:
            try:
                start = time.time()
                word, tokens = next(iterator)
                elapsed = time.time() - start
            except StopIteration:
                break

            partial_output = output[count:count + len(tokens)]
            
            scores = [out['pos_scores'][0, -1].item() for out in partial_output]
            out = {
                'tokens': tokens,
                'scores': scores,
                'elapsed': elapsed,
            }
            
            if count == 0:
                prompt_tokens = output[0]['tokens'][0, :-1].tolist()
                out['prompt_scores'] = output[0]['pos_scores'][0, :-1].tolist()
                out['prompt_tokens'] = list(self.task.debinarize_on_the_fly(prompt_tokens))

            if word is not None:
                out['detok'] = word
                text += word
            
            yield out

            count += len(tokens)
            if stop_regex and regex.search(stop_regex, text):
                break

    @torch.inference_mode()
    def decode(
        self,
        *inputs: Union[str, np.ndarray],
        tokenize: bool = True,
        detokenize: bool = True,
        return_scores: bool = False,
        return_layers: list[str] = [],
        targets: Optional[list[str]] = None,
        **decoding_opts,
    ) -> list[list[dict]]:
        """
        Perform batched decoding (translation or generation) from given inputs.
        Note that if the model is decoder-only, inputs are interpreted as prompts. If it is an encoder-decoder model,
        they are intepreted as sources for the encoder. Optionally, they can contain a decoder prompt after a '|||'
        delimiter. For instance:
            `model.decode('She sells seashells by the seashore.|||Elle distribue')`
            -> 'Elle distribue des coquillages près de la mer.'

        Args:
            - inputs: one or several inputs to translate or use as prompts. They will be batched according to the
                decoding configuration, or the options given in `decoding_opts` ('batch_size', 'lines_per_batch')
            - return_scores: whether to return the score of each output token (this can slow down decoding a bit)
            - return_layers: names of the layers whose outputs should be returned (e.g., f'enc_0_self_attn' for the
                self-attention scores at the first encoder layer)
            - decoding_opts: dictionary containing optional decoding options that will override the ones defined when
                loading the model (e.g., 'sampling', 'beam_size', 'max_output_len', etc.), can also contain
                task-specific metadata (e.g., 'lang', 'source_lang', 'target_lang', 'domain')
        
        Returns: a list of the same length as 'inputs', containing the n-best hypotheses for each input.
            Each hypothesis is a dictionary of this form:
            ```
            {
                'detok': 'Detokenized output',
                'tokens': 'Tokenized output',
                'score': score_of_the_hypothesis,
                'pos_scores': [score_of_each_token],
                **layer_outputs,
            }
            ```
            If beam search is not used, each input will generate a single hypothesis.
        """
        assert inputs, 'empty input'
        
        # Make a copy of the model's configuration and modify it to change decoding options (e.g., beam size, 
        # max output length, etc.)
        cfg = DecodingConfig(self.cfg)
        unknown_opts = cfg.parse_dict(decoding_opts)
        meta = self.task.make_meta(**unknown_opts)  # options that are not in DecodingConfig are assumed to be task 
        # metadata (e.g., lang or domain)
        self.task.check_meta(meta)  # check that the given languages and domains are supported by the model

        # Start the model if not already started
        self.start_model()
        
        self.task.prepare_model_for_inference(self.model, meta)

        return_layers = return_layers or []
        samples = [self.task.input_to_sample(input, meta) for input in inputs]  # user input can be interpreted as a
        # decoder prompt or as an encoder source depending on the type of model and task

        if targets:  # teacher forcing
            cfg.max_output_len = 0
            append_eos = True
            assert len(samples) == len(targets)
            if samples and samples[0].get('target'):
                utils.warn_once(logger, 'input will be ignored because a target was provided')
            for sample, target in zip(samples, targets):
                sample['target'] = target
        else:
            append_eos = False
        
        if cfg.max_output_len == 0:
            cfg.beam_size = 1
            cfg.sampling = False

        batches: list[dict] = self.preprocess(
            samples=samples,
            append_eos=append_eos,
            tokenize=tokenize,
        )

        def encode(batch: dict) -> dict:
            batch = utils.move_to_device(batch, self.devices[0])
            batch['encoder_outputs'], batch['encoder_masks'], batch['enc_layer_outputs'] = self.encoder(
                **batch,
                return_layers=return_layers,
            )
            return batch

        def decode(batch: dict) -> list[list[dict]]:
            outputs = search(
                self.decoder,
                **batch,
                return_layers=return_layers,
                return_scores=return_scores,
                sharded=self.data_parallel or self.tensor_parallel,
                device=self.devices[-1],
                blacklist=self.task.blacklist,
                stop_sequences=self.task.stop_sequences,
                **cfg.as_dict(),
            )
            
            # these keys were only added temporarily and are not needed for post-processing
            batch.pop('encoder_outputs', None)
            batch.pop('encoder_masks', None)
            batch.pop('enc_layer_outputs', {})

            if batch.get('dummy'):
                outputs = []
            batch['outputs'] = outputs
            return utils.move_to_cpu(batch)

        # Decode all the batches one by one
        with self.metrics.timer('wall'):
            if self.cfg.encoder_decoder_swapping:
                # save GPU memory by generating all the encoder outputs, then moving the encoder to CPU and the
                # decoder to GPU before computing decoder outputs
                batches = list(batches)
                batches = [
                    utils.move_to_cpu(encode(batch))
                    for batch in batches
                ]
                self.encoder.to('cpu')
                self.decoder.to(self.devices[0])
                batches = [decode(batch) for batch in batches]
                self.decoder.to('cpu')
                self.encoder.to(self.devices[0])
            else:
                batches = [decode(encode(batch)) for batch in batches]

        if self.data_parallel:
            torch.cuda.empty_cache()
            batches = utils.gather_list(self.cfg, batches)
        
        # Re-order the hypotheses and add source info
        hyps = self.postprocess(batches, detokenize=detokenize)

        # Compute statistics
        num_words = num_tokens = score = 0
        for nbest in hyps:
            best_hyp = nbest[0]
            score += best_hyp.get('score', 0)
            num_words += len(best_hyp['detok'].split())
            num_tokens += len(best_hyp['tokens'])

        # Update metrics
        self.metrics.update('num_words', num_words)
        self.metrics.update('num_tokens', num_tokens)
        self.metrics.update('num_lines', len(hyps))
        self.metrics.update('steps', len(batches))
        self.metrics.update('loss', score / math.log(2))
        return hyps

    def decode_corpus(
        self,
        corpus: InferenceCorpus,  # obtained by task.get_inference_corpora
        buffer_size: int = 100,
        bleu_tok: Optional[str] = None,
        eval_lc: bool = False,
        continue_: bool = False,
        verbose: bool = False,
        metrics: list[str] = ['chrf', 'bleu'],
        max_lines: Optional[int] = None,
        teacher_forcing: bool = False,
        return_layers: list[str] = [],
        **decoding_opts,
    ) -> list[list[dict]]:
        """
        Decode an entire corpus and optionally compute metrics.

        Args:
            - corpus: obtained by calling `task.get_inference_corpus`. Contains at least an input file (can be standard
                input) and optional an output file and reference file.
            - buffer_size: how many lines to read at once from corpus before batching and decoding. Larget buffers are
                more efficient because lines in the buffer are sorted by length before batching to minimize padding
            - continue_: if True and the output file already exists, count the number of lines in it and skip this many
                input lines. If False, the output file is overwritten.
            - verbose: if False, only the detokenized hypotheses are written to the output file. If True, other
                information, like tokenized source, tokenized output, scores and alignment are written.
            - metrics: which metrics to compute between the decoding output and the references (if any)
            - max_lines: truncate the input to this many lines
            - teacher_forcing: instead of auto-regressive decoding from given inputs, force the model to generate the 
                corpus' references. This can be used to a compute a test loss or perform an alignment using the cross-
                attention matrix
            - return_layers: names of the layers whose outputs should be returned

        Returns: the decoding outputs (in addition to writing them to given output file or to standard output).
            See `decode()` for a description of these outputs.

        Metrics are stored in `self.metrics`. Example:

        ```python
        model = TextGenerator.build('models/NLLB-200/600M_distilled.bin')
        corpus = model.task.get_inference_corpus(
            source_path='data/FLORES/FLORES-valid.eng_Latn',
            ref_path='data/FLORES/FLORES-valid.fra_Latn',
            output_path=False,  # set to None to write to standard output
            source_lang='eng_Latn',
            target_lang='fra_Latn',
        )
        out = model.decode_corpus(corpus)
        out[0][0]['detok']  # "Lundi, des scientifiques de l'École de médecine de l'Université de Stanford [...]"
        model.metrics.val('bleu')  # 46.18
        ```
        """
        input_path = corpus.input_path
        output_path = corpus.output_path
        ref_path = corpus.ref_path

        if corpus.binary_input:
            assert input_path, '--input is required for binary formats'
            assert not continue_, '--continue is not compatible with binary inputs'
        
        if continue_:
            assert input_path, '--continue requires --input'
            assert output_path, '--continue requires --output'

        self.task.check_meta(corpus.meta)  # check that this corpus's languages and domains are supported by the model
        decoding_opts = {**corpus.meta, **decoding_opts}  # will be passed to `decode` and converted into a metadata 
        # dict and DecodingConfig

        inputs, hyps = [], []

        interactive_mode = input_path is None and ref_path is None and buffer_size == 1
        input_file = corpus.input_file(rank=self.rank, world_size=self.world_size)  # if reading from standard input
        # and world_size > 1, only the first rank reads and broadcasts to the other ranks
        ref_file = corpus.ref_file()

        def read_pairs(input_file, ref_file):
            for input, ref in itertools.zip_longest(input_file, ref_file):
                if input is None or ref is None:
                    logger.warning(
                        'decoding interrupted: input and reference files have a different length; '
                        'evaluation results will be biased'
                    )
                    break
                if len(input) > 0 and len(ref) > 0:  # input can be a numpy array
                    yield input, ref

        # create a virtual input file that iterates over both inpits and references
        if ref_file:
            input_file = read_pairs(input_file, ref_file)
        else:
            input_file = ((input, None) for input in input_file)
        input_file = itertools.islice(input_file, max_lines)

        skip = False  # do we need to decode anything at all?
        if continue_ and output_path and os.path.isfile(output_path):
            inputs = list(input_file)
            with open(output_path, errors='ignore') as out_file:
                hyps = [line.strip() for line in out_file]
            
            if verbose:
                pattern = regex.compile('^[A-Z]-(\d+)\t')
                def key(line):
                    match = pattern.match(line)
                    return match.group(1) if match else line
                # Verbose decoding outputs more than one line per input, for example:
                # H-0	▁c <U> ' est ▁un ▁exemple . </s>
                # D-0	C'est un exemple.
                # P-0	2.757	1.159 0.104 0.686 0.083 0.306 0.092 0.214 0.113
                # TODO: handle this in task
                hyps = [list(g) for _, g in itertools.groupby(hyps, key=key)]

            hyps = hyps[:max_lines]

            if len(hyps) >= len(inputs):
                skip = True
            else:
                if self.cfg.moe_stats:
                    # --continue is not compatible with MoE, which needs to collect gate statistics for the entire corpus
                    # It will only skip this corpus if it was already translated
                    hyps = []
                else:
                    hyps = hyps[:-1]  # remove last line as it may be incomplete
                input_file = iter(inputs[len(hyps):])
                inputs = inputs[:len(hyps)]

        out_files = []
        if not utils.is_master(self.cfg):
            pass
        elif not output_path:
            if not interactive_mode:
                tmp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                out_files.append(tmp_file)
            if output_path is not False:
                out_files.append(sys.stdout)
        else:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            out_files.append(open(output_path, 'w'))
        
        def write(*args):
            for out_file in out_files:
                print(*args, file=out_file)
        def flush():
            for out_file in out_files:
                out_file.flush()

        # FIXME: possible race condition here with --dp-size > 1 and --continue
        for hyp in hyps:
            if verbose:
                for s in hyp:  # in verbose mode, a hypothesis consists of several lines
                    write(s)
            else:
                write(hyp)
        flush()
        
        if verbose:
            # Get the detokenized outputs from past verbose outputs. These will be needed for evaluation.
            # TODO: handle this in task
            hyps = [
                line.split('\t')[-1]
                for hyp in hyps for line in hyp if len(hyp) == 1 or line.startswith('D-')
            ]
        # Convert string hypotheses (which were read from the output file in "continue" mode) into dictionaries
        # that can be used for evaluation. Note that some tasks may need more than the detokenized output to compute
        # scores, in which case "continue" mode will not be possible.
        # TODO: make this more explicit in each task
        hyps = [{'detok': hyp_str} for hyp_str in hyps]

        if not skip:  # do not start the model if the decoding output is already complete
            self.start_model()

        if skip:
            logger.warning(f"'{output_path}' already exists: skipping")
        elif len(hyps):
            logger.warning(f"'{output_path}' has {len(hyps)} already translated lines, skipping those")

        options_str = [
            f'buffer_size={buffer_size}',
            f"input={input_path if input_path else '<stdin>'}",
        ]
        if ref_path:
            options_str.append(f"reference={ref_path}")
        if out_files:
            options_str.append(f"output={out_files[0].name}")

        return_layers = list(return_layers)
        if not skip:
            logger.info(f"starting decoding ({', '.join(options_str)})")
            attn_key = f'dec_{self.model_cfg.decoder_layers - 1}_cross_attn'
            if verbose:
                return_layers.append(attn_key)
            if self.cfg.moe_stats:
                for layer_id in range(self.model_cfg.encoder_layers):
                    return_layers.append(f'enc_{layer_id}_gate')
                for layer_id in range(self.model_cfg.decoder_layers):
                    return_layers.append(f'dec_{layer_id}_gate')

        outputs = []

        while not skip:
            buffer_refs = []
            buffer_inputs = list(itertools.islice(input_file, buffer_size))
            inputs += buffer_inputs  # contains (input, ref) pairs
            if buffer_inputs:
                buffer_inputs, buffer_refs = zip(*buffer_inputs)
            else:
                break

            if ref_file is None:
                cmd_pattern = r'!\s*(?P<name>[\w.-]+)\s*=\s*(?P<value>[\w.-]+)\s*'
                if interactive_mode and (m := regex.fullmatch(cmd_pattern, buffer_inputs[0])):
                    # when in interactive mode (i.e., reading from stdin with buffer_size = 1), the user can type
                    # commands, starting with "!", which will modify the decoding options. For instance:
                    # "!beam_size=1" will set the beam search size to 1 (aka greedy decoding)
                    try:
                        name = m.group('name')
                        value = m.group('value')
                        value = DecodingConfig.parse_str(name, value)
                        decoding_opts[name] = value
                        logger.warning(f"set decoding parameter '{name}' to {value}")
                    except ValueError as e:
                        logger.error(str(e))
                    continue

            buffer_outputs = self.decode(
                *buffer_inputs,
                return_scores=verbose,
                return_layers=return_layers,
                targets=buffer_refs if teacher_forcing else None,
                **decoding_opts,
            )  # list[list[dict]]: (len(lines) * beam_size)
            # each item in the dictionary corresponds to a different step of the translation pipeline:
            # 'src_tokens' (tokenized source), 'tokens' (tokenized output), 'detok' (detokenized output), 'score'
            # (score of each hypothesis), etc.

            outputs += buffer_outputs

            line_ids = list(range(len(hyps), len(hyps) + len(buffer_inputs)))
            hyps += [nbest[0] for nbest in buffer_outputs]

            assert len(buffer_outputs) == len(line_ids)

            if not out_files:
                continue

            for line_id, (hyp, *_) in zip(line_ids, buffer_outputs):
                hyp['idx'] = line_id
                hyp_str = self.task.hypothesis_to_str(hyp, verbose=verbose)
                write(hyp_str)

            flush()

        if len(out_files) > 1:
            # to avoid having to scroll all the way back to find the output filename
            logger.info(f'saved output to {out_files[0].name}')

        if self.cfg.moe_stats:
            gate_stats = models.mixture_of_experts.gather_gate_stats(hyps)  # aggregate gate statistics over all the 
            # top ranking hypotheses
            for k, v in gate_stats.items():
                write(f'MOE\t{k}\t' + ' '.join(f'{x:.6f}' for x in v))

        inputs, refs = zip(*inputs)

        if ref_file is None:
            metrics = []
        for metric in metrics:
            score = self.task.compute_score(
                metric,
                hyps,
                refs,
                bleu_tok=bleu_tok,
                eval_lc=eval_lc,
            )
            self.metrics.update(metric, score)

        for out_file in out_files:
            if out_file is not sys.stdout:
                out_file.close()

        return outputs


@torch.no_grad()
def search(*args, **kwargs) -> list[list[dict]]:
    """ Automatically call `beam_search` and `sampling` depending on the parameters """
    
    if not kwargs.pop('sampling', False):  # to avoid overwriting the function name
        kwargs['sampling_temperature'] = 0
    
    if kwargs.get('sampling_temperature', 1.0) > 0 or kwargs.get('beam_size', 1) <= 1:
        return sampling(*args, **kwargs)
    else:
        return beam_search(*args, **kwargs)


def should_stop(decoder_output: LongTensor, stop_sequences: list[LongTensor]) -> BoolTensor:
    """
    Args:
        decoder_output: tensor of shape (B, T)
        stop_sequences: list of tensors of arbitrary length
    
    Returns: boolean tensor of shape (T,)
    """
    bsz, seq_len = decoder_output.shape
    device = decoder_output.device
    
    has_stop_sequence = []
    for stop_seq in stop_sequences:
        if len(stop_seq) > 0 and seq_len >= len(stop_seq):
            has_stop_sequence.append(
                (decoder_output[:,-len(stop_seq):] == stop_seq).all(dim=1)
            )
    if len(has_stop_sequence) > 0:
        return torch.stack(has_stop_sequence).any(dim=0)
    else:
        return torch.zeros(bsz, dtype=torch.bool, device=device)


@torch.no_grad()
def sampling(
    decoder: Decoder,
    encoder_outputs: Union[None, Tensor, list[Tensor]],
    encoder_masks: Union[None, BoolTensor, list[BoolTensor]],
    max_output_len: int,
    meta: dict,
    *,  # keyword-only arguments
    decoder_input: Optional[LongTensor] = None,
    enc_layer_outputs: dict[str, Tensor] = {},
    **kwargs,
) -> list[list[dict]]:
    
    """
    Shape:
        encoder_outputs: (B, S, D)
        encoder_masks: (B, S)
        decoder_input: (B, S')
        enc_layer_outputs: tensors of shape (B, S, ...)
    """

    if not isinstance(encoder_masks, list):
        encoder_masks = [encoder_masks]

    hyps = None

    for out in sample_on_the_fly(
        decoder,
        encoder_outputs,
        encoder_masks,
        max_output_len,
        meta,
        decoder_input=decoder_input,
        **kwargs,
    ):
        for key, values in out.items():
            batch_size = len(values)
            
            if hyps is None:
                hyps = [{} for _ in range(batch_size)]

            for i, value in enumerate(values):
                hyps[i].setdefault(key, []).append(value)

    src_mask = encoder_masks[0]
    
    for i, hyp in enumerate(hyps):
        for k in hyp:
            hyp[k] = torch.cat(hyp[k])

        hyp.update({k: v[i] for k, v in enc_layer_outputs.items()})

        src_mask_ = None if src_mask is None else src_mask[i]
        uncollate(
            hyp,
            padding_idx=decoder.padding_idx,
            src_mask=src_mask_,
        )
        hyps[i] = [hyp]  # same "n-best" format as beam search
    
    return hyps


@torch.no_grad()
def sample_on_the_fly(
    decoder: Decoder,
    encoder_outputs: Union[None, Tensor, list[Tensor]],
    encoder_masks: Union[None, BoolTensor, list[BoolTensor]],
    max_output_len: int,
    meta: dict,
    *,  # keyword-only arguments
    return_scores: bool = False,
    decoder_input: Optional[LongTensor] = None,
    sampling_temperature: float = 1.0,
    sampling_topk: int = 0,
    sampling_topp: float = 1.0,
    return_layers: list[str] = [],
    sharded: bool = False,
    device: str = None,
    repeat_penalty: float = 1.0,
    blacklist: list[int] = [],
    stop_sequences: list[LongTensor] = [],
    **kwargs,  # unused
) -> Iterator[dict]:
    """
    For more flexibility, contrary to `beam_search` and `sampling`, this function generates its outputs one token
    at a time.

    Shape:
        encoder_outputs: (B, S, D)
        encoder_masks: (B, S)
        decoder_input: (B, S')

    Returns: iterator of dicts {'tokens'}
    """
    assert sampling_topk == 0 or sampling_topp == 1, 'combining nucleus sampling and top-k sampling is not allowed'

    if not isinstance(decoder, EnsembleDecoder):
        decoder = EnsembleDecoder([decoder])
    if torch.is_tensor(encoder_outputs):
        encoder_outputs = [encoder_outputs]
    if torch.is_tensor(encoder_masks):
        encoder_masks = [encoder_masks]

    decoder.eval()
    
    if not encoder_outputs:  # decoder-only model
        assert decoder_input is not None
        batch_size = decoder_input.size(0)
        device = device or decoder_input.device
        encoder_outputs, encoder_masks = [None], [None]
    else:
        assert len(encoder_outputs) == len(decoder) and len(encoder_masks) == len(decoder)
        batch_size = encoder_outputs[0].size(0)
        device = device or encoder_outputs[0].device
    
    encoder_outputs = utils.move_to_device(encoder_outputs, device)
    encoder_masks = utils.move_to_device(encoder_masks, device)
    decoder_input = utils.move_to_device(decoder_input, device)
    # encoder_outputs:  B x S x D
    # encoder_masks:  B x S
    
    if decoder_input is None:
        # by default, the decoder is fed <s> at the first time step, unless a different decoder input is given
        decoder_input = torch.full(
            (batch_size, 1),
            fill_value=decoder.bos_idx,
            dtype=torch.long,
            device=device,
        )
    else:
        # this can be used for teacher forcing or prompting, or for target-side language codes
        assert decoder_input.dim() == 2 and decoder_input.size(0) == batch_size

    prompt_len = utils.mask_to_len(decoder_input != decoder.padding_idx)
    assert (prompt_len > 0).all()  # FIXME: fails if PAD == EOS

    min_prompt_len = prompt_len.min()
    max_prompt_len = prompt_len.max()

    if sharded:
        # GPUs may need to sync with each other during decoding. To avoid deadlocks, we should decode for the same 
        # number of steps (start and max_output_len should be the same)
        dist.all_reduce(min_prompt_len, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_prompt_len, op=dist.ReduceOp.MAX)

    start = min_prompt_len
    max_len = min(decoder.max_len, max_prompt_len + max_output_len)
    assert max_len >= 1
    start = min(start, max_len - 1)  # to ensure that we do at least one decoding step
    
    tokens = torch.full(
        (batch_size, max_len),
        decoder.padding_idx,
        dtype=torch.long,
        device=device,
    )

    tokens[:, :decoder_input.size(1)] = decoder_input
    prompt_mask = (tokens == decoder.padding_idx)

    finished = False
    all_finished = torch.tensor(False, device=device)

    eos_idx = decoder.eos_idx
    pad_idx = decoder.padding_idx
    
    # sequences of ids that should be interpreted as end-of-generation
    stop_sequences = list(stop_sequences)
    stop_sequences.append(LongTensor([eos_idx]))
    stop_sequences = utils.move_to_device(stop_sequences, device)

    has_eos = torch.zeros(batch_size, dtype=torch.bool, device=device)

    prev_step = 0
    incremental_state = {}

    for step in range(start, max_len):
        has_eos = torch.logical_or(has_eos, step >= prompt_len + max_output_len)

        logits, dec_layer_output = decoder(
            encoder_outputs, encoder_masks, tokens[:,prev_step:step],
            state=incremental_state,
            meta=meta,
            return_layers=return_layers,
        )
        logits = utils.move_to_device(logits, device)  # BxTxV
        dec_layer_output = {k: v for k, v in dec_layer_output.items() if torch.is_tensor(v)}

        t = sampling_temperature or 1

        if len(decoder) == 1:
            logits = logits[0]

            if repeat_penalty != 1.0:
                # copied from HuggingFace:
                # https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/generation/logits_process.py#L186
                score = torch.gather(logits[:,-1], 1, tokens[:,:step])
                score = torch.where(score < 0, score * repeat_penalty, score / repeat_penalty)
                logits[:,-1].scatter_(1, tokens[:,:step], score)

            if return_scores or (sampling_temperature != 0 and sampling_topk != 1):
                # softmax is slow and not necessary for greedy decoding (unless we want the scores)
                logits = logits.float().log_softmax(dim=-1)
                # Converting to float32 because float16 zeroes-out many values, which may result in more 
                # deterministic sampling. This can be removed without too much risk if there are OOMs.
        else:
            logits = [(x / t).float().log_softmax(dim=-1) for x in logits]
            logits = torch.logsumexp(torch.stack(logits, dim=0), dim=0) - math.log(len(decoder))
        
        next_logit = logits[:, -1]
        
        for token in blacklist:
            next_logit[:,token] = -torch.inf

        # when a sequence is finished, put all logits to -inf except for EOS
        pad_prob = next_logit[:,pad_idx].clone()
        next_logit.masked_fill_(has_eos.unsqueeze(1), -torch.inf)
        next_logit[:,pad_idx] = pad_prob

        if sampling_temperature == 0 or sampling_topk == 1:   # greedy search
            next_token = next_logit.argmax(dim=-1)
        elif sampling_topk:       # top-k sampling: sample from the k highest probabilities words
            topk_values, topk_indices = next_logit.topk(sampling_topk, dim=-1)
            probs = topk_values.exp()
            next_token = probs.multinomial(num_samples=1, replacement=True)
            next_token = topk_indices.gather(1, next_token)
        elif sampling_topp < 1:   # nucleus sampling: sample from top words whose summed probability equals "topp"
            probs = next_logit.exp()
            probs, sort_indices = probs.sort(dim=-1, descending=True)
            cum_probs = probs.cumsum(dim=-1)
            mask = cum_probs <= sampling_topp
            last = mask.cumsum(-1)[:,-1:]
            last.clamp_(0, mask.size(1) - 1)
            mask.scatter_(-1, last, True)
            probs.masked_fill_(~mask, 0)
            max_dim = last.max()
            mask = mask[:,:max_dim + 1]
            probs = probs[:,:max_dim + 1]
            sort_indices = sort_indices[:,:max_dim + 1]
            next_token = probs.multinomial(num_samples=1, replacement=True)
            next_token = sort_indices.gather(1, next_token)
        else:
            probs = next_logit.exp()
            next_token = probs.multinomial(num_samples=1, replacement=True)
        
        next_token = next_token.reshape(-1)
        # replace next token with prompt token where applicable
        next_token = next_token.where(prompt_mask[:,step], tokens[:,step])

        tokens[:,step] = next_token

        out = {}

        out['tokens'] = tokens[:, prev_step + 1 : step + 1]
        
        if return_scores:
            indices = tokens[:, prev_step + 1 : step + 1].unsqueeze(2)
            scores = -logits.gather(2, indices).squeeze(2)
            out['pos_scores'] = scores
        
        for k, v in dec_layer_output.items():
            out[k] = v

        yield out

        # for each sequence in the batch, check whether it contains one of the stop sequences
        has_eos = has_eos + should_stop(tokens[:,:step + 1], stop_sequences)
        has_eos = torch.logical_and(has_eos, (step >= prompt_len))  # EOS in the prompt shouldn't count
        prev_step = step
        finished = has_eos.all()

        if finished and not sharded:
            break

        if sharded:
            all_finished.fill_(finished)
            dist.all_reduce(all_finished, op=dist.ReduceOp.MIN)
            if all_finished:
                break


@torch.no_grad()
def beam_search(
    decoder: Decoder,
    encoder_outputs: Union[None, Tensor, list[Tensor]],
    encoder_masks: Union[None, BoolTensor, list[BoolTensor]],
    max_output_len: int,
    beam_size: int,
    meta: dict,
    *,  # keyword-only arguments
    return_scores: bool = False,
    decoder_input: Optional[LongTensor] = None,
    return_layers: list[str] = [],
    sharded: bool = False,
    enc_layer_outputs: dict[str, Tensor] = {},
    device: str = None,
    len_penalty: float = 1.0,
    blacklist: list[int] = [],
    **kwargs,  # unused
) -> list[list[dict]]:
    """
    Do beam search decoding with given model from given encoder outputs.
    Same algorithm as in fairseq (with probably some code similarities).

    Shape:
        encoder_outputs: (B, S, D)
        encoder_masks: (B, S)
        decoder_input: (B, S')
        enc_layer_outputs: tensors of shape (B, S, ...)
    
    Returns: a list of B dictionaries or BxK dictionaries (if `return_nbest` is True)

    B: current batch size
    K: beam size
    S: input length
    T: current output length
    V: output vocabulary size
    D: embedding dimension
    L: number of decoder layers

    Memory usage (to multiply with the number representation size: 2 for float16, 4 for float32):
    - model parameters
    - decoder incremental state (self-attention keys and values): K*B*D*L*MAX_T * 2
    - encoder output: K*B*D*MAX_S
    - logits and intermediate scores: K*B*V * 3
    """
    if kwargs.get('repeat_penalty', 1.0) != 1.0:
        utils.warn_once(logger, "'repeat_penalty' is not supported in beam search")

    if not isinstance(decoder, EnsembleDecoder):
        decoder = EnsembleDecoder([decoder])
    if torch.is_tensor(encoder_outputs):
        encoder_outputs = [encoder_outputs]
    if torch.is_tensor(encoder_masks):
        encoder_masks = [encoder_masks]

    decoder.eval()
    enc_layer_outputs = {k: v for k, v in enc_layer_outputs.items() if torch.is_tensor(v)}

    if not encoder_outputs:
        assert decoder_input is not None
        batch_size = decoder_input.size(0)
        device = device or decoder_input.device
        encoder_outputs, encoder_masks = [None], [None]
        decoder_only = True
    else:
        assert len(encoder_outputs) == len(decoder) and len(encoder_masks) == len(decoder)
        batch_size = encoder_outputs[0].size(0)
        device = device or encoder_outputs[0].device
        decoder_only = False
    
    encoder_outputs = utils.move_to_device(encoder_outputs, device)
    encoder_masks = utils.move_to_device(encoder_masks, device)
    enc_layer_outputs = utils.move_to_device(enc_layer_outputs, device)
    decoder_input = utils.move_to_device(decoder_input, device)
    # encoder_outputs:  B x S x D
    # encoder_masks:  B x S
    
    if decoder_input is None:
        # by default, the decoder is fed BOS at the first time step, unless a different decoder input is given
        decoder_input = torch.full(
            (batch_size, 1),
            fill_value=decoder.bos_idx,
            dtype=torch.long,
            device=device,
        )
    else:
        # this can be used for teacher forcing or prompting, or for target-side language codes
        assert decoder_input.dim() == 2 and decoder_input.size(0) == batch_size
    
    prompt_len = utils.mask_to_len(decoder_input != decoder.padding_idx)
    assert (prompt_len > 0).all()

    min_prompt_len = prompt_len.min()
    max_prompt_len = prompt_len.max()

    decoder_input = decoder_input.repeat_interleave(beam_size, dim=0)  # shape: B*K x T'
    
    if sharded:
        # GPUs may need to sync with each other during decoding. To avoid deadlocks, we should decode for the same 
        # number of steps (start and max_output_len should be the same)
        dist.all_reduce(min_prompt_len, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_prompt_len, op=dist.ReduceOp.MAX)

    start = min_prompt_len
    max_len = min(decoder.max_len, max_prompt_len + max_output_len)
    assert max_len >= 1

    # B*K x MAX_LEN
    tokens = torch.full(
        (batch_size * beam_size, max_len),
        fill_value=decoder.eos_idx,  # using EOS as padding, to avoid having to manually append EOS at the end
        # of each hypothesis 
        dtype=torch.long,
        device=device,
    )
    
    tokens[:, :decoder_input.size(1)] = decoder_input
    decoder_input = decoder_input[:, :start]

    finished_hyps = [[] for _ in range(batch_size)]

    src_mask = None
    if not decoder_only:
        encoder_outputs = [encoder_out.repeat_interleave(beam_size, dim=0) for encoder_out in encoder_outputs]
        src_mask = encoder_masks[0]  # for post-processing
        encoder_masks = [encoder_mask.repeat_interleave(beam_size, dim=0) for encoder_mask in encoder_masks]

    hyp_scores = None   # B*K
    beam_range = torch.arange(beam_size * 2).to(tokens)
    beam_offset = beam_size * torch.arange(0, batch_size).unsqueeze(1).to(tokens)
    batch_id_mapping = torch.arange(batch_size).to(tokens)  # maps new batch ids to initial batch ids (because 
    # the size of the batch will be reduced when enough finished hypotheses are found for a given input)
    
    if return_scores:
        pos_scores = torch.zeros(
            batch_size * beam_size,
            max_len,
            device=device,
            dtype=torch.float,
        )
    else:
        pos_scores = None
    
    dec_layer_outputs = {}
    incremental_state = {}
    finished = False
    all_finished = torch.tensor(False, device=device)
    prev_step = 0

    for step in range(start, max_len):

        logits, dec_layer_output = decoder(
            encoder_outputs, encoder_masks, decoder_input,
            state=incremental_state,
            meta=meta,
            return_layers=return_layers,
        )

        if sharded:
            all_finished.fill_(finished)
            dist.all_reduce(all_finished, op=dist.ReduceOp.MIN)
            if all_finished:
                break
            if finished:
                continue

        logits = utils.move_to_device(logits, device)
        dec_layer_output = {k: v for k, v in dec_layer_output.items() if torch.is_tensor(v)}
        logits = [value.float().log_softmax(dim=-1) for value in logits]  # Converting to float32 because float16
        # zeroes-out many values, but this can be removed without too much risk if there are OOMs.
        logits = torch.logsumexp(torch.stack(logits, dim=0), dim=0) - math.log(len(decoder))  # B*K x T x V

        scores = logits[:,-1]  # B*K x V
        vocab_size = scores.size(-1)

        if logits.size(1) > 1:
            assert step == start
            prompt_tokens = decoder_input[:,1:]
            prompt_scores = logits[:,:-1]
            prompt_scores = prompt_scores.gather(2, prompt_tokens.unsqueeze(2)).squeeze(2)
            prompt_scores *= (prompt_tokens != decoder.padding_idx)
            scores += prompt_scores.sum(dim=1).unsqueeze(1)
        else:
            prompt_scores = None

        for token in blacklist:
            scores[:,token] = -torch.inf

        if step < max_prompt_len:
            # put -inf everywhere except at prompt tokens, when applicable
            forced = tokens[:,step]
            idx = torch.arange(scores.size(0))
            scores_forced = scores[idx, forced].clone()
            mask = (forced != decoder.padding_idx) & (forced != decoder.eos_idx)
            scores.masked_fill_(mask.unsqueeze(1), -torch.inf)
            scores[idx, forced] = scores_forced

        if hyp_scores is None:
            # at the first step, all K candidates have the same scores, only keep the first one
            scores = scores[::beam_size]
        else:
            scores += hyp_scores.unsqueeze(-1)    # aggregate scores

        scores = scores.reshape(batch_size, -1)   # B x K*V
        topk = scores.topk(beam_size * 2, dim=1)   # (B x K*2, B x K*2)
        # select top K*2, as up to K of these may end with EOS

        # indices in [0, K) of the beams that are being continued
        beam_indices_ = topk.indices.div(vocab_size, rounding_mode='floor')  # B x K*2
        # indices in [0, B*K) of the beams that are being continued (for selection within B*K flattened dimension)
        beam_indices = beam_indices_ + beam_offset     # B x K*2
        vocab_indices = topk.indices.fmod(vocab_size)  # B x K*2 (token indices)
        hyp_scores = topk.values                       # B x K*2

        eos_mask = vocab_indices.eq(decoder.eos_idx)   # B x K*2
        eos_mask &= ~hyp_scores.isinf()   # if EOS has -inf, it should be ignored (this can happen when fewer than
        # K*2 tokens have a non-inf score)
        eos_mask = torch.logical_and(eos_mask, (step >= prompt_len[batch_id_mapping]).unsqueeze(1))  # do not count 
        # EOS that are in the prompt as EOS

        # list of L indices in [0, B*K) identifying partial hypotheses (in `tokens`) that are in the top-K when
        # continued with EOS
        eos_beam_indices = torch.masked_select(beam_indices[:, :beam_size], mask=eos_mask[:, :beam_size])
        
        batch_ids = None
        if eos_beam_indices.numel() > 0:
            finished_ids = []
            # scores of these finished hypotheses
            finished_scores = torch.masked_select(
                topk.values[:, :beam_size], mask=eos_mask[:, :beam_size]
            )  # shape: L
            
            # select the hypotheses from `tokens` that correspond to these indices
            finished_sents = tokens.index_select(0, eos_beam_indices)[:,1:step + 1]  # L * T
            if return_scores:
                finished_pos_scores = pos_scores.index_select(0, eos_beam_indices)[:,:step]
                finished_pos_scores[:,step - 1] = finished_scores
                # pos_scores is a cumulative sum, retrieve the real per-position scores
                finished_pos_scores[:,1:] = finished_pos_scores[:,1:] - finished_pos_scores[:,:-1]
            finished_outputs = {}
            for k, v in dec_layer_outputs.items():
                v = v.index_select(0, eos_beam_indices.to(v.device))[:,:step]
                last_v = dec_layer_output[k].index_select(0, eos_beam_indices.to(v.device))
                if k.endswith('self_attn'):
                    v[:,-1:,...,:last_v.size(-1)] = last_v
                else:
                    v[:,-1:] = last_v
                finished_outputs[k] = v
            for i, beam_id in enumerate(eos_beam_indices):
                # batch index of this hypothesis: [0, B*K) -> [0, B)
                batch_id = torch.div(beam_id, beam_size, rounding_mode='floor')
                init_batch_id = batch_id_mapping[batch_id]
                hyps = finished_hyps[init_batch_id]  # list of finished hypotheses for this batch index
                if len(hyps) >= beam_size:
                    continue
                hyp = {
                    'tokens': finished_sents[i],
                    'score': -finished_scores[i],
                    **{k: v[i] for k, v in finished_outputs.items()},
                }
                if return_scores:
                    hyp['pos_scores'] = -finished_pos_scores[i]
                hyps.append(hyp)
                if len(hyps) == beam_size:
                    # this batch index has K finished hypotheses: it can be
                    # removed from the batch
                    finished_ids.append(batch_id)
            
            if all(len(hyps) >= beam_size for hyps in finished_hyps):
                if sharded:
                    # need to continue decoding dummy outputs until all replicas have finished decoding
                    # this is required for mixtures of experts and FSDP because replicas communicate with each other,
                    # and one ending earlier would cause a deadlock
                    finished = True
                    if not decoder_only:
                        encoder_outputs = [encoder_out[:1,:1] for encoder_out in encoder_outputs]
                        encoder_masks = [encoder_mask[:1,:1] for encoder_mask in encoder_masks]
                    decoder_input = decoder_input[:1]
                    return_layers = []
                    incremental_state = None
                    continue
                else:
                    break

            if finished_ids:
                # batch indices that have found K candidates and can be removed from the batch
                finished_ids = torch.stack(finished_ids)
                # reduce batch size
                new_batch_size = batch_size - len(finished_ids)

                mask = torch.ones(batch_size).to(eos_mask)
                mask[finished_ids] = False
                # indices in the current batch to keep for the next steps
                batch_ids = torch.arange(batch_size).to(tokens).masked_select(mask)
                # maps new batch indices to original indices
                batch_id_mapping = batch_id_mapping.masked_select(mask)
                
                beam_offset = beam_size * torch.arange(0, new_batch_size).unsqueeze(1).to(tokens)
                beam_indices_ = beam_indices_[batch_ids]
                beam_indices = beam_indices_ + beam_offset
                vocab_indices = vocab_indices[batch_ids]
                hyp_scores = hyp_scores[batch_ids]
                eos_mask = eos_mask[batch_ids]

                def reshape(tensor):
                    shape = tensor.shape[1:]
                    return tensor.view(batch_size, -1)[batch_ids].view(new_batch_size * beam_size, *shape)

                tokens = reshape(tokens)
                if return_scores:
                    pos_scores = reshape(pos_scores)
                
                for dict in dec_layer_outputs, dec_layer_output:
                    for k, v in dict.items():
                        dict[k] = reshape(v)
                
                batch_size = new_batch_size

        # we now want to go from B x K*2, to B x K tensors,
        # keeping only candidates that do not end with EOS
        _, indices = torch.topk(
            beam_range + eos_mask * beam_size * 2,
            k=beam_size,
            dim=1,
            largest=False
        )
        beam_indices = torch.gather(beam_indices, dim=1, index=indices)
        vocab_indices = torch.gather(vocab_indices, dim=1, index=indices)
        hyp_scores = torch.gather(hyp_scores, dim=1, index=indices)

        # flatten beam & batch dimensions into a single dimension (B x K -> B*K)
        hyp_scores = hyp_scores.view(-1)
        beam_indices = beam_indices.view(-1)
        vocab_indices = vocab_indices.view(-1)

        # reorder partial hypotheses & corresponding data to correspond to the new beams
        tokens[:, :step] = torch.index_select(tokens[:, :step], dim=0, index=beam_indices)
        tokens[:, step] = vocab_indices
        if return_scores:
            if prompt_scores is None:
                pos_scores[:, :step - 1] = torch.index_select(pos_scores[:, :step - 1], dim=0, index=beam_indices)
            else:
                pos_scores[:, :step - 1] = torch.index_select(prompt_scores.cumsum(dim=1), dim=0, index=beam_indices)
            pos_scores[:,step - 1] = hyp_scores

        for k, v in dec_layer_output.items():
            # decoder layer outputs should be tensors of shape (B, T, ...)
            v = v.index_select(0, beam_indices.to(v.device))
            bsz, time_steps, *other_dims = v.shape
            last_dim = other_dims[-1] if other_dims else None
            if k in dec_layer_outputs:
                padded_v = dec_layer_outputs[k]
                # reorder the layer outputs gathered so far with the new beam indices
                padded_v[:, :step - 1] = torch.index_select(
                    padded_v[:, :step - 1],
                    dim=0,
                    index=beam_indices.to(v.device),
                )
            else:
                if k.endswith('self_attn'):
                    other_dims = other_dims[:-1] + [max_len]
                padded_v = torch.zeros(bsz, max_len, *other_dims).to(v)
                dec_layer_outputs[k] = padded_v
            
            # concatenate the layer outputs at the last time step
            # self-attention is a special case because its last dimension depends on the output length
            if k.endswith('self_attn'):
                padded_v[:, prev_step : step, ..., :last_dim] = v
            else:
                padded_v[:, prev_step : step] = v
        
        prev_step = step
        decoder_input = vocab_indices.unsqueeze(1)

        # take removed sentences into account (those tensors haven't yet been updated with the new batch size)
        if batch_ids is not None:
            beam_indices.view(-1, beam_size).add_(
                beam_size * (
                    batch_ids - torch.arange(batch_ids.size(0)).to(batch_ids)
                ).unsqueeze(1)
            )
        if not decoder_only:
            encoder_outputs = [encoder_output.index_select(0, beam_indices) for encoder_output in encoder_outputs]
            encoder_masks = [encoder_mask.index_select(0, beam_indices) for encoder_mask in encoder_masks]
        
        Decoder.reorder_state(incremental_state, beam_indices)

    incremental_state = None

    output = []
    if return_scores:
        pos_scores[:,1:] = pos_scores[:,1:] - pos_scores[:,:-1]
        pos_scores = -pos_scores
    
    # convert negative scores (higher is better) to positive 'loss' (lower is better)
    hyp_scores = -hyp_scores

    for batch_id, nbest in enumerate(finished_hyps):
        if not nbest:  # reached end before generating EOS
            i = (batch_id_mapping == batch_id).nonzero()[0,0]
            hyp = {
                'tokens': tokens.view(batch_size, beam_size, -1)[i, 0, 1:],
                'score': hyp_scores.view(batch_size, beam_size)[i, 0],
            }
            if return_scores:
                hyp['pos_scores'] = pos_scores.view(batch_size, beam_size, -1)[i, 0, :-1]
            for k, v in dec_layer_outputs.items():
                hyp[k] = v.view(batch_size, beam_size, *v.shape[1:])[i, 0, :-1]
            nbest = [hyp]
        
        enc_layer_output = {k: v[batch_id] for k, v in enc_layer_outputs.items()}
        nbest = [
            {
                **hyp,
                **enc_layer_output,
                'normalized_score': hyp['score'] / (len(hyp['tokens']) + 1)**len_penalty
            }
            for hyp in nbest
        ]

        get_score = lambda hyp: hyp['normalized_score']
        nbest.sort(key=get_score)
        output.append(nbest)

    for i, nbest in enumerate(output):
        src_mask_ = None if src_mask is None else src_mask[i]
        for hyp in nbest:
            uncollate(
                hyp,
                padding_idx=decoder.padding_idx,
                src_mask=src_mask_,
            )
    return output


def uncollate(
    hypothesis: list[dict],
    padding_idx: int,
    src_mask: Optional[BoolTensor] = None,
    max_len: Optional[int] = None,
) -> None:
    """
    Post-processes (in-place modification) a list of decoding hypotheses by truncating layer outputs to remove values
    corresponding to padding tokens
    """
    hypothesis.setdefault('tokens', LongTensor())
    # Mask that is True at padding positions
    hyp_mask = torch.BoolTensor([idx == padding_idx for idx in hypothesis['tokens']])
    if max_len is not None:
        hyp_mask[max_len:] = True

    max_len = (~hyp_mask).sum()
    hypothesis['tokens'] = hypothesis['tokens'][:max_len]

    # Filter layer outputs to remove those of padding tokens
    for k, v in hypothesis.items():
        if v is None:
            continue
        
        if k.startswith('enc_'):
            v = v[~src_mask]
            if k.endswith('_self_attn'):
                v = v[...,~src_mask]
        elif k.startswith('dec_') or k == 'pos_scores':
            v = v[~hyp_mask]
            if k.endswith('_self_attn'):
                # Since the decoder output length cannot be known in advance, decoder self-attention has
                # a last dimension of 'max_output_len'. Pad the mask accordingly:
                pad_length = v.size(-1) - hyp_mask.size(-1)
                mask = nn.functional.pad(hyp_mask, (0, pad_length), value=True)
                v = v[...,~mask]
            elif k.endswith('_cross_attn'):
                v = v[...,~src_mask]

        hypothesis[k] = v
    
    if 'pos_scores' in hypothesis and 'score' not in hypothesis:
        hypothesis['score'] = hypothesis['pos_scores'].sum()


class EnsembleModel(EncoderDecoder):
    """ Groups several Transformer models in an ensemble for decoding """
    def __init__(self, models: list[Transformer]):
        super().__init__()
        assert models
        self.models = models
        self.encoder = EnsembleEncoder([model.encoder for model in models])
        self.decoder = EnsembleDecoder([model.decoder for model in models])


class EnsembleEncoder(Encoder):
    """ Ensemble of encoders: will return lists of tensors which are compatible with EnsembleDecoder. """
    def __init__(self, encoders: list[Encoder]):
        super().__init__()
        self.encoders: list[Encoder] = nn.ModuleList(encoders)
    
    def __len__(self):
        return len(self.encoders)

    @property
    def max_len(self) -> int:
        return self.encoders[0].max_len

    def forward(self, *args, **kwargs) -> tuple[list[Tensor], list[BoolTensor], dict[str, Tensor]]:
        encoder_outputs, encoder_masks, enc_layer_outputs = zip(
            *[encoder(*args, **kwargs) for encoder in self.encoders]
        )
        if all(encoder_out is None for encoder_out in encoder_outputs):  # decoder-only
            encoder_outputs = encoder_masks = None
        else:
            encoder_outputs = list(encoder_outputs)  # zip returns tuples
            encoder_masks = list(encoder_masks)
        enc_layer_output = enc_layer_outputs[0]  # only return layer outputs of the first encoder in the ensemble
        return encoder_outputs, encoder_masks, enc_layer_output


class EnsembleDecoder(Decoder):
    """
    Ensemble of decoders: takes as input the output of EnsembleEncoder and outputs lists of logit tensors
    which are then averaged in sampling(...) or beam_search(...)
    """
    def __init__(self, decoders: list[Decoder]):
        super().__init__()
        self.decoders: list[Decoder] = nn.ModuleList(decoders)
    
    @property
    def padding_idx(self) -> int:
        return self.decoders[0].padding_idx

    @property
    def eos_idx(self) -> int:
        return self.decoders[0].eos_idx

    def __len__(self):
        return len(self.decoders)

    @property
    def max_len(self) -> int:
        return self.decoders[0].max_len

    def forward(
        self,
        encoder_outputs: list[Tensor],
        encoder_masks: list[BoolTensor],
        *args,
        **kwargs,
    ) -> tuple[list[Tensor], dict[str, Tensor]]:
        outputs = []
        for encoder_out, encoder_mask, decoder in itertools.zip_longest(encoder_outputs, encoder_masks, self.decoders):
            output = decoder(encoder_out, encoder_mask, *args, **kwargs)
            outputs.append(output)
        decoder_outputs, dec_layer_outputs = zip(*outputs)
        dec_layer_output = dec_layer_outputs[0]  # only return layer outputs of the first decoder in the ensemble
        return list(decoder_outputs), dec_layer_output
