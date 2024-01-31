# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import os
import sys
import logging
import logging.handlers
import time
import numpy as np
import torch
import math
import collections
import threading
import queue
import itertools
import traceback
import signal
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import Iterator, NoReturn, Optional
from pasero import utils
from pasero.utils import defined
from pasero.tasks import Task, Corpus
from pasero.config import register_dataset, DistributedConfig, TrainingDatasetConfig
from pasero.config import DynamicTrainingDatasetConfig, SimpleDynamicTrainingDatasetConfig, DebugTrainingDatasetConfig


logger = logging.getLogger('data')


def dummy_batch(batch: dict, size: int = 1) -> dict:
    if batch is None:
        return None
    dummy = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            dummy[k] = v[:size]
        else:
            dummy[k] = v
    return dummy


def shard_batch(batch: dict, shard_id: int = 0, shard_count: int = 1) -> dict:
    assert shard_id < shard_count
    if shard_count == 1 or batch is None:
        return batch
    sharded = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            bsz = v.size(0)
            assert bsz % shard_count == 0
            bsz = bsz // shard_count
            sharded[k] = v[bsz * shard_id:bsz * (shard_id + 1)]
        else:
            sharded[k] = v
    return sharded


class LineIndex:
    """
    Stores the starting positions of line tuples in corpora.
    
    No all positions are stored, as this would take too much memory. Lines are grouped into consecutive blocks
    of 256 lines, whose starting position is saved. To sample a line tuple, we actually sample a block and sequentially 
    read the entire block.
    
    Because creating this index requires reading the entire training corpora (which is slow), we save it on the disk
    for future uses.
    """
    def __init__(
        self,
        corpora: list[Corpus],
        block_size: int = 256,
        index_path: Optional[str] = None,
        save_index: bool = True,
        reset: bool = False,
    ):
        self.block_size = block_size
        self.index = {}
        self.changed = False
        self.index_path = index_path

        if index_path is not None and not reset:
            self.load()

        for corpus in corpora:
            self.index_corpus(corpus)

        if index_path is not None and save_index:
            self.save()

        # save memory by keeping only paths which are relevant to this training instance
        paths = set()
        for corpus in corpora:
            paths_ = tuple(sorted(corpus.realpaths))
            paths.add(paths_)
        self.index = {k: v for k, v in self.index.items() if k in paths}

    def __getitem__(self, corpus: Corpus):
        # Get the block positions for this corpus
        paths = tuple(corpus.realpaths)
        indices = np.argsort(paths)
        sorted_paths = tuple(np.array(paths)[indices])  # to avoid duplicates, the index always uses sorted 
        # paths as keys: (path1, path2) and (path2, path1) have the same index

        if sorted_paths in self.index:
            blocks, block_size, size, mtime = self.index[sorted_paths]
            indices = np.argsort(indices)
            blocks = blocks[:,indices]  # permute block to match the order in `paths`
            return blocks, block_size, size, mtime
        else:
            raise KeyError
    
    def index_corpus(self, corpus: Corpus):
        # Compute block positions for this corpus, if it hasn't been indexed yet
        paths = tuple(corpus.realpaths)
        corpus_mtime = corpus.getmtime()

        try:
            *_, index_mtime = self[corpus]
            if corpus_mtime <= index_mtime:
                return
            logger.info(f'index for {corpus} is outdated')
        except KeyError:
            pass

        logger.info(f'indexing {corpus}')

        files = corpus.open_files()
        # sort paths and files alphabetically
        indices = np.argsort(paths)
        paths = np.array(paths)[indices].tolist()
        files = np.array(files)[indices].tolist()

        blocks = []
        size = this_block_size = 0
        
        positions = []
        lengths = []
        for file in files:
            positions_, lengths_ = file.get_positions()
            positions.append(positions_)
            lengths.append(lengths_)

        cur_pos = None

        assert len(set(map(len, positions))) == 1, f"error: source/target length mismatch in corpus '{corpus}'"

        for pos_tuple, len_tuple in zip(zip(*positions), zip(*lengths)):
            if not all(len_tuple):  # skip entries where at least one line is empty
                continue
            
            if this_block_size == 0:
                cur_pos = pos_tuple

            size += 1
            this_block_size += 1
            if this_block_size == self.block_size:
                blocks.append(cur_pos)
                this_block_size = 0
        
        if this_block_size != 0:
            blocks.append(cur_pos)
        
        blocks = np.array(blocks)

        self.index[tuple(paths)] = blocks, self.block_size, size, corpus_mtime
        self.changed = True
        
    def load(self):
        # Load a saved index from the disk
        if not os.path.exists(self.index_path):
            logger.info(f'no cached line positions at {self.index_path}')
            return
        
        logger.info(f'found cached line positions at {self.index_path}')
        index = {}
        with utils.suppress():
            index = torch.load(self.index_path)

        self.index = {}
        self.changed = False
        for paths, value in index.items():   # clean old index versions
            try:
                blocks, block_size, size, mtime = value
                assert (
                    list(paths) == sorted(paths) and
                    all(os.path.exists(path) for path in paths) and
                    isinstance(blocks, np.ndarray) and isinstance(block_size, int) and block_size > 0 and
                    isinstance(size, int) and size > 0 and
                    isinstance(mtime, float) and mtime > 0
                )
                paths = tuple(map(os.path.realpath, paths))
                self.index[paths] = value
            except:   # drop this entry
                self.changed = True

    def save(self):
        # Save the current index to the disk
        if self.changed:
            logger.info(f'saving line positions at {self.index_path}')
            
            with utils.suppress():
                index_dir = os.path.dirname(self.index_path)
                if index_dir and not os.path.isdir(index_dir):  # create the directory where the index will be saved,
                    # if it does not exist ('tmp/' by default)
                    os.makedirs(index_dir, exist_ok=True)
                torch.save(self.index, self.index_path)
                self.changed = False
    
    @classmethod
    def build(
        cls,
        cfg: TrainingDatasetConfig,
        dist_cfg: DistributedConfig,
        corpora: list[Task],
    ) -> 'LineIndex':
        # In multi-process settings, the master process creates the index and sends it to the other processes via NCCL.
        if utils.is_master(dist_cfg):  # also true in single-process settings
            # will try saving the index for future training with the same data
            index = LineIndex(corpora, index_path=cfg.line_index_path, reset=cfg.reset_line_index,
                              save_index=cfg.cache_line_index, block_size=cfg.block_size or 256)
        else:
            index = None
            
        if utils.is_distributed(dist_cfg):
            index = [index]
            # master sends the index to the other processes
            # the other processes wait for the master to load/create and send the index
            dist.broadcast_object_list(index, 0)
            index = index[0]
        
        return index


class CorpusSampler:
    """
    Provides an iterator over corpus ids, which samples corpora based on their size and the lang_temperature
    parameter.
    The generated corpus ids are indices from 0 to len(corpora) - 1
    """
    def __init__(self, corpora: list[Corpus], sizes: list[int], lang_temperature: float = 1.0):
        self.corpora = corpora
        self.corpus_ids = np.arange(len(self.corpora))

        assert all(corpus.exists() for corpus in self.corpora), 'error: some training files do not exist or are empty'

        class_name = self.__class__.__name__
        
        # real corpus size in number of lines
        self.sizes = np.array(sizes, dtype=np.int64)

        multipliers = [defined(corpus.multiplier, 1) for corpus in self.corpora]
        # adjust each corpus "size" depending on its multiplier
        self.sizes = (self.sizes * np.array(multipliers)).astype(np.int64)
        self.total_lines = self.sizes.sum()
        
        logger.info(f'{class_name} | total lines {self.total_lines}')

        line_per_lang = collections.defaultdict(int)   # only used for logging
        sizes = []
        for size, corpus in zip(self.sizes, self.corpora):
            line_per_lang[tuple(corpus.langs)] += size
            if corpus.probability is None:
                sizes.append(size)
            else:
                # give zero size to this corpus, as it shouldn't impact temperature-based probabilities
                sizes.append(0)
        sizes = np.array(sizes).astype(np.int64)

        # compute probability to sample lines from each corpus
        if lang_temperature and lang_temperature != 1:
            # Apply temperature-based sampling at the language pair level
            stats_per_lang = {}
            for corpus_id, (size, corpus) in enumerate(zip(sizes, self.corpora)):
                lang_tuple = tuple(corpus.langs)
                total_size, corpus_ids = stats_per_lang.get(lang_tuple, (0, []))
                stats_per_lang[lang_tuple] = (total_size + size, corpus_ids + [corpus_id])
            
            if lang_temperature >= 100:
                denom = sum(int(size > 0) for size, _ in stats_per_lang.values())
            else:
                denom = sum(size ** (1 / lang_temperature) for size, _ in stats_per_lang.values())

            probs = [0.0] * len(self.corpora)
            
            for size, corpus_ids in stats_per_lang.values():
                if lang_temperature >= 100:
                    prob = int(size > 0) / denom
                else:
                    prob = size ** (1 / lang_temperature) / denom
                
                for corpus_id in corpus_ids:
                    probs[corpus_id] = prob * sizes[corpus_id] / max(1, size)
            
            self.probs = np.array(probs, dtype=np.float32)
        else:
            self.probs = sizes / max(1, sizes.sum())

        fixed_probs = [defined(corpus.probability, -1) for corpus in self.corpora]
        if any(p != -1 for p in fixed_probs):
            fixed_probs = np.array(fixed_probs)
            indices = (fixed_probs != -1)
            remaining_prob = 1 - fixed_probs[indices].sum()
            self.probs[indices] = fixed_probs[indices]
            indices = np.logical_not(indices)
            self.probs[indices] *= remaining_prob / max(1, self.probs[indices].sum())

        assert (self.probs >= 0).all()
        self.probs /= self.probs.sum()   # make sure probabilities sum to 1

        if len(line_per_lang) > 1:
            for lang_tuple, size in line_per_lang.items():
                prob = sum([
                    p for p, corpus in zip(self.probs, self.corpora)
                    if tuple(corpus.langs) == lang_tuple], 0)
                lang_tuple_str = '-'.join(lang_tuple)  # e.g., de-en
                logger.info(
                    f'{class_name} | {lang_tuple_str} | prob {prob:.5f} | lines {size} '
                    f'({size / max(1, self.total_lines):.3%})')

        for corpus, prob, size in zip(self.corpora, self.probs, self.sizes):
            logger.info(
                f'{class_name} | {corpus.corpus_id} | prob {prob:.5f} | lines {size} '
                f'({size / max(1, self.total_lines):.3%})')

    def __iter__(self) -> Iterator[int]:
        while True:  # pick a corpus at random
            yield from np.random.choice(self.corpus_ids, p=self.probs, size=1000)  # sampling multiple values at once 
            # is much faster than multiple calls to np.random.choice()


class LineSampler(CorpusSampler):
    """
    Creates a line tuple generator over several corpora. Picks a corpus at random (with some probability
    that depends on each corpus size), then reads the next line from this corpus. When reaching the end of a corpus,
    start from the beginning.

    Args:
        corpora (List[Corpus]): list of corpora

    How to use:
        ```
        sampler = LineSampler(corpora)
        for sample in sampler:   # infinite iterator
            # do stuff
        ```
    """
    def __init__(
        self,
        corpora: list[Corpus],
        line_index: LineIndex,
        lang_temperature: float = 1.0,
        store_files_under: Optional[int] = None,
        shuffle: bool = True,
        shard_id: int = 0,
        shard_count: int = 1,
        max_lines: Optional[int] = None,
        close_files: bool = False,
    ):
        class_name = self.__class__.__name__
        self.readers = []
        
        for corpus in corpora:
            logger.info(f"{class_name} | opening corpus ({', '.join(corpus.paths)})")

            reader = LineReader(
                corpus,
                line_index=line_index,
                store_files_under=store_files_under,
                shuffle=shuffle,
                shard_id=shard_id,
                shard_count=shard_count,
                max_lines=max_lines,
                close_files=close_files,
            )
            self.readers.append(reader)

        sizes = [reader.size for reader in self.readers]
        super().__init__(corpora, sizes, lang_temperature=lang_temperature)

    def __iter__(self) -> Iterator[dict]:
        for corpus_id in super().__iter__():
            sample = next(self.readers[corpus_id])
            yield sample


class LineReader:
    """
    Iterates over a corpus in a semi-random way: line tuples are grouped into consecutive blocks of 256
    and those blocks are shuffled. This allows for randomness while keeping the speed advantages of sequential
    reads. This iterator also skips line tuples where either side is empty.
    """
    def __init__(
        self,
        corpus: Corpus,
        shuffle: bool = True,
        line_index: Optional[LineIndex] = None,
        store_files_under: Optional[int] = None,
        shard_id: int = 0,
        shard_count: int = 1,
        max_lines: Optional[int] = None,
        endless: bool = True,
        close_files: bool = False,
    ):
        self.shuffle = shuffle
        self.max_lines = max_lines
        self.endless = endless
        self.close_files = close_files
        self.max_doc_size = corpus.max_doc_size
        self.corpus = corpus

        self.paths = tuple(corpus.realpaths)
        self.files = corpus.open_files(store_files_under=store_files_under)

        if self.close_files:
            # Close files too avoid 'too many open files' errors
            # Those will be automatically re-opened when needed
            for file in self.files:
                file.close()

        if line_index is None:
            line_index = LineIndex([corpus])
        self.blocks, self.block_size, self.size, _ = line_index[corpus]
        
        self.block_indices = np.arange(len(self.blocks))
        self.block_sizes = np.full(len(self.blocks), self.block_size)
        self.block_sizes[-1] = self.size - self.block_sizes[:-1].sum()
        assert self.block_sizes[-1] > 0

        if self.max_lines:
            num_blocks = math.ceil(self.max_lines / self.block_size)
            self.block_indices = self.block_indices[:num_blocks]
            self.blocks = self.blocks[:num_blocks]
            self.block_sizes = self.block_sizes[:num_blocks]
            self.size = min(self.block_sizes.sum(), self.max_lines)
            self.block_sizes[-1] = self.size - self.block_sizes[:-1].sum()
            assert self.block_sizes[-1] > 0

        if shard_count > 1:
            num_blocks = len(self.block_indices) // shard_count
            
            if shuffle:   # assumes all workers have the same seed
                np.random.shuffle(self.block_indices)
            if shard_id < shard_count - 1:
                self.block_indices = self.block_indices[shard_id * num_blocks:(shard_id + 1) * num_blocks]
            else:
                self.block_indices = self.block_indices[shard_id * num_blocks:]

        assert len(self.block_indices) > 0, ("some file shards are empty, this can happen with very small files that "
            "are sharded across many workers: try reducing --dataloader-workers or the number of GPUs used for "
            "training")
        # This can happen for instance when training on 4 GPUs with a training corpus of 2000 lines and
        # --dataset-type simple --dataloader-workers 4. Because the block size is 256, we can have at most 8 blocks,
        # while this setting requires at least 16 blocks
        self._iter = iter(self)

    def __next__(self) -> dict:
        try:
            return next(self._iter)
        except StopIteration as e:
            if self.endless:
                self._iter = iter(self)
                return next(self._iter)
            else:
                raise e

    def __iter__(self) -> Iterator[dict]:
        block_indices = np.random.permutation(self.block_indices) if self.shuffle else self.block_indices

        for block_id in block_indices:
            pos_tuple = self.blocks[block_id]
            for pos, file in zip(pos_tuple, self.files):
                file.seek(pos)

            block = []
            while len(block) < self.block_sizes[block_id]:
                line_tuple = tuple(next(file) for file in self.files)

                if all(len(line) > 0 for line in line_tuple):  # using `len` instead of truth value because line 
                    # can be a numpy array and `bool(array)` doesn't work
                    block.append(line_tuple)
            
            if self.close_files:
                for file in self.files:
                    file.close()

            if self.max_doc_size > 1:
                block = iter(block)
                while True:
                    # read N consecutive sentences from this block and merge them into a single document
                    doc_size = np.random.randint(1, self.max_doc_size + 1)
                    line_tuples = list(itertools.islice(block, doc_size))
                    if not line_tuples:
                        break
                    line_tuple = self.merge_consecutive(line_tuples)
                    yield self.corpus.tuple_to_dict(line_tuple)
            else:
                # read entire block at once to maximize sequential reads
                for line_tuple in block:
                    yield self.corpus.tuple_to_dict(line_tuple)

    def merge_consecutive(self, line_tuples: list[tuple]) -> tuple:
        # Usually tuples of strings are returned (e.g., (source, target)), but in the case of document-level translation
        # we return pairs of tuples ((src1, src2, ...), (tgt1, tgt2, ...)) each of these tuple representing a document
        # of several consecutive sentences. It is then up to `DocumentLevelTranslationTask.preprocess` to deal with
        # these pairs of documents.
        if len(line_tuples) == 1:
            return line_tuples[0]  # single sentence tuple
        else:  # document
            return tuple(lines for lines in zip(*line_tuples))  # ((src1, tgt1), (src2, tgt2)) ->
            # ((src1, src2), (tgt1, tgt2))


class ValidationDataset:
    """
    Static dataset for a single corpus, typically a validation set. The entire dataset is loaded and pre-processed at 
    once and kept in memory.

    Contrary to TrainingDataset, this dataset holds a single corpus and __iter__ does a single epoch in a deterministic 
    order. For training (even with a single corpus), TrainingDataset should be used.
    """
    def __init__(
        self,
        dist_cfg: DistributedConfig,
        task: Task,
        corpus: Corpus,
        line_index: Optional[LineIndex] = None,
        truncate: bool = True,
        verbose: bool = True,
        max_lines: Optional[int] = None,
    ):
        assert corpus.max_doc_size == 1, "doc-level validation is not implemented yet"
        self.corpus = corpus
        self.seed = dist_cfg.seed or 42
        self.shard_count = dist_cfg.dp_size or 1
        self.shard_id = dist_cfg.dp_rank or 0
        self.max_lines = max_lines
        self.task = task
        self.collate_fn = task.get_collate_fn(getattr(torch, dist_cfg.dtype))
        self.truncate = truncate
        self.task.eval()  # preprocessing and batching can be different between training and validation data
        self.data, self.references = self.read(line_index, verbose=verbose)
        self.length = len(self.data)

        # build batches (this can be done once and for all, because we don't need to shuffle)
        batches = self.task.build_batches(
            self.data,
            shuffle=False,
        )
        shard_len = math.ceil(len(batches) / self.shard_count)
        batches = batches[self.shard_id::self.shard_count]
        batches += [[]] * (shard_len - len(batches))  # pad with empty batches to ensure all GPUs get the
        # same number of batches (those empty batches will be converted to None by the collater)
        self.batches = batches

        oov_tokens = total_tokens = 0
        for sample_bin in self.data:
            oov, total = self.task.count_oov(sample_bin)
            oov_tokens += oov
            total_tokens += total
        oov_percentage = oov / max(1, total)

        logger.info(
            f'{self.corpus} | {len(self.data)} lines | {len(self.batches)} batches | '
            f'oov tokens {oov_percentage:.2%}'
        )
    
    def read(
        self,
        line_index: Optional[LineIndex] = None,
        verbose: bool = False,
    ) -> Optional[tuple[list[dict], list[Optional[str]]]]:
        data, references = [], []
        reader = LineReader(
            self.corpus,
            shuffle=False,
            line_index=line_index,
            max_lines=self.max_lines,
        )

        for sample in reader:
            reference = self.task.get_reference(sample)
            if reference is not None:
                references.append(reference)
            
            sample_bin = self.task.preprocess(
                sample,
                truncate=self.truncate,
                append_eos=True,
            )
            
            if not sample_bin:  # too long or too short
                continue

            # log tokenized (source, target) every N lines
            line_id = len(data)
            if verbose and line_id % 1000 == 0:
                self.task.log_sample(sample_bin)

            data.append(sample_bin)
        
        return data, references

    def __iter__(self) -> Iterator[dict]:
        for batch in self.batches:
            yield self.collate_fn(batch)


class TrainingDataset(torch.utils.data.IterableDataset):
    """
    Training dataset that can involve multiple corpora and which produces batches indefinitely (with `endless_iterator`)

    This class should not be instantiated directly, but rather its subclasses DynamicTrainingDataset or
    SimpleDynamicTrainingDataset.
    """
    def __init__(
        self,
        cfg: TrainingDatasetConfig,
        dist_cfg: DistributedConfig,
        task: Task,
        corpora: list[Corpus],
        verbose: bool = False,
    ):
        self.cfg = cfg
        self.task = task
        self.collate_fn = task.get_collate_fn(getattr(torch, dist_cfg.dtype))
        self.corpora = corpora
        self.shuffle = cfg.shuffle
        self.seed = dist_cfg.seed or 42
        self.verbose = verbose
        self.buffer_size = cfg.buffer_size
        self.max_lines = cfg.max_lines
        self.lang_temperature = cfg.lang_temperature
        self.batch_by = set(cfg.batch_by or [])
        self.line_index = LineIndex.build(cfg, dist_cfg, corpora)
        self.logging_queue = mp.Queue()
        self.skipped_percentage = mp.Value('d')
        self.oov_percentage = mp.Value('d')
        self.num_workers = cfg.num_workers
        self.dataloader_workers = cfg.dataloader_workers
        self.dataloader_pin_memory = cfg.dataloader_pin_memory
        self.dataloader_prefetch_factor = cfg.dataloader_prefetch_factor
        self.shard_count = dist_cfg.dp_size or 1
        self.shard_id = dist_cfg.dp_rank or 0
        self.truncate = cfg.truncate
        self.store_files_under = cfg.store_files_under
        self.close_files = cfg.close_files
        self.batch_size = task.cfg.batch_size
        if dist_cfg.sequence_parallel and dist_cfg.tp_size > 1:  # simulates data parallelism by increasing the batch 
            # size (because sequence parallelism needs to shard the batch into `tp_size` batches). This is not done
            # in `ValidationDataset` because `sequence_parallelism` is disabled at validation.
            self.batch_size *= dist_cfg.tp_size

    def init_logger(self):
        """
        Should be called in spawned processes to automatically log messages to self.logging_queue
        These messages can then be processed by the main logger when calling self.log_queue()
        """
        handler = logging.handlers.QueueHandler(self.logging_queue)
        logger = logging.getLogger()
        logger.handlers = [handler]
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    def log_queue(self):
        """
        Gets messages from self.logging_queue and logs them to the main process' logger. Spawned processes can write to
        this queue by calling `init_logger(); logger.info(msg)`
        
        This is typically called periodically in a Thread
        """
        while True:
            try:
                record = self.logging_queue.get_nowait()
                if record is not None:
                    record.msg = f'{self.__class__.__name__} | {record.msg}'
                    logging.getLogger(record.name).handle(record)
            except queue.Empty:
                break

    def __iter__(self) -> Iterator[list[dict]]:
        """
        Called by DataLoader in self.endless_iterator()
        
        This method should be overriden in child classes (e.g., TrainingStaticDataset and TrainingDynamicDataset)
        to yield batches in a numpy format compatible with `collate_fn`
        """
        raise NotImplementedError

    def endless_iterator(self) -> Iterator[dict]:
        opts = dict(
            collate_fn=self.collate_fn,
            num_workers=self.dataloader_workers,
            batch_size=None,
            pin_memory=self.dataloader_pin_memory,
            prefetch_factor=self.dataloader_prefetch_factor,
            persistent_workers=True
        )
        if self.dataloader_workers == 0:
            opts.pop('persistent_workers')
            opts.pop('prefetch_factor')
        data_loader = torch.utils.data.DataLoader(self, **opts)
        yield from data_loader

    def buffered_batching(self, buffer: list[dict]) -> list[list[dict]]:
        # build homogeneous batches w.r.t. source lang, target lang and/or domain
        groups = collections.defaultdict(list)
        for sample in buffer:
            key = tuple(sample['meta'].get(k) for k in self.batch_by)
            if sample.get('encoder_input') is not None:
                # for multimodal inputs (we don't want audio and text in the same batch)
                ndim = sample['encoder_input'].ndim
                dtype = sample['encoder_input'].dtype
                key = key + (ndim, dtype)
            groups[key].append(sample)
        groups = groups.values()

        batches = []
        for group in groups:
            batches += self.task.build_batches(
                group,
                shuffle=self.shuffle,
                batch_size=self.batch_size,  # override `task_cfg.batch_size`
            )
        if self.shuffle:
            np.random.shuffle(batches)
        return batches


@register_dataset('dynamic')
class DynamicTrainingDataset(TrainingDataset):
    """
    Multilingual training dataset whose data is loaded and pre-processed on the fly.

    Several processes are spawned to handle different parts of the pre-processing. They communicate via queues:
    
    reader -> workers -> batcher -> DataLoaders

    - The reader samples line tuples from the given corpora and writes them to `reader_queues`
    - The workers read line tuples from these queues, tokenize and binarize them and write to `worker_queues`
    - The batcher reads from these queues and stores samples into a large buffer (100k line tuples by default) before
    sorting them by length and building efficient batches, which are written to `batch_queues`
    
    There can be multiple workers (--num-workers) because the tokenization can be slow and CPU-intensive. However,
    having a single batcher (shared across all GPUs) helps build more efficient batches (i.e., whose samples have a
    similar length).

    Note that the (reader -> workers -> batcher) part is done on the main process of each node only (unless
    --per-gpu-batching is set). The other GPUs on a given node retrieve their batches from `batch_queue`.
    However, each GPU has its own DataLoader, which can also spawn processes depending on the --dataloader-workers 
    value.
    """
    cfg: DynamicTrainingDatasetConfig

    def __init__(
        self,
        cfg: DynamicTrainingDatasetConfig,
        dist_cfg: DistributedConfig,
        task: Task,
        corpora: list[Corpus],
        verbose: bool = False,
        queues: Optional[mp.Queue] = None,
        **kwargs,  # unused
    ):
        super().__init__(cfg, dist_cfg, task, corpora, verbose=verbose)
        line_index = self.line_index
        del self.line_index   # this can take a large amount of memory. Avoid copying it in every worker, only
        # reader() needs it
        
        per_gpu_batching = cfg.per_gpu_batching or dist_cfg.dp_size == 1 or dist_cfg.dp_local_size == 1 or dist_cfg.tp_size > 1

        if not per_gpu_batching:
            # one batcher shared across several GPUs, incompatible with tensor parallelism
            self.shard_id = dist_cfg.dp_rank // dist_cfg.dp_local_size
            self.shard_count = dist_cfg.dp_size // dist_cfg.dp_local_size

        if dist_cfg.tp_rank > 0:  # data is read by rank 0 and sent to the other ranks
            pass
        elif dist_cfg.dp_local_rank == 0 or per_gpu_batching:
            assert self.num_workers > 0

            self.reader_queues = [mp.Queue(maxsize=8192) for _ in range(self.num_workers)]
            self.worker_queues = [mp.Queue(maxsize=8192) for _ in range(self.num_workers)]
            if per_gpu_batching:
                self.batch_queue = mp.Queue(maxsize=1024)
                self.batch_queues = [self.batch_queue]
            else:
                self.batch_queue = queues[dist_cfg.dp_local_rank]
                self.batch_queues = queues

            workers = []
            reader = mp.Process(target=self.reader, args=(line_index,))
            workers.append(reader)
            for worker_id in range(self.num_workers):
                worker = mp.Process(target=self.worker, args=(worker_id,))
                workers.append(worker)
            batcher = mp.Process(target=self.batcher)
            workers.append(batcher)
            logger = threading.Thread(target=self.logger, args=(workers,))
            
            for worker in workers + [logger]:
                worker.daemon = True
                worker.start()
        else:
            assert queues
            self.batch_queue = queues[dist_cfg.dp_local_rank % len(queues)]

    def logger(self, workers: list[mp.Process]) -> NoReturn:
        # Periodically logs statistics about the processes and queues
        while True:
            self.log_queue()
            alive = '/'.join(str(int(worker.is_alive())) for worker in workers)
            msg = f'{self.__class__.__name__} | workers alive {alive}'
            try:
                lines = '/'.join(str(queue.qsize()) for queue in self.reader_queues)
                samples = '/'.join(str(queue.qsize()) for queue in self.worker_queues)
                batches = '/'.join(str(queue.qsize()) for queue in self.batch_queues)
                mem_used = utils.get_cpu_mem_used()
                mem_left = utils.get_cpu_mem_left()
                logger.info(
                    f'{msg} | lines {lines} | samples {samples} | batches {batches} | '
                    f'skipped {self.skipped_percentage.value:.2%} | oov tokens {self.oov_percentage.value:.2%} | '
                    f'cpu_mem_used {mem_used:.2f} | cpu_mem_left {mem_left:.2f}'
                )
            except NotImplementedError:   # macOS does not support queue.qsize
                logger.info(msg)
            if not any(worker.is_alive() for worker in workers):
                break
            time.sleep(180)

    @staticmethod
    def _set_signals(queues: list[mp.Queue]) -> None:
        """
        Upon receiving SIGTERM or SIGINT, put None into given queues before exiting.
        The other processes in the pipeline will automatically stop when reading None from their input queues.
        """
        def fail(sig, *_):
            for q in queues:
                try:
                    q.get_nowait()      # in case q is full
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(None)  # this will be propagated to the workers
                except queue.Full:
                    pass
            sys.exit(128 + sig)
        signal.signal(signal.SIGINT, fail)
        signal.signal(signal.SIGTERM, fail)
        signal.signal(signal.SIGUSR1, fail)

    @staticmethod
    def _terminate() -> None:
        """ Send SIGTERM to current process, hence triggering the fail() handler defined in _set_signals() """
        os.kill(os.getpid(), signal.SIGTERM)

    def reader(self, line_index: LineIndex) -> NoReturn:
        self._set_signals(self.reader_queues)
        self.init_logger()
        # it is important that all workers have the same seed for data sharding in LineReader just below
        np.random.seed(self.seed)
        logger.info(f'started reader')
        try:
            sampler = LineSampler(
                self.corpora,
                line_index=line_index,
                lang_temperature=self.lang_temperature,
                shuffle=self.shuffle,
                shard_id=self.shard_id,
                shard_count=self.shard_count,
                store_files_under=self.store_files_under,
                max_lines=self.max_lines,
                close_files=self.close_files,
            )
            if sampler.total_lines < self.buffer_size:
                logger.warning('warning: buffer_size is higher than corpus size')
            # For different readers to sample different corpora:
            np.random.seed(self.seed + self.shard_id)
            worker_id = 0
            for sample in sampler:
                self.reader_queues[worker_id].put(sample)
                worker_id = (worker_id + 1) % len(self.reader_queues)
        except SystemExit:
            pass
        except:
            traceback.print_exc()
            self._terminate()

    def worker(self, worker_id: int) -> NoReturn:
        self._set_signals(self.worker_queues)
        self.init_logger()
        np.random.seed(self.seed)
        logger.info(f'started worker {worker_id}')
        self.task.train()

        skipped_lines = total_lines = 0
        oov_tokens = total_tokens = 0

        try:
            while True:
                sample = self.reader_queues[worker_id].get()
                
                if sample is None:
                    self._terminate()
                
                sample_bin = self.task.preprocess(
                    sample,
                    truncate=self.truncate,
                    append_eos=True,
                )

                total_lines += 1
                if not sample_bin:
                    skipped_lines += 1
                    if worker_id == 0:
                        self.skipped_percentage.value = skipped_lines / total_lines
                    continue
                else:
                    oov, total = self.task.count_oov(sample_bin)
                    oov_tokens += oov
                    total_tokens += total
                
                self.worker_queues[worker_id].put(sample_bin)

                line_count = total_lines - skipped_lines
                if worker_id == 0 and line_count % 100000 == 0:
                    self.task.log_sample(sample_bin)
                    self.oov_percentage.value = oov_tokens / max(1, total_tokens)

        except SystemExit:
            pass
        except:
            traceback.print_exc()
            self._terminate()

    def batcher(self) -> NoReturn:
        self._set_signals(self.batch_queues)
        self.init_logger()
        np.random.seed(self.seed)
        logger.info(f'started batcher')
        self.task.train()
        try:
            worker_id = 0
            gpu_id = 0
            buffer = []
            while True:
                sample = self.worker_queues[worker_id].get()
                if sample is None:
                    self._terminate()
                worker_id = (worker_id + 1) % len(self.worker_queues)
                buffer.append(sample)
                if len(buffer) == self.buffer_size:
                    for batch in self.buffered_batching(buffer):
                        self.batch_queues[gpu_id].put(batch)
                        gpu_id = (gpu_id + 1) % len(self.batch_queues)
                    buffer = []
        except SystemExit:
            pass
        except:
            traceback.print_exc()
            self._terminate()

    def __iter__(self) -> Iterator[list[dict]]:
        # Used by DataLoader
        while True:
            batch = self.batch_queue.get()
            yield batch
            if batch is None:
                break


@register_dataset('simple')
class SimpleDynamicTrainingDataset(TrainingDataset):
    """
    Similar to DynamicTrainingDataset, but does not use any queue or processes other than those created by 
    Pytorch's DataLoaders.
    
    The reading, pre-processing and batching are all done in the same processes. At least one such process is spawned 
    per GPU. More can be spawned with --dataloader-workers (default: 4 per GPU)
    
    Side effects:
    - less packed batches (instead of one batcher, there are DATALOADER_WORKERS * NUM_GPUS ones)
    - (possibly) less randomness: data is sharded per DataLoader worker. Shuffling is always done on the same shard of
    data

    This dataset is especially useful for binary formats, where the bottleneck is not the pre-processing but
    the reading speed and memory usage.

    Note that if memory is not an issue but disk I/Os are a bottleneck, `--cache-data` can be used to store all 
    examples and avoid having to read and preprocess them again at the next epoch.
    """
    cfg: SimpleDynamicTrainingDatasetConfig

    def __init__(
        self,
        cfg: SimpleDynamicTrainingDatasetConfig,
        dist_cfg: DistributedConfig,
        task: Task,
        corpora: list[Corpus],
        verbose: bool = False,
        **kwargs,  # unused
    ):
        super().__init__(cfg, dist_cfg, task, corpora, verbose=verbose)
        # sort corpora from smallest to largest (useful for caching)
        self.corpora.sort(key=lambda corpus: corpus.getsize()) 
        self.shard_count = self.shard_count * max(self.dataloader_workers, 1)
        self.shard_id = self.shard_id * max(self.dataloader_workers, 1)
        self.cache_data = cfg.cache_data
        self.max_cache_size = cfg.max_cache_size
        if dist_cfg.tp_rank == 0:
            logger = threading.Thread(target=self.logger)
            logger.daemon = True
            logger.start()

    def logger(self) -> NoReturn:
        while True:
            self.log_queue()
            mem_used = utils.get_cpu_mem_used()
            mem_left = utils.get_cpu_mem_left()
            logger.info(
                f'{self.__class__.__name__} | skipped lines {self.skipped_percentage.value:.2%} | '
                f'oov tokens {self.oov_percentage.value:.2%} | cpu_mem_used {mem_used:.2f} | cpu_mem_left {mem_left:.2f}'
            )
            time.sleep(180)

    def __iter__(self) -> Iterator[list[dict]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 0 if worker_info is None else worker_info.num_workers
        shard_id = self.shard_id + worker_id
        shard_count = self.shard_count
        
        if num_workers == 0:  # the dataloader is running in the main process, which means it can (and should) use the
            # main logger: calling init_logger() here would disable logging...
            pass
        elif worker_id == 0:
            self.init_logger()
        else:
            # only the first worker on each GPU is allowed to log
            logger.disabled = True

        # FIXME: if dataloader_workers == 0, this might conflict with the evaluation code
        self.task.train()

        # it is important that all workers have the same seed for data sharding in LineReader just below
        np.random.seed(self.seed)

        readers = []
        sizes = []
        cache_size = 0
        cached_corpora = set()
        for corpus_id, corpus in enumerate(self.corpora):
            logger.info(f"opening corpus ({', '.join(corpus.paths)})")
            # this is only an approximation of the amount of memory this corpus will take
            cache_size += corpus.getsize() / shard_count
            if self.cache_data and cache_size / 2**30 <= self.max_cache_size:
                cached_corpora.add(corpus_id)
                logger.info(f'corpus {corpus} will be cached, current cache size: {cache_size / 2**30:.2f}GiB')
            reader = LineReader(
                corpus,
                line_index=self.line_index,
                store_files_under=self.store_files_under,
                shuffle=self.shuffle,
                shard_id=shard_id,
                shard_count=shard_count,
                max_lines=self.max_lines,
                endless=corpus_id not in cached_corpora,
                close_files=self.close_files,
            )
            readers.append(reader)
        sizes = [reader.size for reader in readers]
        sampler = CorpusSampler(self.corpora, sizes, self.lang_temperature)

        assert sampler.total_lines >= self.buffer_size, 'buffer size is higher than corpus size'
        
        buffer: list[dict] = []

        cache = collections.defaultdict(list)
        def iterator(list: list[dict]) -> Iterator[dict]:  # transforms a list into an infinite iterator
            while True:
                np.random.shuffle(list)
                yield from list
        
        np.random.seed(self.seed + shard_id)   # for different corpora to be sampled across different processes

        skipped_lines = total_lines = 0
        oov_tokens = total_tokens = 0

        for corpus_id in sampler:  # corpus_id is not the attribute of corpus, but an integer in [0, len(corpora))
            corpus = self.corpora[corpus_id]
            
            if readers[corpus_id] is None:  # line reader is exhausted, now reading from cache
                sample = next(cache[corpus_id])
            else:
                try:
                    sample = next(readers[corpus_id])

                    sample_bin = self.task.preprocess(
                        sample,
                        truncate=self.truncate,
                        append_eos=True,
                    )

                    total_lines += 1
                    if not sample_bin:  # too short or too long
                        skipped_lines += 1
                        if worker_id == 0:
                            self.skipped_percentage.value = skipped_lines / total_lines
                        continue
                    else:
                        oov, total = self.task.count_oov(sample_bin)
                        oov_tokens += oov
                        total_tokens += total

                    line_count = total_lines - skipped_lines
                    if worker_id == 0 and line_count % 100000 == 0:
                        self.task.log_sample(sample_bin)
                        self.oov_percentage.value = oov_tokens / max(1, total_tokens)

                    if self.cache_data and corpus_id in cached_corpora:
                        cache[corpus_id].append(sample_bin)
                        # TODO: check that there is enough memory left

                except StopIteration as e:  # finished reading this file
                    if self.cache_data and corpus_id in cached_corpora:
                        # close the file and start using the cached samples
                        readers[corpus_id] = None
                        # transform the cached list of samples into an infinite iterator:
                        cache[corpus_id] = iterator(cache[corpus_id])
                        sample_bin = next(cache[corpus_id])
                    else:
                        # This shouldn't happen normally because we set endless=True to LineReader when
                        # --cache-data is not set
                        raise e

            buffer.append(sample_bin)
            if len(buffer) == self.buffer_size:
                yield from self.buffered_batching(buffer)
                buffer = []


@register_dataset('debug')
class DebugTrainingDataset(SimpleDynamicTrainingDataset):
    cfg: DebugTrainingDatasetConfig
