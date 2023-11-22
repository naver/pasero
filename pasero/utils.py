# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import os
import sys
import torch
import numpy as np
import time
import random
import logging
import contextlib
import socket
import traceback
import subprocess
import regex
import shutil
import psutil
import torch.nn as nn
from collections import defaultdict, namedtuple
from numbers import Number
from typing import Any, Callable, Optional, TypeVar, Type, Iterator
from torch import Tensor, BoolTensor, LongTensor
from torch.nn.utils.rnn import pad_sequence
from torch import distributed as dist
from pasero.config import DistributedConfig, TrackerConfig


T = TypeVar('T')  # generic type
SpecialTokens = namedtuple('SpecialTokens', ['padding_idx', 'eos_idx', 'bos_idx', 'unk_idx', 'sep_idx'])


logger = logging.getLogger('utils')


def defined(*args) -> Any:
    """
    Return first non-None value from args, or None if args is empty or all its elements are None.
    
    This doesn't work if 0 or False are considered as valid values:
    `current_value = current_value or default_value`

    Instead, the following will only overwrite "current_value" if it is None:
    `current_value = defined(current_value, default_value)`
    
    which is more compact than the alternative:
    `current_value = current_value if current_value is not None else default_value`

    """
    return next((arg for arg in args if arg is not None), None)


def is_master(cfg: DistributedConfig) -> bool:
    return not cfg.distributed_rank


def is_distributed(cfg: DistributedConfig) -> bool:
    return cfg.distributed_world_size and cfg.distributed_world_size > 1


def read_stdin(rank=0, world_size=1, interactive=False) -> Iterator[str]:
    if interactive:
        import readline   # when imported, modifies the behavior of input() to interpret special character sequences
        # (e.g., arrow keys)
    while True:
        line = [None]
        if rank > 0:  # only master reads from stdin and then broadcasts the line
            # to the other processes
            dist.broadcast_object_list(line)
        else:
            try:
                line = [input().strip()]
            except EOFError:
                pass
            if world_size > 1:
                dist.broadcast_object_list(line)
        line = line[0]
        if line is None:
            break
        elif not line:
            continue
        yield line


def gather_list(cfg: DistributedConfig, list: list) -> list:
    if is_distributed(cfg):
        lists = [[] for _ in range(cfg.distributed_world_size)]
        dist.all_gather_object(lists, list)
        list = sum(lists, [])
    return list


def gather_dict(cfg: DistributedConfig, dict_: dict) -> dict:
    if is_distributed(cfg):
        dicts = [{} for _ in range(cfg.distributed_world_size)]
        dist.all_gather_object(dicts, dict_)
        dict_ = dicts[0]
        for d in dicts[1:]:
            for k, v in d.items():
                if k in dict_:
                    dict_[k] += v
                else:
                    dict_[k] = v
    return dict_


def barrier(cfg: DistributedConfig):
    if is_distributed(cfg):
        dist.barrier()


def broadcast(cfg: DistributedConfig, value: Tensor, dtype=torch.float):
    if is_distributed(cfg):  # make sure all processes have the same seed
        value = torch.tensor(value, device='cuda', dtype=dtype)  # only CUDA tensors can be sent with NCCL
        dist.broadcast(value, 0)
        return value.item()
    else:
        return value


def get_tp_group(cfg: DistributedConfig):
    if cfg.tp_size > 1:
        groups = []
        for dp_rank in range(cfg.dp_size):
            start_rank = dp_rank * cfg.tp_size
            ranks = list(range(start_rank, start_rank + cfg.tp_size))
            group = dist.new_group(ranks=ranks)
            groups.append(group)
        return groups[cfg.dp_rank]
    else:
        return None


def get_dp_group(cfg: DistributedConfig):
    if cfg.tp_size > 1:
        groups = []
        for tp_rank in range(cfg.tp_size):
            ranks = [rank for rank in range(cfg.distributed_world_size) if rank % cfg.tp_size == tp_rank]
            group = dist.new_group(ranks=ranks)
            groups.append(group)
        return groups[cfg.tp_rank]
    else:
        return None


def distributed_batch_iterator(
    batches: Iterator[dict],
    group: Optional[dist.ProcessGroup] = None,
    rank: int = 0,
) -> Iterator[dict]:
    """
    Iterator over batches that runs in the main rank only and broadcasts the batches to the other ranks
    """
    while True:
        if rank == 0:
            try:
                batch = next(batches)
            except StopIteration:
                batch = False
        else:
            batch = None
        
        if group is not None:
            batch = [batch]
            master_rank = dist.get_global_rank(group, 0)
            dist.broadcast_object_list(batch, master_rank, group)
            # TODO: speed this up by scattering instead of broadcasting
            batch = batch[0]
        
        if batch is False:
            break
        yield batch


class LoggingFormatter(logging.Formatter):
    """ Logging formatter that uses colors for errors and warnings """
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    GREEN = "\x1b[32;20m"
    END = "\x1b[0m"

    def format(self, record):
        fmt = '%(asctime)s | %(name)s | %(message)s'
        if record.levelno >= logging.ERROR:
            fmt = self.RED + fmt + self.END
        elif record.levelno >= logging.WARNING:
            fmt = self.YELLOW + fmt + self.END
        elif record.levelno == logging.DEBUG:
            fmt = self.GREEN + fmt + self.END
        return logging.Formatter(fmt).format(record)


def init_logging(log_file: Optional[str] = None, level=logging.INFO, stream=sys.stdout, append: bool = False):
    logging.basicConfig(
        format='%(asctime)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level='INFO',
        stream=stream,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    formatter = root_logger.handlers[0].formatter
    root_logger.handlers[0].formatter = LoggingFormatter()

    if stream is None:
        root_logger.handlers = []  # only log to file

    if log_file:
        dirname = os.path.dirname(log_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        handler = logging.FileHandler(log_file, mode='a' if append else 'w')
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def merge_models(dicts: list[dict]) -> dict:
    return {k: v for dict in reversed(dicts) for k, v in dict.items()}


def average_models(dicts: list[dict]) -> dict:
    if len(dicts) == 1:
        return dicts[0]
    else:
        return {
            k: sum(dict[k] for dict in dicts) / len(dicts)
            for k in (dicts[0] if dicts else [])
        }


def embed_init(weight: Tensor, padding_idx: Optional[int] = None):
    nn.init.normal_(weight, mean=0, std=weight.size(-1) ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(weight[padding_idx], 0)


def mask_to_len(mask: BoolTensor, dim: int = -1) -> LongTensor:
    """
    Args:
        lengths: tensor of shape (B, T)
    Returns:
        tensor of shape (B,) where T is the length of each sequence in the batch excluding trailing zeros in the mask

    For example: [[0,1,0,1,0],[1,1,0,0,0]] -> [4,2]
    """
    return (mask.flip(dims=(dim,)).cumsum(dim) > 0).sum(dim)


def len_to_mask(lengths: LongTensor, size: Optional[int] = None) -> BoolTensor:
    """
    Args:
        lengths: tensor of shape (B,) with the length of each prompt
    Returns:
        tensor of shape (B, T) where T is the maximum value in `lengths`, with False at every prompt position

    For example: [0,3,1] -> [[1,1,1],[0,0,0],[0,1,1]]
    """
    size = size or lengths.max()
    return torch.arange(size, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)


def prompt_len(tokens: LongTensor, sep: int = 0) -> LongTensor:
    """
    Args:
        tokens: tensor of shape (B, T)
    Returns:
        tensor of shape (B,) containing the length of each sequence in the batch up to given separator
    
    Note: the lengths includes the separator and is 0 if there is no separator

    ```
    tokens = tensor([
        [1, 2, 3, 0, 5],  # prompt length: 4
        [0, 2, 3, 4, 5],  # 1
        [1, 2, 3, 4, 0],  # 5
        [1, 2, 3, 4, 5],  # 0
        [1, 0, 3, 0, 5],  # 4
    ])
    prompt_len(tokens, 0)  # tensor([4, 1, 5, 0, 4])
    ```
    """
    seq_len = tokens.size(1)
    reversed = tokens.flip(dims=(1,))  # reverse to include the separator in the prompt & to return 0 when
    # there is no separator
    return seq_len - ((reversed == sep).cumsum(dim=1) == 0).sum(dim=1)


def mask_by_len(tokens: LongTensor, lengths: LongTensor, fill_value: int, truncate: bool = False) -> LongTensor:
    """
    Replace every value after some length with given value
    
    Args:
        tokens: tensor of shape (B, T)
        lengths: tensor of shape (B,)
    Returns:
        tensor of shape (B, T') where T' <= T (T = T' if `truncate` is False)
    
    Example:
        ([[1,2,3,4,5]], [[3]]) -> [1,2,3,0,0]
    """
    if truncate:
        new_size = lengths.max()
        tokens = tokens[:,:new_size]
    mask = len_to_mask(lengths, tokens.size(1))
    return tokens.masked_fill(mask, fill_value)


def pad(tokens: Tensor, size: int, fill_value: int, dim: bool) -> Tensor:
    """ Pad a tensor to the right, with given value up to given size at given dimension """
    shape = list(tokens.size())
    if size <= shape[dim]:
        return tokens
    shape[dim] = size - shape[dim]
    pad = torch.full(shape, fill_value=fill_value, dtype=tokens.dtype, device=tokens.device)
    return torch.cat([tokens, pad], dim=dim)


@contextlib.contextmanager
def suppress(
    *exceptions: Type[Exception],
    silent: bool = False,
    caller: Optional[str] = None,
    max_attempts: Optional[int] = None,
):
    """
    Function decorator and context manager to ignore exceptions.

    Skips the function if the maximum number of failed attempts for this caller name has been reached.
    """
    attempts = getattr(suppress, 'attempts', defaultdict(int)) 
    if caller and max_attempts and attempts[caller] >= max_attempts:
        return

    exceptions = exceptions or (Exception,)
    try:
        yield
        if caller and max_attempts:
            attempts[caller] = 0
    except exceptions:
        if not silent:
            traceback.print_exc()
        if caller and max_attempts:
            attempts[caller] += 1
    
    setattr(suppress, 'attempts', attempts)


@contextlib.contextmanager
def disable_logging():
    """
    Temporarily disable logging:
    
    ```
    with disable_logging():
        do_some_stuff()  # there won't be any logging here
    
    do_some_other_stuff()  # logging back to normal
    ```
    """
    logging.disable()
    yield
    logging.disable(logging.NOTSET)


def log_once(logger: logging.Logger, msg: str, level: int = logging.INFO, id: Optional[str] = None) -> None:
    """
    Log given message only once. If it was already logged by this logger, nothing happens.
    Different messages may share the same id, thanks to the optional argument `id`.
    """
    if not hasattr(logger, '_already_logged'):  # save previously logged message as an attribute of the logger
        logger._already_logged = set()
    
    id = id or msg
    if id not in logger._already_logged:
        logger.log(level, msg)
        logger._already_logged.add(id)

def warn_once(logger: logging.Logger, msg: str, id: Optional[str] = None) -> None:
    """
    Like `log_once` but with the "warning" level.
    """
    log_once(logger, msg, level=logging.WARNING, id=id)


def retry(
    *exceptions: Type[Exception],
    max_attempts: int = 3,
    delay: int = 60,
    silent: bool = False,
    fail: bool = True,
):
    """
    Function decorator to retry calling the function several times when an error occurs
    """
    exceptions = exceptions or (Exception,)
    def retry_(func):
        def print_error():
            if not silent:
                traceback.print_exc()
        def inner(*args, **kwargs):
            attempts = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        if fail:
                            raise e
                        else:
                            print_error()
                            break
                    print_error()
                    time.sleep(delay)
        return inner
    return retry_


def find_file(*paths, dirs=None, fail=True):
    """
    Look for a file within a list of candidate filenames and directories.
    """
    for path in paths:
        if path is None:
            continue
        
        if not dirs or os.path.isabs(path):
            if os.path.isfile(path):
                return path
            continue
        
        for dirname in dirs:
            if dirname is None:
                continue
            path_ = os.path.join(dirname, path)
            if os.path.isfile(path_):
                return path_
    if fail:
        raise FileNotFoundError(f"file '{paths[0]}' does not exist")
    else:
        return None


class Metrics:
    """
    Implements a basic way to store and aggregate metrics and measure time.

    All metrics are stored as simple dictionaries. It is the responsibility of the caller to know, for a given metric 
    name, what type of metric it is and to call the appropriate methods.
    
    For instance, this can be used to record evaluation scores:
    ```
    def evaluate(model, test_set, metrics):
        bleu_score = ...
        metrics.update('bleu', bleu_score)
    
    metrics = Metrics(history_size=-1)
    evaluate(model, test_set_1, metrics)
    print(metrics.val('bleu'))  # print the BLEU score over test_set_1
    evaluate(model, test_set_2, metrics)
    print(metrics.avg('bleu'))  # print the average BLEU score over test_set_1 and test_set_2
    ```

    Or to record the training time and number of tokens and compute words per second:
    ```
    log_interval = 10
    metrics = Metrics(history_size=log_interval)

    for i, batch in enumerate(batches, 1):
        with metrics.timer('wall'):
            loss = train(model, batch)
            num_tokens = size(batch)
        
        metrics.update('num_tokens', num_tokens)
        metrics.increment('steps')
    
        if i % log_interval == 0:
            wps = metrics.rolling_divide('num_tokens, wall')
            # num_tokens and wall time are aggregated over 10 steps and WPS is computed by dividing their rolling sums
            print(f'{wps=}')

    steps = metrics.sum('steps')   # gives the total number of steps (regardless of `history_size`)
    print(f'{steps=}')
    ```
    """
    def __init__(self, history_size: int = 10):
        """
        Args:
            - history_size: number of measurements to aggregate over when calling `rolling_sum`, `rolling_divide`, or
            `avg`. Set to -1 for infinite history.
        """
        self.history_size = history_size
        self.reset()
    
    def reset(self) -> None:
        self.values = defaultdict(list)  # latest values for each metric (up to `history_size` measurements are kept)
        self.sums = defaultdict(int)  # overall sum for each metric (regardless of `history_size`)
        self._starting_times = defaultdict(int)  # used to keep track of the times during pauses

    def __iadd__(self, other: 'Metrics') -> 'Metrics':
        """
        `Metrics` objects can be added to each other like that: `metrics += other_metrics`
        This does the union of all metrics. Metrics that are in both objects are aggregated by adding their sum and 
        taking the latest `history_size` values where `other_metrics` is considered more recent.
        """
        for name, value in other.sums.items():
            self.sums[name] += value
        for name, values in other.values.items():
            self.values[name] += values
        if self.history_size > 0:
            for name, values in self.values.items():
                values[:] = values[-self.history_size:]
        return self

    def update(self, name: str, value: Number) -> None:
        """
        Record given value for given metric. The metric's sum will be updated as well as its latest values.
        """
        self.values[name].append(value)
        if len(self.values[name]) > self.history_size > 0:
            self.values[name].pop(0)
        if isinstance(value, Number):   # value can sometimes be a list...
            self.sums[name] += value

    def increment(self, name: str) -> None:
        """
        Just update given metric by 1 (typically used to record steps).
        """
        self.update(name, 1)

    @property
    def names(self) -> list[str]:
        """
        Get the names of all the metrics that have been recorded so far.
        """
        return list(self.values)
        
    def val(self, name: str) -> Number:
        """
        Get the latest value for given metric.
        """
        vals = self.values[name] or [0]
        return vals[-1]

    def max(self, name: str) -> Number:
        """
        Get the maximum value for given metric over the last `history_size` values.
        """
        vals = self.values[name] or [0]
        return max(vals)

    def min(self, name: str) -> Number:
        """
        Get the minimum value for given metric over the last `history_size` values.
        """
        vals = self.values[name] or [0]
        return min(vals)

    def sum(self, name: str) -> Number:
        """
        Sum all values for given metric. Contrary to `rolling_sum`, this is over infinite history.
        """
        self._flush(name)
        return self.sums[name]

    def rolling_sum(self, name: str) -> Number:
        """
        Sum the last `history_size` values for given metric.
        """
        values = self.values[name]
        return sum(values, 0)

    def avg(self, name: str) -> Number:
        """
        Average the last `history_size` values for given metric.
        """
        values = self.values[name]
        return sum(values, 0) / max(1, len(values))

    def divide(self, num: str, denom: str) -> Number:
        """ 
        Divide one metric's aggregated sum by the other's. Contrary to `rolling_divide`, this is over infinite history.
        For instance: `loss = metrics.divide('loss', 'num_tokens')`
        """
        num, denom = self.sum(num), self.sum(denom)
        return num / denom if denom else 0

    def rolling_divide(self, num: str, denom: str) -> Number:
        """ 
        Divide one metric by the other after aggregating their last `history_size` values.
        For instance: `wps = metrics.divide('num_tokens', 'wall')`
        """
        num, denom = self.rolling_sum(num), self.rolling_sum(denom)
        return num / denom if denom else 0

    def __contains__(self, name: str) -> bool:
        return bool(self.values[name])

    # Measuring time
    
    def is_timer(self, name: str) -> bool:
        """
        Check whether given metric is a timer
        """
        return name in self._starting_times

    @contextlib.contextmanager
    def timer(self, name: str):
        """
        Measure the time spent running the enclosed code block or decorated function.
        The total time for given timer can then be obtained by calling `metrics.sum(name)`
        
        Note that contrary to `Benchmark`, this method won't cause any slowdown, but it also won't accurately measure 
        the time spent in asynchronous Pytorch operations.
        """
        self.start(name)
        yield
        self.stop(name)

    @contextlib.contextmanager
    def pause(self, name: str):
        """
        Pause the given timer while running the enclosed code block or decorated function. For instance:
        ```
        with metrics.pause('wall'):
            pass  # time won't be recorded here
        ```
        """
        if self._starting_times[name] == 0:
            yield
        else:
            start = time.perf_counter()
            yield
            elapsed = time.perf_counter() - start
            self._starting_times[name] += elapsed

    def start(self, name: str):
        """
        Start measuring time under given name. The timer can be stopped with `stop` or paused with `pause`.
        """
        self._flush(name)
        self._starting_times[name] = time.perf_counter()
    
    def stop(self, name: str):
        """
        Stop measuring time under given name. This can be resumed by calling `start` again. For instance:
        ```
        metrics.stop('wall')
        # time won't be recorded here
        metrics.start('wall')
        ```
        """
        self._flush(name)
        self._starting_times[name] = 0
    
    def _flush(self, name: str):
        # times are never truly recorded until we call this method (this is typically called when pausing or saving
        # metrics)
        start = self._starting_times[name]
        if start:
            elapsed = time.perf_counter() - self._starting_times[name]
            self.update(name, elapsed)
            self._starting_times[name] = time.perf_counter()

    # Saving and loading

    def state_dict(self) -> dict:
        for name in self._starting_times:
            self._flush(name)
        return {
            'values': dict(self.values),
            'sums': dict(self.sums),
        }

    def load_state_dict(self, state_dict: dict):
        self.values.clear()
        self.sums.clear()
        self.values.update(state_dict.get('values', {}))
        self.sums.update(state_dict.get('sums', {}))


def move_to_device(obj: T, device: str, non_blocking: bool = False, dtype: torch.dtype = None) -> T:
    if isinstance(obj, nn.Module):
        obj.device = device
        obj.dtype = dtype
        return obj.to(device=device, dtype=dtype, non_blocking=non_blocking)

    if device == 'cuda':
        device = torch.cuda.current_device()
    def move(x):
        dtype_ = dtype if x.is_floating_point() else None
        return x.to(device, dtype=dtype_, non_blocking=non_blocking)
    return apply(move, obj, torch.Tensor)


def move_to_cpu(obj: T) -> T:
    return move_to_device(obj, device='cpu')


def tokens_as_tensor(
    token_list: list[np.ndarray],
    special_tokens: SpecialTokens,
    shift: bool = False,
    dtype: torch.dtype = None,
) -> tuple[Tensor, LongTensor]:
    """
    Args:
        token_list: list of numpy arrays defining sequences (of ids or features)
        special_tokens: named tuple defining the ids of the special tokens (eos_idx, bos_idx, padding_idx, etc.)
        shift: whether to shift each sequence one position to the right (and put `bos_idx` as the first token)
        dtype: convert float tensors to this data type
    
    Returns: a tuple (tokens, length) with
        tokens: padded tensor of shape (B, T) or (B, T, D)
        lengths: tensor of shape (B,)
    """
    token_list = [torch.as_tensor(tokens) for tokens in token_list]
    
    if token_list[0].is_floating_point():
        assert not shift
        tokens = pad_sequence(token_list, batch_first=True, padding_value=0.0)
        tokens = tokens.to(dtype)
    elif token_list[0].dtype is torch.bool:
        tokens = pad_sequence(token_list, batch_first=True, padding_value=False).bool()
    else:
        if shift:
            bos = torch.tensor([special_tokens.bos_idx])
            token_list = [torch.cat([bos, tokens[:-1]]) for tokens in token_list]
        tokens = pad_sequence(token_list, batch_first=True, padding_value=special_tokens.padding_idx)
        tokens = tokens.long()

    lengths = torch.LongTensor(list(map(len, token_list)))
    return tokens, lengths


def tensor_to_array(obj: T) -> T:
    """
    Recursively convert all tensors in given object (which can be a dict, a list, a tuple or a tensor) to 
    numpy arrays.
    """
    def to_numpy(x: torch.Tensor) -> np.ndarray:
        x = x.cpu()
        if x.dtype == torch.bfloat16:  # numpy does not support this data type, converting to float32
            x = x.float()
        return x.numpy()

    return apply(to_numpy, obj, torch.Tensor)


def tensor_to_list(obj: T) -> T:
    """
    Recursively convert all tensors in given object (which can be a dict, a list, a tuple or a tensor) to 
    Python lists.
    """
    return apply(lambda x: x.cpu().tolist(), obj, torch.Tensor)


def array_to_list(obj: T) -> T:
    """
    Recursively convert all numpy arrays in given object (which can be a dict, a list, a tuple or a numpy array) to 
    Python lists.
    """
    return apply(lambda x: x.tolist(), obj, np.ndarray)


def apply(f: Callable, obj: T, type: Type) -> T:
    """
    Apply a function to an object if it is from the given type, or apply this function recursively to all its elements 
    if it is a list, a tuple, or a dict. Returns the converted object.
    """
    if isinstance(obj, type):
        return f(obj)
    elif isinstance(obj, dict):
        return {k: apply(f, v, type) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [apply(f, v, type) for v in obj]
    return obj


def build_batches(
    indices: np.ndarray,
    size_fn: Callable,
    batch_size: int,
    batch_size_multiple: Optional[int] = None,
    max_lines: Optional[int] = None,
) -> list[list[int]]:
    """
    Given an array of indices and a function that gives the size of the element (or "line") each index corresponds to,
    group these indices into batches whose total size does not exceed `batch_size`, and whose number of elements does
    not exceed `max_lines` and is a multiple of `batch_size_multiple`.
    Note that the `total_size` is not simply the sum of the sizes of a batch's elements, it is the size after padding
    of this batch (i.e., the size of its the biggest element multiplied by the number of elements).

    For instance:

    ```
    indices = [0, 1, 2, 3, 4, 5]
    size_fn = [1000, 2000, 4000, 500, 1000, 1000].__getitem__
    batch_size = 4000
    batch_size_multiple = 1
    max_lines = 2

    build_batches(indices, size_fn, batch_size, batch_size_multiple, max_lines)

    [
        [0, 1],  # size=4000, lines=2
        [2],     # size=4000, lines=1
        [3, 4],  # size=2000, lines=2
        [5],     # size=1000, lines=1
    ]
    """
    batch_size_multiple = batch_size_multiple or 1
    batches = []
    batch = []
    max_len = 0
    lengths = []
    for idx in indices:
        size = size_fn(idx)
        
        if size > batch_size:  # skip elements that are already too big to fit alone in a batch
            continue
        
        lengths.append(size)
        max_len = max(max_len, size)  # these batches will be padded to the longest element
        
        if (max_lines and len(batch) >= max_lines) or max_len * (len(batch) + 1) > batch_size:
            multiple_len = max(
                batch_size_multiple * (len(batch) // batch_size_multiple),
                len(batch) % batch_size_multiple
            )
            batches.append(batch[:multiple_len])
            batch = batch[multiple_len:]
            lengths = lengths[multiple_len:]
            max_len = max(lengths) if lengths else 0
        batch.append(idx)

    if batch:
        batches.append(batch)  # the size of the last batch may not be a multiple of `batch_size_multiple``

    return batches


def convert_from_fairseq(args: dict):
    """
    Convert the names of model hyper-parameters from fairseq, or older versions of Pasero to the current version
    of Pasero.
    """
    # options whose name has changed:
    mapping = [  # (new, old)
        ('embed_dim', 'encoder_embed_dim'),
        ('embed_dim', 'decoder_embed_dim'),
        ('encoder_ffn_dim', 'encoder_ffn_embed_dim'),
        ('decoder_ffn_dim', 'decoder_ffn_embed_dim'),
        ('shared_embeddings', 'share_all_embeddings'),
        ('tied_output_projection', 'share_decoder_input_output_embed'),
        ('encoder_adapter_dim', 'adapter_dim'),
        ('decoder_adapter_dim', 'adapter_dim'),
        ('encoder_adapters', 'adapter_uids'),
        ('encoder_prenorm', 'encoder_normalize_before'),
        ('decoder_prenorm', 'decoder_normalize_before'),
        ('max_source_len', 'max_source_positions'),
        ('max_target_len', 'max_target_positions'),
        ('shared_encoder_embeddings', 'share_encoder_embeddings'),
        ('shared_decoder_embeddings', 'share_decoder_embeddings'),
        ('encoder_expert_interval', 'moe_freq'),
        ('decoder_expert_interval', 'moe_freq'),
        ('encoder_expert_count', 'moe_expert_count'),
        ('decoder_expert_count', 'moe_expert_count'),
        ('encoder_embed_norm', 'layernorm_embedding'),
        ('decoder_embed_norm', 'layernorm_embedding'),
        # old version of Pasero:
        ('encoder_attention_heads', 'attention_heads'),
        ('decoder_attention_heads', 'attention_heads'),
        ('encoder_ffn_dim', 'ffn_dim'),
        ('decoder_ffn_dim', 'ffn_dim'),
        ('encoder_positional_encoding', 'positional_encoding'),
        ('decoder_positional_encoding', 'positional_encoding'),
    ]
    # boolean options that should be reversed:
    reversed = [  # (new, old)
        ('scale_embed', 'no_scale_embedding'),
        ('lang_decoders', 'share_decoders'),
    ]
    # options that used to be optional and now have a default value:
    defaults = {
        'activation_dropout': 0.0,
        'decoder_softmax_sum': True,
        'scale_embed': True,
    }
    for new_name, old_name in mapping:
        if args.get(new_name) is None and args.get(old_name) is not None:
            args[new_name] = args[old_name]

    for new_name, old_name in reversed:
        if args.get(new_name) is None and args.get(old_name) is not None:
            args[new_name] = not args[old_name]

    if args.get('encoder_positional_encoding') is None:
        args['encoder_positional_encoding'] = 'learned' if args.get('encoder_learned_pos') else 'sinusoidal'
    if args.get('decoder_positional_encoding') is None:
        args['decoder_positional_encoding'] = 'learned' if args.get('decoder_learned_pos') else 'sinusoidal'
    if args.get('decoder_only') is not None:
        args['model_type'] = 'decoder' if args.pop('decoder_only') else 'encoder_decoder'
    for name, value in defaults.items():
        if args.get(name) is None:
            args[name] = value
    # remove verbose fairseq options (annoying when logging model arguments)
    for name in 'data', 'langs', 'lang_pairs':
        args.pop(name, None)


@suppress()
def get_cuda_info(cfg: DistributedConfig):
    if not torch.cuda.is_available():
        return []
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(f'cuda:{device}')
    info = [{
        'name': properties.name,
        'memory': f'{properties.total_memory / 2**30:.2f}GiB',
    }]
    info = gather_list(cfg, info)
    return info


@suppress()
def get_sys_info() -> dict:
    info = {}
    slurm_vars = [
        'SLURM_JOB_ID',
        'SLURM_JOB_PARTITION',
        'SLURM_JOB_CPUS_PER_TASK',
        'SLURM_JOB_CPUS_PER_NODE',
        'SLURM_CPUS_PER_TASK',
        'SLURM_MEM_PER_NODE',
        'SLURM_NODELIST',
        'SLURM_MEM_PER_GPU',
        'SLURM_NTASKS',
        'SLURM_NNODES',
        'SLURM_PROCID',
        'SLURM_LOCALID',
        'SLURM_JOB_GPUS',
        'NCCL_SOCKET_IFNAME',
        'SLURP_RUN_DIR',
    ]
    for key in slurm_vars:
        value = os.environ.get(key)
        if value:
            info[key] = value
    info['Hostname'] = socket.gethostname()
    try:
        import git
        info['git commit'] = git.Repo().head.object.hexsha[:7]
    except:
        pass
    info['Torch version'] = torch.__version__
    info['CUDA version'] = torch.version.cuda
    info['cuDNN version'] = torch.backends.cudnn.version()
    info['Python version'] = sys.version.replace('\n', ' ')
    info['working dir'] = os.getcwd()
    info['python path'] = sys.executable
    return info


@suppress()
def get_sys_stats():
    """ Get GPU usage information (memory and %utilization) using the `nvidia-smi` command """
    if not torch.cuda.is_available():
        return {}
    out = subprocess.check_output(
        ['nvidia-smi', '--format=csv,noheader', '--query-gpu=utilization.gpu,utilization.memory'],
        stderr=subprocess.DEVNULL,
    ).decode().strip().split('\n')
    stats = {}
    for gpu_id, gpu_stats in enumerate(out):
        use, mem = gpu_stats.split(',')
        use = int(use.strip(' %'))
        mem = int(mem.strip(' %'))
        stats[f'gpu_{gpu_id}_use'] = use
        stats[f'gpu_{gpu_id}_mem'] = mem
    return stats


@suppress(silent=True)
def get_cpu_mem_used() -> float:
    """ Returns the amount of physical memory used by current process and all its children in GiB """
    process = psutil.Process()
    mem_usage = process.memory_info().rss
    for child in process.children(recursive=True):
        mem_usage += child.memory_info().rss
    return mem_usage / 2**30


@suppress(silent=True)
def get_cpu_mem_left() -> float:
    """ Returns the amount of available physical memory in GiB """
    return psutil.virtual_memory().available / 2**30


class Benchmark:
    """
    Measures time and GPU memory usage. `utils.benchmark` is an instance of `Benchmark` which can be accessed as 
    a global variable. It is currently used in `Transformer` (to benchmark its encoder, decoder, attention, output
    projection and loss computation) and in `Trainer` (to benchmark the forward and backward passes, the optimizer
    steps and the data loading). It is disabled by default and needs to be enabled by calling
    `utils.benchmark.enable()`. Note that it can cause a significant slow-down, which is the reason why we do not
    enable it by default.
    
    Example of uses:

    As a context manager:
    ```
    with utils.benchmark('forward'):  # GPU memory and total time will be recorded under the name 'forward'
        loss = model(**batch)

    with utils.benchmark('backward'):
        loss.backward()
    ```

    As a function decorator:
    ```
    @utils.benchmark('forward')
    def forward(self, x):
        pass
    ```

    It can be temporarily disabled like this:
    ```
    with utils.benchmark.pause():
        evaluate(model)  # time and GPU usage of the `forward` function won't be recorded
    ```

    All the metrics can be accessed thanks to the `utils.benchmark.metrics` property:
    ```
    print(utils.benchmark.metrics)
    {
        'max_mem': ...,
        'forward_wall': ...,
        'forward_mem': ...,
        'forward_peak_mem': ...,
        'backward_wall': ...,
        ...
    }
    ```

    `utils.benchmark.reset()` can be called to reset all memory statistics.
    """
    def __init__(self, use_cuda: bool = True, enabled: bool = True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.enabled = enabled
        self.timers = {}
        self.mem_usage = {}        # measures the maximum allocated memory by a block of code (difference of allocated
        # memory after and before)
        self.peak_mem_usage = {}   # measures the maximum allocated memory after a block of code
        self.ongoing = {}          # keeps track of the peak allocated memory for the ongoing context managers
        self.max_mem = 0           # global peak memory, reset after each log() call
    
    def reset(self):
        """
        Reset all memory statistics, currently called after each training step, after aggregating the statistics in
        a `Metrics` object.
        """
        self.timers.clear()
        self.mem_usage.clear()
        self.peak_mem_usage.clear()
        self.ongoing.clear()
        self.max_mem = 0
        if self.use_cuda:
            torch.cuda.reset_peak_memory_stats()

    @contextlib.contextmanager
    def __call__(self, name: str):
        """
        Can be used as a context manager (`with` block) or as a function decorator. Will measure the time to run
        the enclosed block or decorated function and GPU memory difference between its start and its end, as well
        as peak GPU memory usage. The will be recorded under the given name. Note that the CUDA operations are likely
        to be slower, due to the added synchronization primitives (which are necessary to make the measurement
        reliable).

        This is called like this: 
        
        ```
        benchmark = utils.Benchmark()
        with benchmark('name'):
            pass  # do stuff
            
        # or
        @benchmark('name')
        def function():
            pass  # do stuff
        ```
        """
        if not self.enabled or name in self.ongoing:
            yield
            return
        total = self.timers.get(name, 0)
        if self.use_cuda:
            torch.cuda.synchronize()
            max_allocated = torch.cuda.max_memory_allocated()
            for k, v in self.ongoing.items():
                self.ongoing[k] = max(v, max_allocated)
            self.max_mem = max(self.max_mem, max_allocated)
            torch.cuda.reset_peak_memory_stats()
            allocated_before = torch.cuda.memory_allocated()
            self.ongoing[name] = allocated_before
        start = time.perf_counter()
        yield
        if self.use_cuda:
            torch.cuda.synchronize()
            allocated_after = max(torch.cuda.max_memory_allocated(), self.ongoing.pop(name))
            self.mem_usage[name] = max(allocated_after - allocated_before, self.mem_usage.get(name, 0))
            self.peak_mem_usage[name] = max(allocated_after, self.peak_mem_usage.get(name, 0))

        elapsed = time.perf_counter() - start
        self.timers[name] = total + elapsed

    @property
    def metrics(self) -> dict[str, float]:
        """
        Get the recorded statistics as a dictionary:
        ```
        {
            'max_mem': ...,
            '{name}_wall': ...,
            '{name}_mem': ...,
            '{name}_peak_mem': ...,
            ...
        }  # {name} is a name given to the context manager: `with benchmark(name)`
        ```
        """
        if self.use_cuda:   # in case benchmark is disabled
            self.max_mem = max(self.max_mem, torch.cuda.max_memory_allocated())
        metrics_ = {}
        for k, v in self.timers.items():
            metrics_[f'{k}_wall'] = v
        if self.use_cuda:
            metrics_['max_mem'] = self.max_mem / 2**20
            for k, v in self.mem_usage.items():
                metrics_[f'{k}_mem'] = v / 2**20
            for k, v in self.peak_mem_usage.items():
                metrics_[f'{k}_peak_mem'] = v / 2**20
        return metrics_

    @contextlib.contextmanager
    def pause(self):
        """
        Can be used as a context manager (`with` block) or as a function decorator. Temporarily disable benchmarking
        while running the enclosed code or decorated function.
        """
        enabled = self.enabled
        self.enabled = False
        yield
        self.enabled = enabled
    
    def enable(self) -> None:
        self.enabled = True
    
    def disable(self) -> None:
        self.enabled = False
    
    def cpu(self) -> None:
        """
        Disable CUDA memory measurements.
        """
        self.use_cuda = False


# Global instance of `Benchmark` that is used in `Trainer` and `Transformer` to measure GPU memory usage and running
# time of different components at training or inference. Because it can impact performance, it is disabled by default
# (i.e., it won't record anything) and can be enabled by calling `benchmark.enable()`
benchmark = Benchmark(enabled=False)


@suppress()
def safe_symlink(src: str, dst: str):
    """
    Try to create a symbolic link at given destination. Try deleting the destination file if it already exists.
    This won't cause an error if this fails.
    """
    safe_delete(dst)
    os.symlink(src, dst)


@suppress()
def safe_delete(path: Optional[str]):
    """
    Delete a file without causing an error if the file doesn't exist.
    Doesn't delete the file if a symlink to this file exists in the same directory
    """
    if path is None:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@suppress()
def safe_copy(src: str, dst: str):
    """ Try coping given file to given destination without causing an error if it fails. """
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


class ExperimentTracker:
    def __init__(self, cfg: TrackerConfig, sys_info: dict, model_dir: str):
        project_name = cfg.tracker_project_name or os.environ.get('USER')
        run_name = cfg.tracker_run_name or os.path.basename(model_dir)
        backend = cfg.tracker if is_master(cfg) else None
        self.backend = None
        self.run = None
        
        with suppress():
            sys_info = {k.replace(' ', '_'): v for k, v in sys_info.items()} if sys_info else {}
            if backend == 'wandb':
                logger.warning('--tracker mlflow is currently deprecated')  # FIXME: the API has probably changed since
                # this was last tried
                import wandb
                self.run = wandb.init(project=project_name, name=run_name, config={**cfg.as_dict(), **sys_info})
                self.backend = backend
            elif backend == 'neptune':
                logger.warning('--tracker neptune is currently deprecated')  # FIXME: the API has probably changed since
                # this was last tried
                import neptune.new as neptune
                from neptune.new.integrations.python_logger import NeptuneHandler
                self.run = neptune.init(project=project_name, name=run_name, source_files=[])
                logging.getLogger().addHandler(NeptuneHandler(run=self.run))
                self.run['parameters'] = {**cfg.as_dict(), **sys_info}
                self.backend = backend
            elif backend == 'mlflow':
                import mlflow
                # no need for a proxy as MLflow runs locally
                os.environ.pop('http_proxy', None)
                os.environ.pop('https_proxy', None)
                URI = os.environ.get('MLFLOW_TRACKING_URI')
                if URI:
                    logger.info(f'initializing MLflow at URI {URI}')
                    # this will possibly raise an exception, in which case no need for further calls
                    mlflow.set_experiment(project_name)   # FIXME: this runs for a long time if URI is not found
                    mlflow.start_run(run_name=run_name)
                    for k, v in cfg.as_dict().items():
                        try: mlflow.log_param(k, v)
                        except mlflow.exceptions.MlflowException as e: logger.warning(str(e))
                    mlflow.log_dict(cfg.as_dict(), 'config.json')
                    # to get a good idea of the command line that was issued
                    mlflow.log_text(' '.join(sys.argv), 'sys.argv.txt')
                    mlflow.log_dict(sys_info, 'sys_info.json')
                    for name in 'SLURM_JOB_ID', 'SLURP_RUN_DIR', 'git_commit':
                        if sys_info.get(name):
                            mlflow.set_tag(name, sys_info[name])
                    self.run = mlflow
                    self.backend = backend
                else:
                    logger.warning('MLFLOW_TRACKING_URI is not set')
        if backend is not None and backend != 'none' and self.backend is None:
            logger.info(f'failed to initialize experiment tracker: {backend}')

    @suppress(caller='tracker', max_attempts=3)
    def log_step(self, step: int):
        if self.backend == 'neptune':
            self.run['step'].log(step)

    @suppress(caller='tracker', max_attempts=3)
    def log(self, data: dict, step: int):
        if self.backend == 'wandb':
            self.run.log(data, step=step)
        elif self.backend == 'neptune':
            for k, v in data.items():
                self.run[k].log(v)   # no step argument in new API?
        elif self.backend == 'mlflow':
            sys_stats = get_sys_stats() or {}
            # Hack for chrF++
            # mlflow.exceptions.RestException: INVALID_PARAMETER_VALUE: Invalid metric name: 'law.test.100.de-en/chrf++'. 
            # Names may only contain alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/).
            data = {k.replace('++', 'pp'): v for k, v in data.items()}
            self.run.log_metrics({**data, **sys_stats}, step=step)

    @suppress(caller='tracker', max_attempts=3)
    def log_data(self, name, data):
        # Log the data under an artifact with given name (used to log segment-level scores)
        if self.backend == 'mlflow':
            self.run.log_dict(data, f'{name}.yaml')

    @suppress()
    def finish(self):
        if self.backend == 'wandb':
            self.run.finish()
        elif self.backend == 'neptune':
            self.run.stop()
        elif self.backend == 'mlflow':
            self.run.end_run()


def heatmap(xlabels, ylabels, weights, output_file=None):
    """
    Draw a heatmap showing the alignment between two sequences.
    :param xlabels: input words
    :param ylabels: output words
    :param weights: array of shape (len(ylabels), len(xlabels))
    :param output_file: write the figure to this file, or show it into a window if None
    """
    from matplotlib import pyplot as plt
    import numpy as np

    weights = np.array(weights) * 100
    weights = np.array(weights)[:len(ylabels),:len(xlabels)]

    fig, ax = plt.subplots()
    im = ax.imshow(weights, cmap=plt.cm.Reds)

    xlabels = [label.replace('</s>', 'EOS') for label in xlabels]
    ylabels = [label.replace('</s>', 'EOS') for label in ylabels]

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    ax.xaxis.tick_top()
    ax.set_xticklabels(xlabels, minor=False)
    ax.set_yticklabels(ylabels, minor=False)
    ax.tick_params(axis='both', which='both', length=0)
    # ax.grid(True)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)

    # Loop over data dimensions and create text annotations.
    if len(xlabels) <= 20 and len(ylabels) <= 20:
        max_ = np.max(weights)
        min_ = np.min(weights)
        for i in range(len(xlabels)):
            for j in range(len(ylabels)):
                v = weights[j, i]
                d = max(1, max_ - min_)
                color = 'white' if (v - min_) / d > 0.5 else 'black'
                text = ax.text(i, j, int(weights[j, i]),
                    ha='center', va='center', color=color)

    # ax.set_frame_on(False)
    xsize = max(len(xlabels) / 2, 2)
    ysize = max(len(ylabels) / 2, 2)

    fig.set_size_inches(xsize, ysize, forward=True)
    # plt.autoscale(enable=True, axis='x', tight=True)
    # plt.subplots_adjust(wspace=0, hspace=0)
    # ax.set_aspect('equal')
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if output_file is None:
        plt.show()
    else:
        dirname = os.path.dirname(output_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(output_file, bbox_inches='tight')


def setup_distributed(cfg: DistributedConfig) -> tuple[int, int]:
    """
    For simplicity, we limit the number of possible TP / DP configurations.

    No TP:
        - any dp_size (= node_size by default)
        - tp_size = 1
    
    Or per-node TP:
        - tp_size = node_size
        - dp_size = num_nodes
        - dp_local_size = 1
    
    Or single-GPU nodes (via SLURM):
        - any tp_size (= 1 by default)
        - dp_size = num_tasks / tp_size
        - node_size = dp_local_size = 1
    
    `dp_local_size` controls the way the datasets read their data. If `per_gpu_batching` is not set to True,
    only the local master (`dp_local_rank == 0`) will read data and send it to the other local ranks (on the same node)
    via queues. Tensor parallelism forces

    """
    if int(os.environ.get('SLURM_NTASKS', 1)) > 1:  # using --ntasks N --gpus-per-task 1
        # spawning N processes of 1 GPU each
        assert int(os.environ.get('SLURM_GPUS_PER_TASK', '1')) == 1
        world_size = int(os.environ['SLURM_NTASKS'])
        cfg.dp_local_size = node_size = 1
        cfg.dp_local_rank = 0
        cfg.start_rank = int(os.environ['SLURM_PROCID'])  # task id in [0, NTASKS)
        start_device = int(os.environ['SLURM_LOCALID'])  # may be != 0 if several tasks are assigned to the same SLURM node
        assert world_size % cfg.tp_size == 0
        cfg.dp_size = world_size // cfg.tp_size
        node_list = os.environ['SLURM_JOB_NODELIST']
        # take the first node in the list of nodes as the master:
        # tholus-[3,4] | tholus-[3-4] | tholus-3,tholus-4 -> tholus-3
        master = regex.sub(r'\[(\d+)([,-]\d+)*\]', r'\1', node_list).split(',')[0]
        master = socket.gethostbyname_ex(master)[0]
        # for the port, find a number that is unique but shared between tasks, the SLURM job id is a good candidate
        slurm_job_id = int(os.environ['SLURM_JOB_ID'])
        port = 10000 + slurm_job_id % 10000
        cfg.distributed_init_method = f'tcp://{master}:{port}'
    else:  # --ntasks 1 --gpus-per-task N
        node_size = torch.cuda.device_count() or 1
        if cfg.tp_size > 1:
            cfg.dp_size = cfg.dp_size or 1
        if cfg.dp_size:  # manually set the number of GPUs
            node_size = min(node_size, cfg.dp_size * cfg.tp_size)  # may use fewer GPUs than available
        else: # single-node setting, infer dp_size from the number of available GPUs
            cfg.dp_size = node_size // cfg.tp_size
        assert cfg.tp_size == 1 or cfg.tp_size == node_size
        world_size = cfg.tp_size * cfg.dp_size
        assert world_size % node_size == 0
        num_nodes = world_size // node_size
        if cfg.tp_size > 1:  # tp_size = node_size
            cfg.dp_local_size = 1
        else:  # dp_size = node_size
            cfg.dp_local_size = node_size
        if num_nodes == 1:
            cfg.start_rank = 0
            port = random.randint(10000, 20000)
            cfg.distributed_init_method = f'tcp://localhost:{port}'
        else:
            assert cfg.distributed_init_method, ('distributed training requires --distributed-init-method')
            assert cfg.start_rank is not None, ('distributed training requires --start-rank')
        start_device = 0  # assumes that all visible GPUs are available
    
    if world_size > 1:
        assert torch.cuda.is_available()
    
    assert cfg.tp_size == 1 or cfg.dp_size == 1, 'combined data and tensor parallelism is not yet supported'
    return start_device, node_size


def parse_logs(log_filename: str) -> dict:
    """
    Parse a log file as created by `pasero-train` and return a dictionary of {corpus_name: {step: metrics}} where
    metrics are dictionaries mapping metric names (e.g., 'nll_loss') to their float or integer value.
    """
    import dateutil.parser
    regex_ = r"(?P<date>[\d-:, ]+) \| train( \| (?P<corpus>[^ ]+))? \| steps (?P<steps>\d+)( \| (?P<metric>[\w+]+) (?P<value>[^ ]+))+"
    logs = {}

    def float_or_int(x):
        x_as_float = float(x)
        try:
            x_as_int = int(x)
            if x_as_int == x_as_float:
                return x_as_int
        except:
            pass
        return x_as_float

    with open(log_filename) as log_file:
        for line in log_file:
            line = line.strip()
            if (m := regex.match(regex_, line)):
                date = dateutil.parser.parse(m.group('date'))
                corpus = m.group('corpus') or 'train'
                steps = int(float(m.group('steps')))
                metrics = m.captures('metric')
                values = [float_or_int(val.rstrip('%')) for val in m.captures('value')]
                assert len(values) == len(metrics)
                metrics = dict(zip(metrics, values))
                metrics['date'] = date.timestamp()

                logs.setdefault(corpus, {})[steps] = metrics

    return logs


def optimizer_checkpoint(model_ckpt_path: str):
    """
    'model/model_1000_001_of_004.bin' -> 'model/optimizer_1000_001_of_004.bin'
    """
    model_ckpt_path = os.path.realpath(model_ckpt_path)  # resolves symlinks
    dirname, filename = os.path.split(model_ckpt_path)
    assert filename.startswith('model_')
    filename = 'optimizer_' + filename.removeprefix('model_')
    return os.path.join(dirname, filename)


def metrics_checkpoint(model_ckpt_path: str):
    """
    'model/model_1000_001_of_004.bin' -> 'model/metrics_1000_001_of_004.bin'
    """
    model_ckpt_path = os.path.realpath(model_ckpt_path)  # resolves symlinks
    dirname, filename = os.path.split(model_ckpt_path)
    assert filename.startswith('model_')
    filename = 'metrics_' + filename.removeprefix('model_')
    filename = regex.sub(r'_\d{3}_of_\d{3}', '', filename)  # remove shard names, as metrics are not sharded
    return os.path.join(dirname, filename)


def find_checkpoint_to_load(
    model_dir: str,
    other_ckpt: Optional[str] = None,
    reset: bool = False,
) -> tuple[Optional[str], bool]:
    """
    Look for an existing checkpoint in given model directory. If none exists, return `other_ckpt` (which can 
    be a filename in `model_dir` or a path relative to the working directory).

    This is used by `Trainer` to resume training (the default, unless `--reset` is set) or load the checkpoint that 
    was specified with `--ckpt`.
    
    If some finetuning instance is aborted (e.g., preempted), restarting the training script with the same options 
    should resume training rather than reload the initial checkpoint that was being finetuned. This is why 
    existing checkpoints have a higher priority than `--ckpt`.

    Returns: tuple `(path_to_checkpoint, found_existing)` where `found_existing` is a boolean indicating whether
        the returned checkpoint was found in `model_dir` or is `other_ckpt`
    """
    latest_ckpt = os.path.join(model_dir, 'model_latest.bin')   # saved by train.py when it is interrupted
    last_ckpt = os.path.join(model_dir, 'model_last.bin')       # saved periodically by train.py
    
    existing_ckpt = None
    
    def checkpoint_exists(ckpt):
        # Do not resume training if some parts of the model (e.g., the optimizer) are missing.
        # If those files exist but are corrupted (as can happen with 'model_latest.bin'), this will be handled
        # in `Trainer`
        return (
            os.path.isfile(ckpt) and
            os.path.isfile(optimizer_checkpoint(ckpt)) and
            os.path.isfile(metrics_checkpoint(ckpt))
        )

    if not reset:   # look for existing checkpoints in model_dir to restore in priority (and ignore `other_ckpt`)
        if checkpoint_exists(latest_ckpt):
            existing_ckpt = latest_ckpt
        elif checkpoint_exists(last_ckpt):
            existing_ckpt = last_ckpt

    found_ckpt = existing_ckpt
    found_existing = existing_ckpt is not None

    # no existing checkpoint found in model_dir (or reset is set), try other_ckpt
    if found_ckpt is None and other_ckpt is not None:
        dirs = None
        if other_ckpt == os.path.basename(other_ckpt):
            # interpret simple names (e.g., "model_best.bin" or "model_100000.bin") as being files by that 
            # name in the model directory
            dirs = [model_dir]
        # if a path is given (i.e., containing '/'), it is interpreted as being relative to the working directory
        found_ckpt = find_file(other_ckpt, dirs=dirs, fail=True)

    return found_ckpt, found_existing


def load_checkpoint(*paths: str, load_train_state: bool = False) -> dict:
    """
    Load given checkpoint. If several paths are given: the first is assumed to be the main checkpoints and the others
    checkpoint's model parameters are loaded and merged into this main checkpoint. Several paths may be given when
    loading a base model and adapters for instance.

    If `load_train_state` is True, also attempt to load optimizer and metrics checkpoints (whose names are inferred 
    from the main checkpoint's path).

    This method should also work with older checkpoints that stored everything (model parameters and optimizer states)
    into a single file.

    Returns a single dictionary with all the relevant keys: model, optimizer, metrics, etc.
    """
    assert len(paths) >= 1
    ckpt = None

    for path in paths:
        ckpt_ = torch.load(path, map_location='cpu')
        if 'model' not in ckpt_:  # HF format
            ckpt_['model'] = {}
            for k in list(ckpt_):
                if torch.is_tensor(ckpt_[k]):
                    ckpt_['model'][k] = ckpt_.pop(k)
    
        if ckpt is None:
            ckpt = ckpt_
        else:  # merge model parameters
            for k, v in ckpt_['model'].items():
                # do not overwrite existing keys: the first checkpoint has precedence
                ckpt['model'].setdefault(k, v)

    if load_train_state:
        # In old versions of Pasero, the metrics and optimizer states were stored in the model checkpoint.
        # Now, they are stored in separate checkpoints. This code works for both formats.
        optimizer_path = optimizer_checkpoint(paths[0])
        if os.path.exists(optimizer_path):
            ckpt_: dict = torch.load(optimizer_path, map_location='cpu')
            ckpt.update(ckpt_)
    
        metrics_path = metrics_checkpoint(paths[0])
        if os.path.exists(metrics_path):
            ckpt_: dict = torch.load(metrics_path, map_location='cpu')
            ckpt.update(ckpt_)
    else:
        # these can take a lot of memory, remove them if they are not needed
        for key in 'optimizer', 'scheduler', 'scaler':
            ckpt.pop(key, None)

    return ckpt


def find_checkpoint_shards(model_ckpt_path: str) -> list[str]:
    """
    Locate the shards of given checkpoint path, which should be the first shard or a symbolic link to it.
    """
    # resolve symbolic links (e.g., model_last.bin -> model_10000.bin)
    model_ckpt_path = os.path.realpath(model_ckpt_path)
    ckpt_dir, ckpt_name = os.path.split(model_ckpt_path)
    paths = [model_ckpt_path]
    
    # find the other shards
    match = regex.search(r'_(\d{3})_of_(\d{3}).bin', ckpt_name)
    if match:
        shard_id = int(match.group(1)) - 1
        shard_count = int(match.group(2))
        suffix = match.group(0)
        assert shard_id == 0

        for shard_id in range(1, shard_count):
            name = ckpt_name.removesuffix(suffix) + f'_{shard_id + 1:03}_of_{shard_count:03}.bin'
            path = os.path.join(ckpt_dir, name)
            assert os.path.exists(path), f'shard {name} does not exist'
            paths.append(path)
    
    return paths


def load_and_reshard_checkpoint(
    model_cls,
    model_ckpt_path: str,
    model_shard_id: int = 0,
    model_shard_count: int = 1,
    load_train_state: bool = False,
) -> dict:
    """
    With Tensor Parallelism or Tutel MoEs, the checkpoints may be in several shards. At finetuning, we may want
    to use a different number of shards. Find the list of checkpoint shards and which ones should go to this 
    rank, and reshard them if needed.

    Args:
        - model_cls
        - model_ckpt_path: path to the first shard
        - model_shard_id: the id of the local model shard (aka rank)
        - model_shard_count: number of shards the model now has
        - load_train_state: whether to load the optimizer states (incompatible with resharding)
    
    Returns:
        - the checkpoint that should go to the local rank, whose model parameters may have been resharded, in the same 
          dict format as returned by `load_checkpoint`
    """
    shard_paths = find_checkpoint_shards(model_ckpt_path)

    if len(shard_paths) > model_shard_count:
        # there are more checkpoint shards than GPUs: they will need to be merged
        assert not load_train_state, 'optimizer states do not support resharding, either train on the same ' \
            'number of GPUs or use --reset-optimizer'
        assert len(shard_paths) % model_shard_count == 0
        ckpt_per_gpu = len(shard_paths) // model_shard_count
        local_shard_paths = shard_paths[
            model_shard_id * ckpt_per_gpu:(model_shard_id + 1) * ckpt_per_gpu
        ]
        shards = [
            load_checkpoint(shard_path, load_train_state=False)
            for shard_path in local_shard_paths
        ]
        
        model_states = [shard['model'] for shard in shards]
        main_ckpt = shards[0]
        main_ckpt['model'] = model_cls.unshard_state_dict(
            *model_states,
            total_shard_count=len(shard_paths)
        )
    elif len(shard_paths) < model_shard_count:
        # there are more GPUs than checkpoint shards: they will need to be resharded
        assert not load_train_state, 'optimizer states do not support resharding, either train on the same ' \
            'number of GPUs or use --reset-optimizer'
        assert model_shard_count % len(shard_paths) == 0
        gpus_per_ckpt = model_shard_count // len(shard_paths)
        local_shard_path = shard_paths[model_shard_id // gpus_per_ckpt]
        main_ckpt = load_checkpoint(local_shard_path, load_train_state=False)
        main_ckpt['model'] = model_cls.shard_state_dict(
            main_ckpt['model'],
            shard_id=model_shard_id % gpus_per_ckpt,  # which part of this shard should go to this rank
            shard_count=gpus_per_ckpt,  # number of parts this shard should be split into
            total_shard_count=len(shard_paths),  # total number of checkpoint shards
        )
    else:
        local_shard_path = shard_paths[model_shard_id]
        main_ckpt = load_checkpoint(local_shard_path, load_train_state=load_train_state)

    return main_ckpt
