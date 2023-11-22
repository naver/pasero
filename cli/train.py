#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import sys
import os
import regex
import signal
import logging
import time
import traceback
import yaml
import tarfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional

try:
    import torch.distributed.nn
    # disable annoying logging messages
    torch.distributed.nn.jit.instantiator.logger.disabled = True
except:
    pass

logger = logging.getLogger('train')

from pasero import utils, models, tasks, evaluation, datasets
from pasero.models import modules
from pasero.training import Trainer, MultiprocessingStatus
from pasero.datasets import ValidationDataset, TrainingDataset
from pasero.config import TrainingConfig
from pasero.tasks import Task


class SignalHandler:
    def __init__(self, status):
        self.status = status
        self.sig = self
        self.restart = False
    
    def set_signals(self):
        signal.signal(signal.SIGTERM, self.sig)
        signal.signal(signal.SIGINT, self.sig)
        signal.signal(signal.SIGUSR1, self.sig)
    
    def __call__(self, sig, frame):
        if sig == signal.SIGUSR1:
            self.restart = True
        self.status.interrupt()
        with utils.suppress(silent=True):  # sometimes print fails in signal handlers
            print(f'Caught {signal.Signals(sig).name}', file=sys.stderr)


class IgnoreHandler(SignalHandler):
    def __init__(self):
        self.sig = signal.SIG_IGN


def run_dist_training(
    cfg: TrainingConfig,
    status_mp: MultiprocessingStatus,
    queues: Optional[list[mp.Queue]] = None,
    signal_handler: Optional[SignalHandler] = None,
):
    """ Wrapper for main() which ignores signals and kills children upon exit """
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(cfg.device)
        
        if cfg.dtype == 'bfloat16':
            # save time and CPU memory by automatically creating tensors of the right type
            torch.set_default_dtype(torch.bfloat16)
        
        if utils.is_distributed(cfg):
            dist.init_process_group(
                'nccl',
                rank=cfg.distributed_rank,
                world_size=cfg.distributed_world_size,
                init_method=cfg.distributed_init_method,
            )

        if cfg.distributed_rank > 0:  # disable breakpoints on non-master ranks
            sys.breakpointhook = lambda *args, **kwargs: None

        run_training(cfg, status_mp, queues, signal_handler)
    except:
        status_mp.fail()
        traceback.print_exc()
    finally:
        if utils.is_distributed(cfg):
            dist.destroy_process_group()
        for p in mp.active_children():
            p.kill()


def run_training(
    cfg: TrainingConfig,
    status_mp: MultiprocessingStatus,
    queues: Optional[list[mp.Queue]] = None,
    signal_handler: Optional[SignalHandler] = None,
):
    if utils.is_master(cfg) and cfg.benchmark:
        utils.benchmark.enable()   # if disabled, utils.Benchmark(...) will be a no-op

    if not torch.cuda.is_available():
        utils.benchmark.cpu()
    
    cfg.dp_size = cfg.dp_size or 1   # in case it is still None
    cfg.seed = utils.broadcast(cfg, cfg.seed, dtype=torch.int)  # make sure all processes have the same seed
    tp_group = utils.get_tp_group(cfg)
    modules.set_tp_group(tp_group)
    modules.set_sequence_parallel(cfg.sequence_parallel)
    
    utils.set_random_seed(cfg.seed)  # sets the PyTorch, numpy and Python seeds

    os.makedirs(cfg.model_dir, exist_ok=True)

    # Write logging outputs to files in the checkpoint directory:
    # train.log for master process, and train.log.{i} for the other processes
    if cfg.log_file:
        log_file = f'{cfg.log_file}.{cfg.distributed_rank}'.removesuffix('.0')
        log_file = os.path.join(cfg.model_dir, log_file)
    else:
        log_file = None
    level = logging.DEBUG if cfg.verbose else logging.INFO
    stream = sys.stdout if utils.is_master(cfg) else None
    utils.init_logging(log_file, level=level, stream=stream, append=not cfg.reset)

    logger.info(' '.join(sys.argv))  # log command line parameters
    sys_info = utils.get_sys_info() or {}
    tracker = utils.ExperimentTracker(cfg, sys_info=sys_info, model_dir=cfg.model_dir)

    logger.info(cfg.as_dict())

    for k, v in sys_info.items():
        logger.info(f'{k} {v}')
    for rank, info in enumerate(utils.get_cuda_info(cfg)):
        logger.info(f"rank {rank} | {info['name']} | {info['memory']}")

    # log_interval decides the window size for the rolling averages
    train_metrics = utils.Metrics(history_size=cfg.log_interval)
    train_metrics.start('total_wall')

    # find the model, task and dataset classes specified by the training config (--arch, --task, --dataset-type)
    arch_cls = models.get_architecture(cfg.model_cfg)
    task_cls = tasks.get_task_class(cfg.task)
    dataset_cls = datasets.get_dataset_class(cfg.dataset_cfg)
    
    # corpus definitions (do not contain actual data)
    train_corpora = task_cls.get_train_corpora(cfg.task_cfg, cfg.data_dir, cfg.train_corpora)
    valid_corpora = task_cls.get_valid_corpora(cfg.task_cfg, cfg.data_dir, cfg.valid_corpora)

    if train_corpora:
        logger.info('Training corpora: ' + repr(train_corpora))
    if valid_corpora:
        logger.info('Validation corpora: ' + repr(valid_corpora))

    if cfg.max_steps > 0:
        assert train_corpora and valid_corpora
        assert any(corpus.early_stopping for corpus in valid_corpora), "at least one validation corpus should have " \
        "the early_stopping property set to True"
    else:
        assert valid_corpora
    
    # TODO: if model_cfg.disable_bos, set task.special_tokens.bos_idx to None
    task = task_cls(cfg.data_dir, cfg.task_cfg)
    task.set_model_type(cfg.model_cfg.model_type)
    task.register_corpora(*train_corpora, *valid_corpora)

    if utils.is_master(cfg) and cfg.max_steps > 0:
        # Copy the tokenizers' files to the model directory (BPE model, dict, etc.)
        # This helps with inference, since dictionary and tokenizer paths at inference are relative to the model
        # directory (contrary to training, where they are relative to the data directory). However, this copies all
        # files at the model root. For instance, with "--dict de/dict.txt --target-dict en/dict.txt" will result in a 
        # single "dict.txt" in the model directory.
        basenames = {os.path.basename(path) for path in task.preprocessor_files}
        if len(basenames) != len(task.preprocessor_files):
            logger.warning('some preprocessing files (dicts, tokenizers) share the same filename: this will cause '
                           'issues at inference because they are copied to the model directory under the same name')
        for path in task.preprocessor_files:
            utils.safe_copy(path, cfg.model_dir)
        
        # Save inference options into "inference.yaml": lets us decode from the model without having to manually
        # specify these options
        inference_config_file = os.path.join(cfg.model_dir, 'inference.yaml')
        inference_options = {**cfg.inference_options, **task.inference_options}
        with open(inference_config_file, 'w') as file:
            yaml.safe_dump(inference_options, file)

        with utils.suppress():
            # Try saving the Pasero code as an archive in the model directory for reproducibility
            with tarfile.open(os.path.join(cfg.model_dir, 'pasero.tgz'), 'w:gz') as tar:
                # location of pasero (__file__ is pasero/cli/train.py)
                this_dir = os.path.dirname(os.path.dirname(__file__))
                dest_dir = 'pasero'
                for subdir in 'pasero', 'cli':
                    tar.add(os.path.join(this_dir, subdir), arcname=os.path.join(dest_dir, subdir))

    # Build model
    # -----------
    # This should be done before creating the datasets, as the architecture can modify `cfg` (which is copied in
    # TrainingDataset)
    cfg.amp = cfg.amp and torch.cuda.is_available()
    if not torch.cuda.is_available():
        cfg.dtype = 'float32'
    cfg.fsdp = cfg.fsdp and torch.cuda.is_available() and cfg.dp_size > 1

    if cfg.tp_size > 1:
        # set different seeds on different Tensor Parallelism ranks, so that the parameter shards are initialized 
        # to different values
        torch.manual_seed(cfg.seed + cfg.tp_rank)
        torch.cuda.manual_seed(cfg.seed + cfg.tp_rank)

    model: models.Transformer = arch_cls(cfg.model_cfg, cfg, task=task)
    
    if cfg.tp_size > 1:
        # with Tensor Parallelism, all ranks don't always have the exact same number of parameters, which means that
        # the RNG will be out of sync after creating the model. So we need to reset it to ensure that we get the same
        # dropout across all ranks.
        torch.manual_seed(cfg.seed + cfg.dp_rank)
        torch.cuda.manual_seed(cfg.seed + cfg.dp_rank)
    
    unfreeze = set()
    for name, param in model.named_parameters():
        if 'frozen_embedding' in name.split('.'):
            param.requires_grad = False
            # unfreeze the "non-frozen" embeddings, avoids having to manually specify --freeze-params-regex
            unfreeze.add(name.replace('.frozen_embedding', ''))
        elif cfg.freeze_params_regex:
            param.requires_grad = not regex.match(cfg.freeze_params_regex, name)
        elif cfg.train_params_regex:
            param.requires_grad = bool(regex.match(cfg.train_params_regex, name))
        elif cfg.model_cfg.lora_rank:
            param.requires_grad = '.lora.' in name
    for name, param in model.named_parameters():
        if name in unfreeze:
            param.requires_grad = True

    logger.info(model)
    for name, param in model.named_parameters():
        shape = 'x'.join(map(str, param.shape))
        frozen = '' if param.requires_grad else ', frozen'
        logger.info(f'{name} ({shape}{frozen})')
    
    params = model.total_param_count
    trained_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    params_without_embed = sum(
        param.numel() for name, param in model.named_parameters()
        if not name.endswith('.embed_tokens.weight') and not name.endswith('.output_projection.weight')
    )
    logger.info(f"model params: {params:,} (w/o embed: {params_without_embed:,}, trained: {trained_params:,})")
    
    # Create datasets
    # ---------------
    with train_metrics.timer('load_wall'):
        valid_sets = []
        for valid_corpus in valid_corpora:
            logger.info(f'loading validation set: {valid_corpus}')
            valid_set = ValidationDataset(dist_cfg=cfg, task=task, corpus=valid_corpus)
            valid_sets.append(valid_set)

        if cfg.max_steps == 0:
            train_set = None
        else:
            # TODO: set a dataset seed that changes when resuming training            
            cfg.dataset_cfg.batch_by = cfg.dataset_cfg.batch_by or model.batch_by
            train_set = dataset_cls(
                cfg=cfg.dataset_cfg,
                dist_cfg=cfg,
                task=task,
                corpora=train_corpora,
                queues=queues,
                verbose=cfg.verbose,
            )

    # Load checkpoint
    # ---------------
    trainer = Trainer(cfg, task, model, train_metrics, status_mp)

    # Training
    # ========
    if signal_handler is not None:
        signal_handler.set_signals()

    if cfg.validate_at_start:
        evaluate(cfg, task, valid_sets, trainer, tracker, tp_group=tp_group)

    if trainer.steps < cfg.max_steps:
        if cfg.save_initial_checkpoint:
            trainer.save_model('model_init')
        train(cfg, task, train_set, valid_sets, trainer, tracker, tp_group=tp_group)
    else:
        trainer.status.finish()

    if trainer.status.interrupted():
        # interrupted 'manually' with CTRL+C, or by SLURM
        # try to save a checkpoint, unless the model is sharded (because it is very likely to result in inconsistent
        # models)
        logger.error('training interrupted')
        log(trainer.steps, trainer.metrics, tracker=None)
        if not trainer.saved and not trainer.model.is_sharded:
            trainer.save('model_latest')
    elif trainer.status.failed():
        logger.error('training failed')
        log(trainer.steps, trainer.metrics, tracker=None)
    
    tracker.finish()
    if status_mp is not None:
        status_mp += trainer.status


def train(
    cfg: TrainingConfig,
    task: Task,
    train_set: TrainingDataset,
    valid_sets: list[ValidationDataset],
    trainer: Trainer,
    tracker: utils.ExperimentTracker,
    tp_group: Optional[dist.ProcessGroup] = None,
):
    logger.info(f'training for max {cfg.max_steps} steps')
    warmup_time = 0   # can include data loading and pre-processing time
    with trainer.metrics.timer('load_wall'):
        train_iterator = None if train_set is None else train_set.endless_iterator()
        train_iterator = utils.distributed_batch_iterator(train_iterator, tp_group, cfg.tp_rank)
    patience = cfg.patience
    keep_last = cfg.keep_last if cfg.keep_last > 0 else None   # always keep at least one last checkpoint
    # (a zero or negative value means keep all)

    while trainer.steps < cfg.max_steps:
        # Train step
        # ==========
        prev_step = trainer.steps

        if trainer.steps % cfg.valid_interval == 0 and trainer.last_valid < trainer.steps:
            logger.warning(
                'evaluation metrics are out of date, training was probably interrupted before validation could finish: '
                'validating before training'
            )
            pass  # skip training and saving and go straight to validation
        else:
            with trainer.metrics.timer('wall'):
                trainer.train_step(train_iterator)

            if not trainer.status.running():   # training failed or interrupted
                return

            if trainer.steps == prev_step:  # overflow
                continue

            # Logging
            # =======
            if trainer.steps % cfg.log_interval == 0:
                log(trainer.steps, trainer.metrics, tracker, expected_scores=cfg.expected_scores)
                tracker.log_step(trainer.steps)
            
            if trainer.steps == 1:
                warmup_time = trainer.metrics.sum('total_wall')
            
            # Saving checkpoints and validation
            # =================================
            if trainer.steps % cfg.save_interval != 0:
                utils.benchmark.reset()   # resets peak memory statistics
                continue

            # Saving checkpoints
            # ==================
            # This is done before validation, which can cause errors or freezes
            # However, the checkpoint's metrics will be slightly out of date and will need to be updated after 
            # validation
            logger.info('saving last checkpoint')
            trainer.save()
            trainer.checkpoint_symlink('model_last')
        
        prev_checkpoints = trainer.previous_checkpoints()  # dictionary of {step: path}, ordered by step
        last_checkpoints = list(prev_checkpoints.values())
        if keep_last is not None:  # if None, keep all checkpoints
            last_checkpoints = last_checkpoints[:keep_last - 1]
        average_checkpoints = cfg.average_checkpoints and cfg.keep_last > 1 and len(prev_checkpoints) >= 1

        # Validation
        # ==========
        if trainer.steps % cfg.valid_interval == 0:
            if average_checkpoints:
                # Store the current model in memory to restore it after evaluation
                state_dict = trainer.model_state_dict()
                # Load the last checkpoints and average them before evaluation
                trainer.average_checkpoints(*last_checkpoints, include_self=True)
            
            with utils.benchmark.pause():
                avg_score = evaluate(cfg, task, valid_sets, trainer, tracker, tp_group=tp_group)  # FIXME: CTRL+C
                # during evaluation does not work

            lower_is_better = evaluation.lower_is_better(cfg.early_stopping_metric)

            # update trainer's best score and patience
            trainer.last_valid = trainer.steps
            new_best = False
            if (trainer.best_score is None or
                lower_is_better and avg_score < trainer.best_score or
                not lower_is_better and avg_score > trainer.best_score):
                
                new_best = True
                trainer.best_score = avg_score

                trainer.metrics.update('best_score', trainer.best_score)
                logger.info(f'new best score: {trainer.best_score:.2f} @{trainer.steps}')
                trainer.patience = cfg.patience

            elif trainer.patience and trainer.steps >= cfg.patience_min_steps:
                trainer.patience -= 1
                logger.info(f'patience: {trainer.patience}')

            trainer.save_metrics()  # update metrics with the new best score

            if average_checkpoints:
                if new_best:
                    logger.info('saving averaged checkpoint')
                    trainer.save_model('model_best')
                    # restore the model's state
                    trainer.load_state_dict(state_dict, load_train_state=False)
            elif new_best and utils.is_master(cfg):
                trainer.checkpoint_symlink('model_best')
                # restore the last checkpoint state
        
        utils.barrier(cfg)  # wait for every rank to finish writing its checkpoints before deleting any checkpoint
        # remove old checkpoints: keep last checkpoints and those whose step number is a multiple of 
        # --keep-interval
        trainer.delete_checkpoint('model_latest')
        for step, ckpt_name in prev_checkpoints.items():
            keep_model = cfg.keep_interval and step % cfg.keep_interval == 0 or ckpt_name in last_checkpoints
            trainer.delete_checkpoint(ckpt_name, keep_model=keep_model)

        utils.benchmark.reset()   # resets peak memory statistics

        if patience == 0:
            logger.info('ran out of patience')
            break
    
    if trainer.status.running():
        trainer.status.finish()
        total_time = max(1, trainer.metrics.sum('total_wall'))
        total_lines = trainer.metrics.sum('num_lines')
        logger.info(f'finished training in {total_time:.1f} seconds | warmup {warmup_time:.1f} seconds | '
                    f'lines per second {total_lines / total_time:.1f} ('
                    f'w/o warmup: {total_lines / (total_time - warmup_time):.1f})')


def evaluate(
    cfg: TrainingConfig,
    task: Task,
    valid_sets: list[ValidationDataset],
    trainer: Trainer,
    tracker: Optional[utils.ExperimentTracker] = None,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> float:
    task.eval()  # in case `train_set` uses the same preprocessors and sets them to training mode
    
    scores = []
    for valid_set in valid_sets:
        corpus = valid_set.corpus
        logger.info(f'{corpus} | starting evaluation')
        
        valid_metrics = utils.Metrics(history_size=-1)
        references = valid_set.references
        should_decode = references and cfg.metrics  # only decode if there are references to evaluate against and 
        # evaluation metrics
        hypotheses = []

        with valid_metrics.timer('wall'):
            valid_iterator = iter(valid_set) if cfg.tp_rank == 0 else None
            valid_iterator = utils.distributed_batch_iterator(valid_iterator, tp_group, cfg.tp_rank)
            for batch in valid_iterator:
                valid_logs = trainer.valid_step(batch)
                valid_metrics.increment('steps')
                for k, v in valid_logs.items():
                    valid_metrics.update(k, v)

                if should_decode:
                    hypotheses += trainer.inference_step(
                        batch,
                        # TODO: allow any DecodingConfig option
                        max_output_len=cfg.max_output_len,
                        beam_size=cfg.beam_size,
                        return_scores=cfg.verbose,
                    )

        hypotheses.sort(key=lambda hyp: hyp['idx'])  # sort hypotheses by their original index in the valid set

        main_score = valid_metrics.rolling_divide('nll_loss', 'num_tokens')
        if utils.is_master(cfg) and hypotheses:
            for hyp in hypotheses[:5]:
                for line in task.hypothesis_to_str(hyp, verbose=cfg.verbose).split('\n'):
                    logger.info(f"{corpus} | example output: {line}")

            output_dir = os.path.join(cfg.model_dir, 'outputs')
            os.makedirs(output_dir, exist_ok=True)
            output_path = f'{corpus}.out' if trainer.steps == 0 else f'{corpus}.{trainer.steps}.out'
            output_path = os.path.join(output_dir, output_path)
            with open(output_path, 'w') as output_file:
                output_file.writelines(f'{task.hypothesis_to_str(hyp)}\n' for hyp in hypotheses)

        for metric in cfg.metrics:
            if not hypotheses:
                continue
            
            if utils.is_master(cfg):  # let the master compute the metric (some metrics can be slow) and broadcast to
                # the other ranks, this also avoid inconsistencies across ranks
                score = task.compute_score(
                    metric,
                    hypotheses,
                    references,
                    # TODO: allow any EvalConfig option
                    merge_bpe=cfg.merge_bpe,
                    bleu_tok=cfg.bleu_tok,
                    eval_lc=cfg.eval_lc,
                )
            else:
                score = 0.0
            score = utils.broadcast(cfg, score)

            valid_metrics.update(metric, score)
            if cfg.early_stopping_metric == metric:
                main_score = score

        if corpus.early_stopping:
            scores.append(main_score)

        log(trainer.steps, valid_metrics, tracker, prefix=str(corpus), expected_scores=cfg.expected_scores)
    
    return sum(scores, 0) / max(len(scores), 1)


def log(
    steps: int,
    metrics: utils.Metrics,
    tracker: utils.ExperimentTracker,
    prefix: Optional[str] = None,
    expected_scores: Optional[list[dict]] = None,
):  # TODO: move this to Task
    # steps can be different from metrics.steps
    loss = metrics.rolling_divide('loss', 'num_tokens')
    prompt_nll_loss = metrics.rolling_divide('prompt_nll_loss', 'num_prompt_tokens')
    num_tokens = metrics.rolling_sum('num_tokens') - metrics.rolling_sum('num_prompt_tokens')
    nll_loss = metrics.rolling_sum('nll_loss') / max(1, num_tokens)

    try:
        ppl = 2 ** nll_loss
    except OverflowError:
        ppl = float('inf')

    metrics_ = [
        ('{}', prefix),
        ('steps {}', steps),
        ('loss {:.3f}', loss),
        ('nll_loss {:.3f}', nll_loss),
        ('prompt_nll_loss {:.3f}', prompt_nll_loss),
        ('lb_loss {:.4f}', metrics.rolling_divide('lb_loss', 'num_tokens')),
        ('capacity {:.0f}', metrics.rolling_divide('capacity', 'steps')),
        ('ppl {:.2f}', ppl),
        ('lines {:.4g}', metrics.sum('num_lines')),
        ('tokens {:.4g}', metrics.sum('num_tokens')),
        ('bleu {:.2f}', metrics.val('bleu')),
        ('spbleu {:.2f}', metrics.val('spbleu')),
        ('len_ratio {:.3f}', metrics.val('len_ratio')),
        ('chrf {:.2f}', metrics.val('chrf')),
        ('chrf++ {:.2f}', metrics.val('chrf++')),
        ('langid {:.1%}', metrics.val('langid')),
        ('wps {:.2f}', metrics.rolling_divide('num_tokens', 'wall')),
        ('ups {:.2f}', metrics.rolling_divide('steps', 'wall')),
        ('lps {:.2f}', metrics.rolling_divide('num_lines', 'wall')),
        ('wpb {:.2f}', metrics.rolling_divide('num_tokens', 'steps')),
        ('bsz {:.2f}', metrics.rolling_divide('num_lines', 'steps')),
        ('lr {:.4e}', metrics.val('lr')),
        ('loss_scale {}', metrics.val('loss_scale')),
        ('gnorm {:.3f}', metrics.rolling_divide('gnorm', 'steps')),
    ]

    for name in metrics.names:
        if 'mem' in name:
            val = metrics.min(name) if 'mem_left' in name else metrics.max(name)
            # FIXME: it seems this can be None
            metrics_.append((f'{name} {{:.1f}}', val))
        elif 'wall' in name:
            metrics_.append((f'{name} {{:.1f}}', metrics.sum(name)))

    scores = {fmt.split()[0]: v for fmt, v in metrics_[2:]}
    if tracker is not None:
        data = {(f'{prefix}/{k}' if prefix else k): v for k, v in scores.items() if v}
        tracker.log(data, step=steps)

    logger.info(' | '.join(fmt.format(v) for fmt, v in metrics_ if v))

    if expected_scores:
        check_scores(prefix, scores, expected_scores, steps)


def check_scores(
    corpus: str,
    scores: dict[str, float],
    expected_scores: list[dict],
    steps: int,
    epsilon: float = 0.01,
):
    """
    Verify that all scores are equal or better than expected scores and stops training if that is not the case.
    This is used to do regression tests and check that a given training configuration performs as expected.

    `expected_scores` is a list of dicts defining scores for one or several metrics for a given corpus at a given
    training step.
    
    For example (as defined in the YAML config file):

    expected_scores:
      - corpus: train
        steps: 1000
        metrics:
            nll_loss: 6.686
      - corpus: valid.de-en
        steps: 2000
        metrics:
            chrf: 33.95
      - corpus: valid.de-en
        steps: 5000
        metrics:
            chrf: 52.58
    """
    corpus = corpus or 'train'

    for eval_info in expected_scores:
        if eval_info.get('steps') != steps or eval_info.get('corpus') != corpus:
            continue
        for metric_name, expected_score in eval_info['metrics'].items():
            score = scores[metric_name]
            delta = (
                (expected_score - score) if evaluation.lower_is_better(metric_name) else
                (score - expected_score)
            )
            summary = (
                f'corpus {corpus} | '
                f'steps {steps} | '
                f'metric {metric_name} | '
                f'expected {expected_score:.2f} | '
                f'result {score:.2f}'
            )
            if delta + epsilon >= 0:
                logger.info(f'test passed | {summary}')
            else:
                logger.error(f'test failed | {summary}')
                raise Exception('Test failed, worse performance than expected')


def main():
    # Parsing options
    # ===============
    cfg = TrainingConfig()

    # Save full configuration (combination of command line arguments and config file)
    # FIXME: there may be a race condition when doing multi-node training
    config_file = os.path.join(cfg.model_dir, 'training.yaml')
    # This is done here rather than in main, because main can modify args, and we might get a different result
    # when restarting training with these modified args instead of the original ones
    os.makedirs(cfg.model_dir, exist_ok=True)
    with open(config_file, 'w') as file:
        yaml.safe_dump(cfg.as_dict(), file)

    # Setting distributed training options
    # ============================
    start_device, node_size = utils.setup_distributed(cfg)
    
    # Launching processes
    # ===================
    mp.set_start_method("spawn")
    status = MultiprocessingStatus()
    handler = SignalHandler(status)
    ign_handler = IgnoreHandler()
    
    while True:
        queues = [mp.Queue(maxsize=1024) for _ in range(cfg.dp_local_size)]
        handler.restart = False
        status.run()

        processes = []
        if node_size == 1:
            assert cfg.dp_local_size == 1
            cfg.device = f'cuda:{start_device}'
            cfg.dp_local_rank = 0
            # start_rank is the actual rank
            cfg.dp_rank = cfg.start_rank // cfg.tp_size
            cfg.tp_rank = cfg.start_rank % cfg.tp_size
            run_dist_training(cfg, status, queues, handler)
        else:
            # for rank in range(node_size):
            for rank in reversed(range(node_size)):
                cfg.device = f'cuda:{start_device + rank}'

                if cfg.dp_local_size > 1:
                    # ranks on a given node are part of the same DP group
                    assert cfg.dp_local_size == node_size
                    assert cfg.tp_size == 1
                    cfg.dp_local_rank = rank
                    cfg.dp_rank = cfg.start_rank + rank
                    cfg.tp_rank = 0
                else:  # ranks on a given node are part of the same TP group
                    assert cfg.tp_size == node_size
                    cfg.dp_local_rank = 0
                    cfg.dp_rank = cfg.start_rank // cfg.tp_size
                    cfg.tp_rank = rank
                
                if cfg.debug and rank == 0:  # this will allow breakpoints in the first rank, but this may have adverse 
                    # effects on training speed
                    run_dist_training(cfg, status, queues, handler)
                else:
                    p = mp.Process(target=run_dist_training, args=(cfg, status, queues, ign_handler))
                    p.start()
                    processes.append(p)

            # Ignore SIGTERM and SIGINT and set "status" to INTERRUPTED to inform all the GPUs that they should do their
            # last update and synchronize.
            # We don't do that earlier because the datasets spawn their own processes that shouldn't ignore SIGTERM.
            handler.set_signals()
            
            while status.running():
                time.sleep(1)

            if status.failed():
                print(f'Killing workers', file=sys.stderr)
                time.sleep(10)   # leave some time for them to exit gracefully
                for p in processes:
                    p.kill()   # FIXME: this won't kill grandchildren (and those won't be killed automatically because
                    # SIGTERM is ignored)
            else:
                print(f'Waiting for workers to finish', file=sys.stderr)
                for p in processes:
                    p.join()

        if status.interrupted() and handler.restart:
            print(f'Restarting in 60 seconds...', file=sys.stderr)
            time.sleep(60)
            # Leave some time to perform actions (e.g., delete checkpoints, modify config file)
            with open(config_file) as file:
                # Overwrite args with the options in MODEL_DIR/training.yaml. The user can modify this file before 
                # restarting to modify the training options.
                config = yaml.safe_load(file)
                config = {k: v for k, v in config.items() if not k.startswith('distributed_')}  # do not let the user
                # change distributed training parameters
                cfg.__dict__.update(config)
        elif status.interrupted():
            sys.exit(143)   # exit code that is normally issued when being terminated by SIGTERM 
            # (this will tell SLURM that the job should be requeued)
        elif status.failed():
            sys.exit(1)
        else:
            break


if __name__ == '__main__':
    main()
