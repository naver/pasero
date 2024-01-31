#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import os
import sys
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional

logger = logging.getLogger('decode')

from pasero import utils, decoding
from pasero.config import get_task_class, DecodingCLIConfig


def log(
    metrics: utils.Metrics,
    prefix: Optional[str] = None,
):
    metrics_ = [
        ('{}', prefix),
        ('bleu {:.2f}', metrics.avg('bleu')),
        ('spbleu {:.2f}', metrics.avg('spbleu')),
        ('len_ratio {:.3f}', metrics.avg('len_ratio')),
        ('chrf {:.2f}', metrics.avg('chrf')),
        ('chrf++ {:.2f}', metrics.avg('chrf++')),
        ('langid {:.1%}', metrics.avg('langid')),
        ('loss {:.3f}', metrics.divide('loss', 'num_tokens')),
        ('wps {:.2f}', metrics.divide('num_tokens', 'wall')),
        ('true_wps {:.2f}', metrics.divide('num_words', 'wall')),
        ('ups {:.2f}', metrics.divide('steps', 'wall')),
        ('wpb {:.2f}', metrics.divide('num_tokens', 'steps')),
        ('bsz {:.2f}', metrics.divide('num_lines', 'steps')),
    ]
    for name in metrics.names:
        if 'mem' in name:
            metrics_.append((f'{name} {{:.1f}}', metrics.max(name)))
        elif 'wall' in name:
            metrics_.append((f'{name} {{:.1f}}', metrics.sum(name)))

    logger.info(' | '.join(fmt.format(v) for fmt, v in metrics_ if v))


def run_decoding(cfg: DecodingCLIConfig):
    if utils.is_distributed(cfg):
        dist.init_process_group(
            'nccl',
            rank=cfg.distributed_rank,
            world_size=cfg.distributed_world_size,
            init_method=cfg.distributed_init_method,
        )
    assert not cfg.encoder_decoder_swapping or len(cfg.devices) == 1

    if cfg.teacher_forcing:
        assert cfg.reference or cfg.eval_corpus

    log_file = os.path.join(cfg.model_dir, cfg.log_file) if cfg.log_file and utils.is_master(cfg) else None
    level = logging.DEBUG if cfg.verbose and not cfg.quiet else logging.INFO
    stream = sys.stdout if utils.is_master(cfg) else None
    utils.init_logging(log_file, level=level, stream=stream)

    def get_device(name=None):
        if name is None:
            name = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif name.isnumeric():
            name = f'cuda:{name}'
        return name

    cfg.devices = cfg.devices or [None]
    cfg.devices = [get_device(device) for device in cfg.devices]
    if cfg.devices[0].startswith('cuda:'):
        torch.cuda.set_device(cfg.devices[0])

    logger.info(' '.join(sys.argv))  # log command line parameters
    sys_info = utils.get_sys_info() or {}

    for k, v in sys_info.items():
        logger.info(f'{k} {v}')

    task_cls = get_task_class(cfg.task)

    # Replace placeholders in input, output and reference files with the correct languages, infer languages 
    # from filenames if needed, and make all lists the same length
    inference_corpora = task_cls.get_inference_corpora(
        cfg.task_cfg,
        input_paths=cfg.input,
        output_paths=cfg.output,
        ref_paths=cfg.reference,
        corpus_prefix=cfg.eval_corpus,
    )  # also updates task_cfg with inferred languages

    generator = decoding.TextGenerator(cfg, start=False)
    generator.sync_seed()
    # generator.task.register_corpora(inference_corpora)  # this shouldn't be needed since langs are updated by
    # `get_inference_corpora`
    
    avg_metrics = utils.Metrics(history_size=-1)

    for corpus in inference_corpora:
        if corpus.input_path is not None and not os.path.isfile(corpus.input_path):
            logger.error(f"input file '{corpus.input_path}' does not exist: skipping")
            continue
        
        if corpus.ref_path is not None and not os.path.isfile(corpus.ref_path):
            logger.error(f"reference file '{corpus.ref_path}' does not exist: skipping")
            continue

        if cfg.quiet and corpus.output_path is None:
            corpus.output_path = False

        generator.decode_corpus(corpus, buffer_size=cfg.buffer_size, bleu_tok=cfg.bleu_tok,
                                eval_lc=cfg.eval_lc, continue_=cfg.continue_, verbose=cfg.verbose,
                                metrics=cfg.metrics, max_lines=cfg.max_lines,
                                teacher_forcing=cfg.teacher_forcing)

        prefix = corpus.corpus_id
        for k, v in utils.benchmark.metrics.items():
            generator.metrics.update(k, v)
        log(generator.metrics, prefix=prefix)
        avg_metrics += generator.metrics
        generator.metrics.reset()
        utils.benchmark.reset()

    if len(inference_corpora) > 1:
        log(avg_metrics, prefix='average')


def main():
    cfg = DecodingCLIConfig()
    
    start_device, node_size = utils.setup_distributed(cfg)

    assert cfg.tp_size == 1 or cfg.dp_size == 1, 'combined data and tensor parallelism is not supported at inference'
    mp.set_start_method("spawn")

    def set_rank(rank):
        # set tp_rank or dp_rank depending on whether we're doing tensor or data parallelism
        if cfg.tp_size > 1:
            cfg.tp_rank = rank
            cfg.dp_rank = 0
        else:
            cfg.tp_rank = 0
            cfg.dp_rank = rank

    if node_size == 1:
        set_rank(cfg.start_rank)
        if cfg.devices is None:
            cfg.devices = [f'cuda:{start_device}' if torch.cuda.is_available() else 'cpu']
        run_decoding(cfg)
    else:
        for rank in reversed(range(node_size)):
            set_rank(cfg.start_rank + rank)
            cfg.devices = [f'cuda:{start_device + rank}']

            if rank == 0:  # run first rank in main process to allow breakpoints
                run_decoding(cfg)
            else:
                p = mp.Process(target=run_decoding, args=(cfg,))
                p.daemon = True  # other ranks will be automatically killed if the first rank is killed
                p.start()


if __name__ == '__main__':
    main()