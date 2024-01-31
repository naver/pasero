# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import os
import logging
import traceback
import functools
import regex
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from typing import Iterator, Optional, Any
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from pasero import utils, decoding, optimization, datasets
from pasero.models.transformer import Transformer
from pasero.models import modules
from pasero.tasks import Task
from pasero.config import TrainingConfig


@contextmanager
def autocast(device, enabled, dtype):
    if not enabled:  # torch.autocast fails with device='cpu', even with enabled=False...
        yield
        return
    with torch.autocast(device, dtype=dtype, enabled=enabled):
        yield


logger = logging.getLogger('train')


class Status:
    RUNNING = 1
    FINISHED = 2
    INTERRUPTED = 3
    FAILED = 4
    
    def __init__(self, value=RUNNING):
        self.value = value

    def __iadd__(self, other):
        """ Update this status with another status (usage: `status += other_status`)
        The 'worst' of both status is kept. For instance, if self is FINISHED and other is FAILED,
        self becomes FAILED.
        Using addition for compatibility with `utils.gather_dict`
        """
        self.value = max(self.value, 0 if other is None else other.value)
        return self
    
    def __eq__(self, other):
        return self.value == other.value

    def running(self):
        return self.value == Status.RUNNING
    def finished(self):
        return self.value == Status.FINISHED
    def interrupted(self):
        return self.value == Status.INTERRUPTED
    def failed(self):
        return self.value == Status.FAILED

    def run(self):
        self.value = Status.RUNNING
    def finish(self):
        self.value = Status.FINISHED
    def interrupt(self):
        self.value = Status.INTERRUPTED
    def fail(self):
        self.value = Status.FAILED



class MultiprocessingStatus(Status):
    """ Version of `Status` that can be shared across Python processes """
    def __init__(self):
        self._value = mp.Value('i')
        self.run()

    @property
    def value(self) -> int:
        return self._value.value

    @value.setter
    def value(self, value):
        self._value.value = value


class Trainer:
    def __init__(
        self,
        cfg: TrainingConfig,
        task: Task,
        model: Transformer,
        metrics: utils.Metrics,
        status_mp: Optional[MultiprocessingStatus] = None,
    ):
        self.cfg = cfg
        self.task = task
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.amp = bool(cfg.amp)
            self.dtype = getattr(torch, cfg.dtype)
            if self.amp:
                assert self.dtype == torch.float16
        else:
            logger.warning('No GPU is available, training on CPU')
            self.device = 'cpu'
            self.amp = False
            self.dtype = torch.float
        
        self.fsdp = cfg.fsdp and cfg.dp_size > 1

        dp_size = cfg.dp_size if cfg.tp_size == 1 else cfg.tp_size if cfg.sequence_parallel else 1  # tensor 
        # parallelism with --sequence-parallel simulates data parallelism by multiplying the batch size by --tp-size
        # (in `TrainingDataset`)
        if cfg.virtual_dp_size % dp_size != 0:
            logger.warning(
                'for better reproducibility (constant batch sizes across runs) --virtual-dp-size should be a multiple '
                'of --dp-size and --tp-size')
        self.update_freq = max(1, cfg.virtual_dp_size // dp_size)  # normalize by the number of GPUs to have a 
        # constant total batch size (regardless of --dp-size or --tp-size)

        self.model = model
        self.param_names = []
        self.param_shapes = []

        # the gradients of some parameters shouldn't be synced by DDP, because each GPU holds a different parameter
        # (e.g., experts)
        model.set_ddp_params_and_buffers_to_ignore()

        for name, param in self.model.named_parameters():
            # find parameters that are being trained
            if param.requires_grad:
                self.param_names.append(name)
                self.param_shapes.append(param.shape)
        
        assert self.param_names, 'no parameter is being trained'
        assert not self.fsdp or not model.is_sharded, 'FSDP is incompatible with sharded models (e.g., TP or MoEs)'
        self.sharded = self.fsdp or model.is_sharded  # whether to sync grad norm & to ensure 
        # at decoding that all ranks decode for the same number of steps.
        # Note: this is not the same as `self.model.is_sharded`, which indicates that different ranks have different
        # model parameters that need to be saved in different checkpoints

        if not self.fsdp:
            self.model.to(self.dtype)
        
        # TODO: do this in a separate 'resume' method, which reverts to '--ckpt', or no checkpoint if
        # 'model_latest.bin' is corrupted and 'model_last.bin' doesn't exist
        ckpt_path, found_existing = utils.find_checkpoint_to_load(cfg.model_dir, cfg.ckpt, cfg.reset)
        if found_existing:
            cfg.continue_ = True
        
        load_train_state = cfg.continue_ and not cfg.reset_optimizer
        # this needs to be done before FSDP:
        ckpt = self.read_checkpoint(
            ckpt_path,
            load_train_state=load_train_state,
            remap=not cfg.continue_,
            # we don't want the model to try re-mapping the embeddings again if it already has a 
            # checkpoint with re-mapped embeddings
        ) if ckpt_path else None

        if found_existing and cfg.save_trainable_only and cfg.ckpt:
            # Load the initial model if we're resuming for a training instance that has --save-trainable-only.
            # This is typically done with parameter-efficient tuning to only save the adapters. Since the model's 
            # checkpoints won't contain the other parameters, we need to load the initial model again.
            initial_ckpt = self.read_checkpoint(cfg.ckpt, load_train_state=False, remap=True)
            ckpt['model'] = {**initial_ckpt['model'], **ckpt['model']}

        if self.device != 'cpu' and not self.fsdp:
            # model is automatically moved to the GPU by FSDP, which could let us train larger-than-memory models
            self.model.to(torch.cuda.current_device())
        
        if cfg.tp_size > 1 and cfg.dp_size > 1:
            # This won't work correctly, as the sharded parameters need to be synced across dp ranks.
            # TODO: apply DDP on sharded parameters *only* and manually sync the other gradients for shared parameters 
            raise NotImplementedError

        if not utils.is_distributed(cfg):
            self.ddp_model = self.model
        elif self.fsdp:
            # FIXME: FSDP does not work with lang-specific parameters (e.g., --encoder-adapters-by). Maybe the batches
            # need to be homogeneous across all shards?
            if self.dtype is torch.float16:
                mixed_precision = MixedPrecision(
                    param_dtype=self.dtype,
                    reduce_dtype=self.dtype,
                    buffer_dtype=self.dtype,
                )
            else:
                mixed_precision = None
            from pasero.models.transformer import TransformerEncoderLayer, TransformerDecoderLayer
            from pasero.models.modules import ConvolutionSubsampler, LearnedPositionalEmbedding, AdapterLayer
            from pasero.models.modules import WrappableLayerNorm, WrappableRMSNorm, WrappableLinear
            # All these layers will be wrapped separately. This should work in most settings (e.g., freeze everything 
            # but adapters)
            # TODO: do not shard parameters that are small (e.g., layer norm), using 'ignored_parameters' or
            # 'auto_wrap_policy'
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={
                    AdapterLayer,
                    TransformerEncoderLayer,
                    TransformerDecoderLayer,
                    ConvolutionSubsampler,
                    LearnedPositionalEmbedding,
                    WrappableLayerNorm,
                    WrappableRMSNorm,
                    WrappableLinear,
                },
            )
            self.ddp_model = FSDP(
                self.model,
                mixed_precision=mixed_precision,
                auto_wrap_policy=auto_wrap_policy,
                sync_module_states=True,
                device_id=torch.cuda.current_device(),
                cpu_offload=None,
                backward_prefetch=None,
            )

            logger.debug(self.ddp_model)
            for name, param in self.ddp_model.named_parameters():
                shape = 'x'.join(map(str, param.shape))
                frozen = '' if param.requires_grad else ', frozen'
                logger.debug(f'{name} ({shape}{frozen})')

            self.model = self.ddp_model.module
        else:
            find_unused_parameters = (
                cfg.find_unused_parameters or
                self.model.find_unused_parameters or
                self.task.find_unused_parameters
            )
            assert not find_unused_parameters or not self.cfg.model_cfg.checkpoint_activations, ('DDP does not support '
                'checkpointed models with unused parameters')
            self.ddp_model = DDP(
                self.model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters,
                gradient_as_bucket_view=True,
                process_group=None,
            )
            # FIXME: tp_size > 1 and dp_size > 1 won't work correctly, as the sharded parameters need to be synced
            # across dp ranks.
            # TODO: apply DDP on sharded parameters *only* and manually sync the other gradients for
            # the shared parameters 
        
        self.steps = 0
        self.metrics = metrics
        self._dummy_batch = None
        self._valid_dummy_batch = None
        self._status_mp = status_mp   # MultiprocessingStatus used to communicate with the main process
        self.status = Status()
        self.optimizer = None   # the optimizer should be build after FSDP & after loading the model state dict
        self.scheduler = None
        self.scaler = None

        self.last_valid = 0
        self.patience = cfg.patience
        self.best_score = None
        self.saved = True

        if ckpt is not None:  # finish loading the checkpoint
            self.load_state_dict(ckpt, load_train_state=load_train_state)
            step = ckpt.get('steps', 0)
            msg = f'loaded checkpoint(s) {ckpt_path} @{step}'
            msg += f" (training state {'restored' if load_train_state else 'reset'})"
            logger.info(msg)

        if cfg.tp_size > 1:
            # make sure that all the shared parameters have the same value
            for param in model.parameters():
                if not getattr(param, '_is_sharded', False):
                    dist.broadcast(param.data, 0)

    def build_optimizer(self):
        if self.optimizer is not None:
            return
        self.trainable_params = [param for param in self.ddp_model.parameters() if param.requires_grad]

        self.scaler = optimization.GradScaler(
            init_scale=2**7,
            growth_interval=2**14 // self.update_freq,
            enabled=self.dtype is torch.float16,
        )

        opts = dict(
            lr=self.cfg.lr,
            betas=self.cfg.adam_betas,
            weight_decay=self.cfg.weight_decay,
            optimizer_states_as_fp32=self.cfg.optimizer_states_as_fp32,
        )
        
        if self.dtype == torch.float32 or self.sharded or self.cfg.memory_efficient_fp16:
            # Version of Adam which converts on the fly float16 grads and params to float32 before updating statistics
            # This is slower than MixedPrecisionAdam, but it uses less memory as no float32 copy of the parameters and 
            # gradients needs to be maintained
            self.optimizer = optimization.Adam(self.trainable_params, **opts)
        elif self.cfg.flat_fp16:  # note that this does not work with sharded parameters (e.g., tp_size > 1)
            # All parameters and gradients are flattened into single tensors. This is faster but takes more memory
            self.optimizer = optimization.FlatFP16Adam(self.trainable_params, **opts)
        else:
            # Like Adam, but keeps a float32 copy of the parameters and gradients
            self.optimizer = optimization.MixedPrecisionAdam(self.trainable_params, **opts)

        self.scheduler = optimization.LRScheduler(self.cfg, self.optimizer)
    
    def log_gpu_mem_usage(self, prefix: str) -> None:
        if self.device != 'cpu':
            torch.cuda.empty_cache()
            msg = f'{torch.cuda.memory_allocated() / 2**20:.1f}MiB'
            if prefix:
                msg = f'{prefix} {msg}'
            logger.info(msg)

    def train_step(self, batch_iterator: Iterator[dict]) -> None:
        
        logs = defaultdict(int)
        self.build_optimizer()
        self.ddp_model.train()
        self.optimizer.zero_grad(set_to_none=True)

        batch_id = 0
        while batch_id < self.update_freq:
            with ExitStack() as stack:
                if batch_id == 0 and self.steps == 0:
                    # don't count the load time of the first batch, as this can take
                    # a long time and it would impact WPS
                    stack.enter_context(self.metrics.pause('wall'))
                
                batch = None
                with self.metrics.timer('load_wall'), utils.benchmark('data'):
                    if self.status.running():
                        try:
                            batch = next(batch_iterator)
                        except:
                            traceback.print_exc()
                    if self.cfg.sequence_parallel and batch is not None:
                        batch_size = batch['decoder_input'].size(0)
                        if batch_size % self.cfg.tp_size != 0:  # skip batches that cannot be sharded
                            continue
                        batch = datasets.shard_batch(
                            batch,
                            shard_id=self.cfg.tp_rank,
                            shard_count=self.cfg.tp_size,
                        )

            if self._dummy_batch is None:
                # this is a small batch that shouldn't cause OOMs
                self._dummy_batch = datasets.dummy_batch(batch, size=self.task.cfg.batch_size_multiple)
            
            if batch is None:
                batch = self._dummy_batch  # FIXME: this won't work if pre-processing fails at the very first step
                ignore_grads = True        # FIXME: this causes an error with distributed training:
                # "This error indicates that your module has parameters that were not used in producing loss."
                self.status += self._status_mp            # get the global status of the job
                if not self.status.interrupted():         # maybe training has been interrupted?
                    self.status.fail()                    # no interruption: it's a data loading error
            else:
                ignore_grads = False

            if batch is None:
                logger.error('no dummy batch found, cannot synchronize workers: you might have to stop them manually')
                break

            try:
                batch = utils.move_to_device(batch, self.device, dtype=self.dtype)
                
                batch['meta']['step'] = self.steps  # used by CLSR
                
                with utils.benchmark('forward'), autocast(self.device, enabled=self.amp, dtype=self.dtype):
                    ### FORWARD
                    loss, batch_logs = self.ddp_model(**batch)
                    # OOM errors typically appear here (end of the forward pass)

                if ignore_grads:
                    loss *= 0
                else:
                    for k, v in batch_logs.items():  # aggregate logs from all batche
                        logs[k] += v
                
                ### BACKWARD
                with utils.benchmark('backward'):
                    if (
                        batch_id == self.update_freq - 1 or
                        self.fsdp or
                        not hasattr(self.ddp_model, 'no_sync') or
                        self.status.failed()   # synchronize and stop
                    ):
                        with utils.benchmark('sync'):  # note that this includes the time of one backward pass,
                            # the actual communication time can be approximated when update_freq > 1:
                            # one_backward = (backward - sync) / (update_freq - 1)
                            # sync -= one_backward
                            self.scaler.scale(loss).backward()
                            # TODO: manually reduce grads for the TP params (which are ignored by DDP)
                            # It may be faster to use DDP for the TP params and CustomDataParallel for the 
                            # shared params (embeddings and layer norms), which are less numerous
                    else:
                        with self.ddp_model.no_sync():
                            self.scaler.scale(loss).backward()
                del loss
                batch_id += 1
            except RuntimeError as e:
                if 'out of memory' in str(e) and not self.status.failed():
                    traceback.print_exc()
                    logger.warning(torch.cuda.memory_summary())
                    # Try finishing this update with dummy_batch to exit gracefully
                    logger.error('attempting to gracefully handle OOM error')
                    torch.cuda.empty_cache()
                    self.status.fail()
                else:
                    # Will likely cause some processes hanging (especially in multi-node training)
                    raise e

        # Done in fairseq to reduce risks of OOM
        if self.steps == 0:
            torch.cuda.empty_cache()

        self.status += self._status_mp      # update with the global status of the job

        logs = {**logs, 'status': self.status}
        # synchronize status across workers and aggregates num_tokens and losses for logging (by summing them)
        logs = utils.gather_dict(self.cfg, logs)
        
        if self.cfg.tp_size > 1 and not self.cfg.sequence_parallel:
            # With tensor parallelism and --no-sequence-parallel, all TP ranks get the same batch, which means 
            # that the token counts and losses are duplicated in the 'gather_dict' line just above; we need to correct 
            # this by normalizing them by TP size.
            # This doesn't happen with --sequence-parallel, where the batch is sharded and TP ranks get a different
            # set of tokens.
            for k, v in logs.items():
                if isinstance(v, int):
                    logs[k] = v // self.cfg.tp_size
                elif isinstance(v, float):
                    logs[k] = v / self.cfg.tp_size

        self.status = logs.pop('status')   # status has been synchronized across all workers (if one is finished, all
        # are finished)
        
        if self.status.failed():  # if training has failed, we don't want to update the model parameters
            return
        # we're not stopping early when status.interrupted() because this could cause corrupted checkpoints with FSDP

        # FIXME: KeyboardInterrupt may arrive and fail here (e.g., when done before starting to train)
        num_tokens = logs['num_tokens']

        with utils.benchmark('optimizer'):
            self.scaler.unscale_(self.optimizer, sharded=self.sharded)  # makes sure we always have the same loss scale
            # across all GPUs
            params = self.optimizer.param_groups[0]['params']
            # divide the gradients by the number of tokens (because the loss was not reduced); and
            # multiply the gradients by the world size to cancel the division done by DDP (because `num_tokens`, which  
            # we normalize with is already proportional to the world size)
            for p in params:
                if p.grad is not None:
                    coeff = self.cfg.dp_size / num_tokens
                    # with tensor paralellism + sequence parallelism, the gradients of the shared parameters shouldn't
                    # be normalized by the total number of tokens, but by the size of the sharded batch. Correct this
                    # by multiplying them by the number of shards (i.e., TP size)
                    if self.cfg.sequence_parallel and not getattr(p, '_is_sharded', False):
                        coeff *= self.cfg.tp_size
                    p.grad.data.mul_(coeff)
            
            gnorm = optimization.clip_grad_norm_(
                params,
                self.cfg.clip_norm,
                sharded=self.sharded,
                shard_id=self.model.shard_id,
            )

            ### OPTIMIZER STEP
            self.scaler.step(self.optimizer)
            old_loss_scale = self.scaler.get_scale()
            self.scaler.update()
            loss_scale = self.scaler.get_scale()

            if loss_scale >= old_loss_scale:  # no overflow
                self.scheduler.step()
                logs['gnorm'] = gnorm.item()
                overflow = False
            elif loss_scale >= 1e-4:
                logger.warning(f'overflow detected, setting loss scale to: {loss_scale}')
                overflow = True
            else:
                logger.error('too many overflows: the loss scaling factor has reached its minimum value, '
                             'try lowering the learning rate')
                self.status.fail()
                overflow = True

        if not overflow:
            self.steps += 1
            self.metrics.increment('steps')
            logs.update({
                'lr': self.scheduler.get_last_lr()[0],
                'loss_scale': self.scaler.get_scale(),
            })
            
            if 'capacity' in logs:  # average expert capacity per batch
                logs['capacity'] /= (self.update_freq * self.cfg.dp_size)
            
            for k, v in logs.items():
                self.metrics.update(k, v)
            
            for k, v in utils.benchmark.metrics.items():
                self.metrics.update(k, v)

            self.metrics.update('cpu_mem', utils.get_cpu_mem_used())
            self.metrics.update('cpu_mem_left', utils.get_cpu_mem_left())
            self.saved = False
        
    @torch.no_grad()
    def valid_step(self, batch: dict) -> dict:
        if self._valid_dummy_batch is None:
            self._valid_dummy_batch = datasets.dummy_batch(batch)
        dummy = False
        if batch is None:  # avoid deadlocks in models that need to communicate by making sure they process the same
            # number of steps
            batch = self._valid_dummy_batch or self._dummy_batch
            dummy = True
        batch = utils.move_to_device(batch, self.device, dtype=self.dtype)
        self.ddp_model.eval()
        
        modules.set_sequence_parallel(False)
        with autocast(self.device, enabled=self.amp, dtype=self.dtype):
            _, logs = self.ddp_model(**batch)        
        modules.set_sequence_parallel(self.cfg.sequence_parallel)
        
        if dummy:
            logs = {}
        logs = utils.gather_dict(self.cfg, logs)
        if self.cfg.tp_size > 1:
            # with tensor parallelism and --no-sequence-parallel, all TP ranks get the same batch, whose loss and token
            # counts shouldn't be reduced by the line above, correct this by normalizing by TP size:
            for k, v in logs.items():
                if isinstance(v, int):
                    logs[k] = v // self.cfg.tp_size
                elif isinstance(v, float):
                    logs[k] = v / self.cfg.tp_size
        return logs

    @torch.no_grad()
    def inference_step(self, batch: dict, **decoding_opts) -> list[dict[str, Any]]:
        if self._valid_dummy_batch is None:
            self._valid_dummy_batch = datasets.dummy_batch(batch)
        dummy = False
        if batch is None:
            batch = self._valid_dummy_batch or self._dummy_batch
            dummy = True
        
        batch = dict(batch)
        batch = utils.move_to_device(batch, self.device, dtype=self.dtype)

        # Because these batches are produced by ValidationDataset and are also used by `valid_step` for computing 
        # validation loss (via teacher forcing), `decoder_input` contains the entire target sequence and not just 
        # prompt tokens. Here, we don't want to do teacher forcing, except for the prompt tokens (e.g., lang codes),
        # so we mask/truncate `decoder_input`
        batch['decoder_input'] = utils.mask_by_len(
            batch['decoder_input'],
            batch['prompt_length'],
            fill_value=self.model.padding_idx,
            truncate=True,
        )

        self.model.eval()
        # temporarily disable sequence parallelism as this won't work with beam search decoding (all TP ranks should
        # always have the same beam size, batch size and sequence length)
        modules.set_sequence_parallel(False)
        with autocast(self.device, enabled=self.amp, dtype=self.dtype):
            encoder_out, encoder_mask, _ = self.model.encoder(**batch)
            output = decoding.search(
                self.model.decoder,
                encoder_out,
                encoder_mask,
                **batch,
                sharded=self.sharded,
                device=self.device,
                **decoding_opts,
            )
            hypotheses = [nbest[0] for nbest in output]  # we only care for the top-ranking hypothesis
            del encoder_out, encoder_mask
        modules.set_sequence_parallel(self.cfg.sequence_parallel)
        if dummy:
            output = []
        
        if self.cfg.tp_rank > 0 or dummy:
            # avoid duplicate outputs with tensor parallelism
            hypotheses = []
        else:
            real_indices = batch['indices']
            meta = batch.pop('meta')

            for i, (real_idx, hyp) in enumerate(zip(real_indices, hypotheses)):
                sample_bin = {
                    k: v[i] for k, v in batch.items()
                }
                sample_bin['meta'] = meta
                self.task.postprocess(sample_bin, hyp, detokenize=True)
                hyp['idx'] = real_idx

            hypotheses = utils.tensor_to_array(hypotheses)
        
        hypotheses = utils.gather_list(self.cfg, hypotheses)
        return hypotheses

    def read_checkpoint(self, path: str, load_train_state: bool = True, remap: bool = False) -> dict:
        """
        Reads given checkpoint and updates it to be compatible with current Pasero version and settings.
        `Trainer.load_state_dict` should be called with the output of this function.
        """
        try:
            main_ckpt = utils.load_and_reshard_checkpoint(
                self.model,
                path,
                model_shard_id=self.model.shard_id,
                model_shard_count=self.model.shard_count,
                load_train_state=load_train_state,
            )

        except RuntimeError as e:
            if 'model_latest' in path:  # this checkpoint is saved automatically when a job is aborted by 
                # SLURM, but sometimes it doesn't have time to complete. In this case, revert to the older 
                # 'model_last.bin'
                # FIXME: This won't work if 'model_last.bin' doesn't exist and the job will just fail...
                new_path = path.replace('model_latest', 'model_last')
                if os.path.exists(new_path):
                    logger.warning(f'{path} is corrupted, trying {new_path}')
                    return self.read_checkpoint(new_path, load_train_state, remap)
            raise e

        self.update_state_dict(main_ckpt, load_train_state=load_train_state, remap=remap)
        return main_ckpt
    
    def average_checkpoints(self, *names: str, include_self: bool = False) -> None:
        """ Loads model checkpoints and averages their weights. Optimizer states and metrics are not loaded. """
        assert names or include_self
        paths = [self.local_checkpoint_path(name) for name in names]
        checkpoints = [utils.move_to_cpu(self.model.state_dict())] if include_self else []
        checkpoints += [self.read_checkpoint(path, load_train_state=False)['model'] for path in paths]
        ckpt = {'model': utils.average_models(checkpoints)}
        self.load_state_dict(ckpt, load_train_state=False)
        logger.info('loaded checkpoint(s) ' + ', '.join(paths))
    
    @property
    def should_save(self) -> bool:
        """ Whether the current process should save a checkpoint or not """
        return (
            self.cfg.dp_rank == 0 or  # tensor parallelism or data parallelism (all TP ranks have the same DP rank
            self.model.is_sharded and self.cfg.tp_size == 1   # MoE (special case where each DP rank has different 
            # parameters)
        )
    
    def local_checkpoint_path(self, prefix: Optional[str] = None) -> Optional[str]:
        """
        Returns the name of the model checkpoint at current step for this rank.
        
        Example:
            cfg.model_dir = 'model', self.steps = 1000, self.tp_rank = 1, self.tp_size = 4 ->
                'model/model_1000_002_of_004.bin'
            cfg.model_dir = 'model', self.tp_rank = 2, self.tp_size = 4, prefix = 'model_latest' ->
                'model/model_latest_003_of_004.bin'
            self.dp_rank = 1 -> None

        The optimizer and metrics checkpoint paths can be obtained by passing the result of this function to
        `utils.optimizer_checkpoint` and `utils.metrics_checkpoint`
        """
        prefix = prefix or f'model_{self.steps}'
        if self.model.shard_count > 1:
            suffix = f'_{self.model.shard_id + 1:03}_of_{self.model.shard_count:03}.bin'
        else:
            suffix = '.bin'
        return os.path.join(self.cfg.model_dir, prefix + suffix)

    def checkpoint_symlink(self, dest: str):
        dest = dest.removesuffix('.bin') + '.bin'  # add .bin suffix if needed
        if utils.is_master(self.cfg):
            src = self.local_checkpoint_path()  # checkpoint at current step and for this shard
            dest = os.path.join(self.cfg.model_dir, dest)
            utils.safe_symlink(os.path.basename(src), dest)

    def previous_checkpoints(self) -> dict[int, str]:
        """
        Find the previous checkpoints available in `model_dir`.
        Returns a dictionary of {checkpoint_steps: checkpoint_prefix}
        """
        checkpoints = {}
        for filename in os.listdir(self.cfg.model_dir):
            match = regex.fullmatch(r'(?<prefix>model_(?<step>\d+))(_001_of_\d{3})?.bin', filename)
            if match:
                step = int(match.group('step'))
                if step < self.steps:
                    checkpoints[step] = match.group('prefix')
        checkpoints = dict(sorted(checkpoints.items(), reverse=True))  # ordered from most recent
        # to oldest
        return checkpoints

    def delete_checkpoint(self, prefix: str, keep_model: bool = False) -> bool:
        """
        Delete all checkpoint shards matching given checkpoint prefix (e.g., "model_latest" or "model_1000").
        If `keep_model` is True, optimizer and metrics checkpoints are deleted but not models.
        """
        if not utils.is_master(self.cfg):  # let the master handle the cleaning
            return

        assert prefix.startswith('model_')
        model_dir = self.cfg.model_dir

        def has_symlink(filename):
            # Check whether there exists a symlink leading to the main shard of this checkpoint (e.g., 'model_best.bin')
            filename = regex.sub(r'_\d{3}_of_(\d{3}).bin$', r'_001_of_\1.bin', filename)
            # model/model_1000_003_of_004.bin -> model/model_1000_001_of_004.bin
            # model/model_1000.bin -> model/model_1000.bin
            for other_name in os.listdir(model_dir):
                other_path = os.path.join(model_dir, other_name)
                if os.path.islink(other_path) and os.readlink(other_path) == filename:
                    return True
            return False

        pattern = regex.compile(regex.escape(prefix) + r'(_\d{3}_of_\d{3})?\.bin')
        for filename in os.listdir(model_dir):
            if regex.fullmatch(pattern, filename) and not has_symlink(filename):  # only delete files that 
                # start with prefix and that don't have a symlink
                path = os.path.join(model_dir, filename)
                if not keep_model:
                    utils.safe_delete(path)
                utils.safe_delete(utils.optimizer_checkpoint(path))
                utils.safe_delete(utils.metrics_checkpoint(path))

    def save_metrics(self, prefix: Optional[str] = None) -> None:
        if not utils.is_master(self.cfg):  # only the master needs to save metrics
            return
        
        path = self.local_checkpoint_path(prefix)
        path = utils.metrics_checkpoint(path)
        if os.path.islink(path):
            os.unlink(path)
        
        state_dict = self.metrics_state_dict()
        torch.save(state_dict, path)
        logger.info(f'saved {path} @{self.steps}')

    def save_optimizer(self, prefix: Optional[str] = None) -> None:
        state_dict = self.opt_state_dict()  # for FSDP, this needs to be called on all ranks
        
        if not self.should_save:  # some ranks do not need to save checkpoints
            return

        path = self.local_checkpoint_path(prefix)
        path = utils.optimizer_checkpoint(path)
        if os.path.islink(path):
            os.unlink(path)
        
        torch.save(state_dict, path)
        logger.info(f'saved {path} @{self.steps}')

    def save_model(self, prefix: Optional[str] = None):
        state_dict = self.model_state_dict()  # for FSDP, this needs to be called on all ranks

        if not self.should_save:
            return
        
        path = self.local_checkpoint_path(prefix)
        if os.path.islink(path):
            os.unlink(path)
        
        torch.save(state_dict, path)
        logger.info(f'saved {path} @{self.steps}')

    def save(self, prefix: Optional[str] = None) -> None:
        self.save_model(prefix)
        self.save_metrics(prefix)
        self.save_optimizer(prefix)
        self.saved = True

    def update_state_dict(self, state_dict: dict, load_train_state: bool = True, remap: bool = False) -> None:
        """
        remap: apply one-shot operations like shifting layers or re-mapping embeddings, which
        only make sense when finetuning a different model. This shouldn't be done again when resuming training or when 
        decoding.
        """
        param_sizes = {      # used to re-map the fairseq optimizer parameters
            k: v.numel() for k, v in state_dict['model'].items()
            if not k.endswith('.version') and not k.endswith('._float_tensor')
        }  # do that before model.update_state_dict
        if self.cfg.model_cfg.shared_embeddings:
            param_sizes.pop('decoder.embed_tokens.weight', None)
        if self.cfg.model_cfg.tied_output_projection:
            param_sizes.pop('decoder.output_projection.weight', None)

        self.model.train()  # set to training mode before calling update_state_dict(...)
        self.model.update_state_dict(state_dict['model'])
        if remap:
            self.model.remap_state_dict(state_dict['model'])
            if self.cfg.reset_params_regex:
                for key in list(state_dict['model']):
                    if regex.match(self.cfg.reset_params_regex, key):
                        state_dict['model'].pop(key)

        if load_train_state:
            if state_dict.get('last_optimizer_state') and not state_dict.get('optimizer'):   # fairseq-style
                state_dict['optimizer'] = optimization.convert_fairseq_state_dict(
                    state_dict['last_optimizer_state'], param_sizes
                )
                state_dict.pop('last_optimizer_state')
                # note that we still do this if remap is False, as this is operation can be done on
                # Pasero checkpoints without risks; and remap is set to False when --continue is set.

            if state_dict.get('optimizer'):
                optimization.update_state_dict(state_dict['optimizer'], self.param_names, self.param_shapes)

            if state_dict.get('steps') is None and state_dict.get('optimizer_history'):      # fairseq-style
                state_dict['steps'] = state_dict['optimizer_history'][-1]['num_updates']

    def load_state_dict(self, state_dict: dict, load_train_state: bool = True) -> None:
        """
        This should be called before FSDP
        This assumes that self.update_state_dict(state_dict) has already been called
        """
        if self.fsdp:
            self.ddp_model.load_state_dict(state_dict['model'])
        else:  # the option --flexible does not work with FSDP, whose load_state_dict has not "strict" argument
            self.model.load_state_dict(state_dict['model'], strict=not self.cfg.flexible)
            # FIXME: this won't work if parameters exist but have the wrong shape

        if load_train_state:
            self.optimizer = None
            self.build_optimizer()   # it is important for MixedPrecisionAdam that the optimizer is built after
            # loading the model state dict 
            self.steps = state_dict.get('steps')
            optim_state = state_dict.get('optimizer')
            if optim_state:
                if self.fsdp:
                    optim_state = FSDP.shard_full_optim_state_dict(
                        optim_state,
                        self.ddp_model,
                    )
                    # FIXME: training often explodes when resuming training (with --dtype float16 or --amp)
                    # It seems FSPD sometimes saves somewhat broken checkpoints that cannot be finetuned
                    # (either with or without --fsdp). This does not happen with --reset-optimizer
                else:
                    # remap parameter names to parameter indices
                    optim_state['state'] = dict(enumerate(optim_state['state'].values()))
                    optim_state['param_groups'][0]['params'] = list(optim_state['state'])

            if optim_state and optim_state['state']:
                self.optimizer.load_state_dict(optim_state)
                scheduler_step = next(iter(optim_state['state'].values())).get('step')
            else:
                logger.warning('no optimizer state was found in the checkpoint')
                scheduler_step = None

            if state_dict.get('scaler'):
                self.scaler.load_state_dict(state_dict['scaler'])
            if state_dict.get('metrics'):
                self.metrics.load_state_dict(state_dict['metrics'])
                if self.steps is None:  # for older checkpoints that only saved the step count in metrics
                    self.steps = self.metrics.sum('steps')
                if self.best_score is None:  # same, 'best_score' used to be stored in metrics
                    self.best_score = self.metrics.val('best_score')
            
            self.steps = self.steps or 0    # self.steps shouldn't be None
            self.last_valid = state_dict.get('last_valid', self.steps)
            self.best_score = state_dict.get('best_score')
            self.patience = state_dict.get('patience') or self.cfg.patience

            if state_dict.get('scheduler'):
                self.scheduler.load_state_dict(state_dict['scheduler'])
            elif scheduler_step is not None:   # fairseq-style
                self.scheduler.load_state_dict({'last_epoch': scheduler_step - 1, '_step_count': scheduler_step})

    def metrics_state_dict(self) -> dict:
        ckpt = {
            'args': self.cfg.as_dict(),
            'metrics': self.metrics.state_dict(),
            'last_valid': self.last_valid,  # used when resuming training to know whether the evaluation metrics 
            # are up to date, or whether the model should be evaluated again (e.g., in case training was interrupted
            # in the middle of evaluation)
            'best_score': self.best_score,
            'patience': self.patience,
            'steps': self.steps,
        }
        return utils.move_to_cpu(ckpt)

    def opt_state_dict(self) -> dict:
        if self.optimizer is None:
            optim_state = None
        elif self.fsdp:
            # FIXME: deprecation warning
            optim_state = FSDP.full_optim_state_dict(self.ddp_model, self.optimizer)
            # automatically unflattens the optimizer parameters & uses their names as keys (instead of numerical ids)
        else:
            optim_state = self.optimizer.state_dict()
            # convert the numerical parameter keys to real parameter names in the checkpoint
            # this is more robust to changes in parameter order and to missing parameters due to parameter freezing
            try:
                optim_state['state'] = dict(zip(self.param_names, optim_state['state'].values()))
                optim_state['param_groups'][0]['params'] = list(optim_state['state'])
            except KeyError:  # this can happen when no update has been done yet
                optim_state.pop('state', None)
        
        ckpt = {
            'args': self.cfg.as_dict(),
            'optimizer': optim_state,
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'scaler': self.scaler.state_dict() if self.scaler is not None else None,
            'steps': self.steps,
        }
        ckpt = {k: v for k, v in ckpt.items() if v is not None}
        return utils.move_to_cpu(ckpt)

    def model_state_dict(self) -> dict:
        if self.fsdp:
            state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.ddp_model, StateDictType.FULL_STATE_DICT, state_dict_config):
                model_state_dict = self.ddp_model.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        self.model.clean_state_dict(model_state_dict)  # remove weights that are redundant
        if self.cfg.save_trainable_only:  # only keep the parameters that are being trained
            model_state_dict = {k: v for k, v in model_state_dict.items() if k in self.param_names}
        
        ckpt = {
            'args': self.cfg.as_dict(),   # will be used at inference to initialize the model with the right 
            # architecture and hyper-parameters
            'model': model_state_dict,
            'steps': self.steps,
        }
        # args and steps are saved in every checkpoint (model, optimizer and metrics) for reproducibility (in case 
        # some checkpoints go missing or are renamed)
        ckpt = {k: v for k, v in ckpt.items() if v is not None}
        return utils.move_to_cpu(ckpt)
