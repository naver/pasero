# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import torch
import logging
import itertools
import math
import torch.distributed as dist
from torch import Tensor
from pasero.config import TrainingConfig


logger = logging.getLogger('optimizer')


# Just for documentation purposes:
Float16Parameter = torch.nn.Parameter
Float32Parameter = torch.nn.Parameter


class LRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        cfg: TrainingConfig,
        optimizer: torch.optim.Optimizer,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup = cfg.warmup

        assert cfg.init_lr >= 0 and cfg.warmup >= 0 and cfg.init_lr <= cfg.lr and cfg.max_steps > 0

        self.max_steps = cfg.max_steps
        self.max_lr = cfg.lr
        self.min_lr = cfg.min_lr
        self.init_lr = cfg.init_lr

        super(LRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        step = self.last_epoch

        if self.warmup == 0:  # linear decay (no warmup): max_lr -> min_lr (in 'max_steps' steps)
            lr = self.max_lr - step * (self.max_lr - self.min_lr) / self.max_steps
        elif step < self.warmup:  # linear warmup phase: init_lr -> max_lr (in 'warmup' steps)
            lr = self.init_lr + step * (self.max_lr - self.init_lr) / self.warmup
        else:  # warmup phase finished, inverse square root decay
            lr = self.max_lr * math.sqrt(self.warmup / step)
        
        lr = max(lr, self.min_lr)

        return [lr] * len(self.optimizer.param_groups)



class Adam(torch.optim.AdamW):
    """
    Version of Adam copied from fairseq, which automatically converts float16 tensors to float32
    before updating its statistics (AKA --memory-efficient-fp16)
    """
    @torch.no_grad()
    def step(self, closure=None):
        """ Copied from fairseq/optim/adam.py """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group.get("amsgrad", False)

                p_data_fp32 = p.data
                if p.data.dtype is torch.float16:
                    p_data_fp32 = p_data_fp32.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].to(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data_fp32)
                    if amsgrad:
                        state["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(
                            p_data_fp32
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                step_size = group["lr"] / bias_correction1
                
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                    )

                p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss

    def load_state_dict(self, state_dict):
        """ Copied from fairseq: prevent torch.optim.AdamW from casting its states to float16 """
        super().load_state_dict(state_dict)
        id_map = {
            old_id: p
            for old_id, p in zip(
                itertools.chain(*(g["params"] for g in state_dict["param_groups"])),
                itertools.chain(*(g["params"] for g in self.param_groups)),
            )
        }
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                self.state[param] = v


class FP16Adam(torch.optim.AdamW):
    """
    Partially copied from fairseq.
    
    After the backward pass, the float16 gradients are converted to float32; an optimization step is done on the 
    float32 parameter copy; and the updated parameters are converted back to float16.

    Memory usage (in addition to float16 model parameters and gradients):
    - float32 copy of the model parameters
    - float32 copy of the model gradients
    - float32 "exp_avg" statistics
    - float32 "exp_avg_sq" statistics
    - intermediate tensors used for Adam computation
    Total (model + optimizer): MODEL_PARAM_COUNT * 20 bytes

    Note that the "exp_avg" and "exp_avg_sq" statistics are only created at the first optimizer
    step. So OOM errors might manifest after the forward and backward pass or only at the second forward pass.
    """
    def __init__(
        self,
        params: list[Float16Parameter],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        self.fp16_params = params
        self.param_sizes = [p.numel() for p in params]
        self.fp32_params = self.build_fp32_params(params)
        super().__init__(self.params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    @classmethod
    def build_fp32_params(cls, params: list[Float16Parameter]) -> list[Float32Parameter]:
        fp32_params = []
        for fp16_param in params:
            fp32_param = torch.nn.Parameter(fp16_param.detach().float())
            fp32_param.grad = torch.zeros_like(fp32_param)
            fp32_param._is_sharded = getattr(fp16_param, '_is_sharded', False)
            fp32_params.append(fp32_param)
        return fp32_params

    def fp16_to_fp32(self) -> None:
        for fp16_param, fp32_param in zip(self.fp16_params, self.fp32_params):
            if not fp16_param.requires_grad:
                continue
            grad_data = torch.zeros_like(fp16_param) if fp16_param.grad is None else fp16_param.grad
            fp32_param.grad.copy_(grad_data)

    def fp32_to_fp16(self) -> None:
        for fp16_param, fp32_param in zip(self.fp16_params, self.fp32_params):
            fp16_param.data.copy_(fp32_param.data)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.fp16_params:
            p.grad = None
        for p in self.params:
            p.grad.zero_()

    @property
    def params(self) -> list[Float32Parameter]:
        return self.fp32_params


class FlatFP16Adam(FP16Adam):
    """
    Partially copied from fairseq.
    
    Like in fairseq, we keep a copy of the model parameters as a flattened float32 vector.
    After the backward pass, the float16 gradients are converted to float32; an optimization step is done on the 
    float32 parameter copy; and the updated parameters are converted back to float16.
    
    This can reach higher peak memory usage than FP16Adam because the Adam computation needs to build
    several intermediate tensors with the same size as the entire model.
    """
    @classmethod
    def build_fp32_params(cls, params: list[Float16Parameter]) -> list[Float32Parameter]:
        total_param_size = sum(p.data.numel() for p in params)
        fp32_params = params[0].new(0).float().new(total_param_size)
        offset = 0
        for p in params:
            numel = p.data.numel()
            fp32_params[offset:offset + numel].copy_(p.data.view(-1))
            offset += numel
        fp32_params = torch.nn.Parameter(fp32_params)
        fp32_params.grad = fp32_params.data.new(total_param_size)
        return fp32_params

    def fp16_to_fp32(self) -> None:
        offset = 0
        for p in self.fp16_params:
            if not p.requires_grad:
                continue
            grad_data = p.grad.data if p.grad is not None else p.data.new_zeros(p.data.shape)
            numel = grad_data.numel()
            self.fp32_params.grad.data[offset:offset + numel].copy_(grad_data.view(-1))
            offset += numel

    def fp32_to_fp16(self) -> None:
        offset = 0
        for p in self.fp16_params:
            numel = p.data.numel()
            p.data.copy_(
                self.fp32_params
                .data[offset:offset + numel]
                .view_as(p.data)
            )
            offset += numel
    
    @property
    def params(self) -> list[Float32Parameter]:
        return [self.fp32_params]

    def state_dict(self) -> dict:
        # unflatten the optimizer parameters
        state_dict = super().state_dict()
        if not state_dict['state']:
            return state_dict
        state = state_dict['state'].pop(0)
        offset = 0
        for i, p in enumerate(self.fp16_params):
            numel = p.data.numel()
            state_dict['state'][i] = {'step': state['step']}
            for k in 'exp_avg', 'exp_avg_sq':
                v = state[k][offset:offset + numel].view(p.shape)
                state_dict['state'][i][k] = v
            offset += numel
        state_dict['param_groups'][0]['params'] = sorted(state_dict['state'].keys())
        return state_dict
    
    def load_state_dict(self, state_dict: dict) -> None:
        # Flatten the optimizer parameters
        state = state_dict.pop('state')
        step = next((state_['step'] for state_ in state.values() if 'step' in state_), 0)
        new_state = {'step': step}
        for k in 'exp_avg', 'exp_avg_sq':
            new_state[k] = torch.cat([state[i][k].view(-1) for i in sorted(state)], dim=0)
        state_dict['state'] = {0: new_state}
        state_dict['param_groups'][0]['params'] = [0]
        return super().load_state_dict(state_dict)


def convert_fairseq_state_dict(state_dict: dict, param_sizes: dict[str, int]) -> dict:
    if not state_dict or not state_dict.get('state'):
        return None

    logger.info('found fairseq-style optimizer state, attempting to convert it')
    state = next(iter(state_dict['state'].values()))
    new_state = {}
    if state['exp_avg'].numel() == state['exp_avg_sq'].numel() == sum(param_sizes.values()):
        i = 0
        for name, size in param_sizes.items():
            new_state[name] = {
                'step': state['step'],
                'exp_avg': state['exp_avg'][i:i + size],
                'exp_avg_sq': state['exp_avg_sq'][i:i + size],
            }
            i += size
        param_groups = [{**state_dict['param_groups'][0], 'params': list(new_state)}]
        return {'state': new_state, 'param_groups': param_groups}
    else:
        logger.warning('failed to re-map the optimizer state, resetting it')
        return None


def update_state_dict(state_dict: dict, param_names: list[str], param_shapes: list[torch.Size]) -> None:
    if not state_dict or not state_dict.get('state'):
        return None
    
    state_dict['state'] = {k.removeprefix('module.'): v for k, v in state_dict['state'].items()}
    
    state = state_dict['state']
    step = next(iter(state.values()))['step']

    missing = set()
    wrong_shape = set()

    zeros = lambda shape: {'step': step, 'exp_avg': torch.zeros(shape), 'exp_avg_sq': torch.zeros(shape)}

    new_state = {}
    for name, shape in zip(param_names, param_shapes):
        if name in state:
            state_ = dict(state[name])
            try:
                state_['exp_avg'] = state_['exp_avg'].view(shape)
                state_['exp_avg_sq'] = state_['exp_avg_sq'].view(shape)
            except:
                state_ = zeros(shape)
                wrong_shape.add(name)
            new_state[name] = state_
        else:
            new_state[name] = zeros(shape)
            missing.add(name)

    unused = set(state) - set(new_state)

    if unused:
        logger.info('unused optimizer states: ' + ' '.join(unused))
    if missing:
        logger.info('missing optimizer states initialized to zero: ' + ' '.join(missing))
    if wrong_shape:
        logger.info('wrongly-shaped optimizer states initialized to zero: ' + ' '.join(wrong_shape))

    state_dict['state'] = new_state
    state_dict['param_groups'][0]['params'] = list(new_state)


class GradScaler(torch.cuda.amp.GradScaler):
    def unscale_(self, optimizer: torch.optim.Optimizer, sharded: bool = False) -> None:
        if hasattr(optimizer, 'fp16_to_fp32'):
            optimizer.fp16_to_fp32()
        super().unscale_(optimizer)

        if sharded and self.is_enabled():  # copied from Pytorch's ShardedGradScaler
            optimizer_state = self._per_optimizer_states[id(optimizer)]
            handles = []
            for v in optimizer_state['found_inf_per_device'].values():
                handles.append(dist.all_reduce(v, async_op=True).get_future())
            if handles:
                torch.futures.wait_all(handles)

    def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_fp16):
        return super()._unscale_grads_(optimizer, inv_scale, found_inf, True)

    def step(self, optimizer):
        value = super().step(optimizer)
        if hasattr(optimizer, 'fp32_to_fp16'):
            optimizer.fp32_to_fp16()
        return value


@torch.no_grad()
def clip_grad_norm_(params, max_norm: float, sharded: bool = False, shard_id: int = 0) -> Tensor:
    """
    Computes total gradient norm and clips gradients if max_norm > 0
    The total gradient norm is reduced across all GPUs if sharded is True (e.g., with FSDP)
    """
    if isinstance(params, torch.Tensor):
        params = [params]
    
    params = [p for p in params if p is not None and getattr(p, 'grad', None) is not None]

    grads = [
        p.grad.detach() for p in params
        if (shard_id == 0 or getattr(p, '_is_sharded', False))
    ]  # avoid counting the non-sharded parameters twice (gradients for these parameters were reduced already by DDP)

    if len(grads) > 0:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(g, p=2, dtype=torch.float) for g in grads]
            )
        )
    else:
        device = params[0].device if len(params) > 0 else None
        total_norm = torch.tensor(0.0, device=device)

    if sharded:
        total_norm = total_norm ** 2
        dist.all_reduce(total_norm)
        total_norm = total_norm**0.5 

    if max_norm > 0:
        max_norm = float(max_norm)
        clip_coef = (max_norm / (total_norm + 1e-6)).clamp_(max=1)
        for p in params:
            p.grad.detach().mul_(clip_coef)

    return total_norm
