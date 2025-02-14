from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


def cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor) -> torch.Tensor:
    log_probs = log_softmax(inputs, dim=1)
    loss = -log_probs[range(inputs.shape[0]), targets].mean()
    return loss

def log_softmax(input: torch.FloatTensor, dim: int)-> torch.FloatTensor:
    #Same thing as the softmax function but we do the log calcualtion before the subtraction for numerical stability
    c = torch.max(input, dim=dim, keepdim=True).values
    scaled = input - c
    log_sum_exp = torch.log(torch.sum(torch.exp(scaled), dim=dim, keepdim=True))
    return scaled - log_sum_exp

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
    

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                t = state['step']

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias corrections
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                # Compute step-size
                alpha_t = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Apply decoupled weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])  # Corrected

                # Update parameters
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-alpha_t)

        return loss


def cosine_schedule(it: int, max_lr: float, min_lr: float, warmup_iters: int, cosine_cycle_iters: int):
    #Convert to float to ensure no loss in division
    it = float(it)
    warmup_iters = float(warmup_iters)
    if(it < warmup_iters):
        return ((it / warmup_iters) * max_lr)
    elif (it > cosine_cycle_iters):
        return min_lr
    else:
        return (min_lr + (0.5*(1.0+ math.cos(((it-warmup_iters)/(cosine_cycle_iters-warmup_iters))*math.pi)))*(max_lr-min_lr))


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    # total_norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in parameters))
    torch.nn.utils.clip_grad_norm_(parameters, max_l2_norm)
    # if total_norm > max_l2_norm:
    #     scale = max_l2_norm / (total_norm + (1e-6))
    #     for p in parameters:
    #         p.grad.mul_(scale)

    

if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e3)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.
