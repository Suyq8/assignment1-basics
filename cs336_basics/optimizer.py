from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

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
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if not isinstance(betas, Iterable) or len(betas) != 2:
            raise ValueError(f"Invalid betas: {betas}")
        beta1, beta2 = betas
        if beta1 < 0 or beta1 >= 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if beta2 < 0 or beta2 >= 1:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps <= 0:
            raise ValueError(f"Invalid eps: {eps}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "beta1": beta1, "beta2": beta2, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["m"] = torch.zeros_like(p.data)
                state["v"] = torch.zeros_like(p.data)
            t = state["step"] + 1
            grad = p.grad.data
            m = beta1 * state["m"] + (1 - beta1) * grad
            v = beta2 * state["v"] + (1 - beta2) * grad ** 2
            state["m"] = m
            state["v"] = v
            state["step"] = t
            
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            p.data -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * p.data)
        return loss
    
if __name__ == "__main__":
    num_iterations = 10
    for lr in [1e1, 1e2, 1e3]:
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)
        losses = []
        for t in range(num_iterations):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            losses.append(loss.cpu().item())
            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.
        print(f"Learning rate: {lr}, Loss: {losses}")