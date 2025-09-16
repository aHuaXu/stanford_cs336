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
            lr = group["lr"]  # Get the learning rate for this parameter group
        
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get optimizer state associated with parameter p
                t = state.get("t", 0)  # Get iteration number from state, defaulting to 0
                grad = p.grad.data  # Get gradient of loss with respect to parameter p
                p.data -= lr / math.sqrt(t + 1) * grad  # Update parameter using adaptive learning rate
                state["t"] = t + 1  # Increment iteration counter

        return loss
    
"""
AdamW Optimizer Algorithm

This implements the AdamW (Adam with Weight Decay) optimization algorithm as described in
"Decoupled Weight Decay Regularization" by Loshchilov & Hutter (2017).

Algorithm:
1. Initialize parameters θ, first moment vector m = 0, second moment vector v = 0
2. For each iteration t = 1, ..., T:
   a. Sample batch of data B_t
   b. Compute gradient: g = ∇_θ L(θ; B_t)
   c. Update biased first moment estimate: m = β₁m + (1-β₁)g
   d. Update biased second moment estimate: v = β₂v + (1-β₂)g²
   e. Compute bias-corrected learning rate: α_t = α * √(1-β₂^t) / (1-β₁^t)
   f. Update parameters: θ = θ - α_t * m / (√v + ε)
   g. Apply weight decay: θ = θ - α * λ * θ

Parameters:
- α (lr): Learning rate
- β₁ (beta1): Exponential decay rate for first moment estimates (default: 0.9)
- β₂ (beta2): Exponential decay rate for second moment estimates (default: 0.999)
- ε (eps): Small constant for numerical stability (default: 1e-8)
- λ (weight_decay): Weight decay coefficient (default: 1e-8)
"""
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if beta1 < 0 or beta1 >= 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if beta2 < 0 or beta2 >= 1:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            beta1, beta2 = betas
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data
                
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss
            
    
    

if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e2)
    
    for t in range(20):
        opt.zero_grad()  # Reset gradients for all learnable parameters
        loss = (weights**2).mean()  # Compute scalar loss value (L2 regularization)
        print(loss.cpu().item())
        loss.backward()  # Run backward pass to compute gradients
        opt.step()  # Execute optimizer step to update parameters