from torch import nn
from cs336_basics.base_module import (
    softmax,
    to_onehot,
)
import torch

def log_softmax(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    log_exp_x = torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))
    return x - log_exp_x

def cross_entropy_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (torch.Tensor): The input tensor of shape (batch_size, vocab_size).
        targets (torch.Tensor): The target tensor of shape (batch_size,).

    Returns:
        torch.Tensor: The average cross-entropy loss across examples.
    """

    # this method has loss, and lead to wrong result
    # softmax_inputs = softmax(inputs, -1)
    # gather_inputs = softmax_inputs.gather(1, targets.unsqueeze(1)).squeeze(1)
    # return torch.mean(-torch.log(softmax_inputs))

    """
    • Subtract the largest element for numerical stability.
    • Cancel out log and exp whenever possible.
    """
    log_softmax_inputs = log_softmax(inputs, dim=-1)
    gather_inputs = log_softmax_inputs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return torch.mean(-gather_inputs)