import random

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

def get_batch(
    dataset: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    n = len(dataset)
    # [0, n-m-1]
    assert batch_size <= n - context_length - 1

    start_indices = torch.tensor(
        random.sample(range(n - context_length), batch_size),
        device=device,
        dtype=torch.int
    )
    offset = torch.arange(context_length, device=device, dtype=torch.int)

    x_indices = start_indices.unsqueeze(1) + offset

    x, y = dataset[x_indices], dataset[x_indices+1]
    return x.to(device), y.to(device)