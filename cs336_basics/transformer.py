import torch
from torch import nn
from cs336_basics.base_module import (
    EmbeddingLayer,
    RMSNormLayer,
    SwiGLU,
    RotaryPositionalEmbedding,
    MultiHeadAttention,
)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,

        theta: float | None = None,  # Θ value for the RoPE
        max_seq_len: int | None = None,  # Maximum sequence length that will be inputted
    ):
        """
        :param d_model: Dimensionality of the Transformer block inputs
        :param num_heads: Number of heads to use in multi-head self-attention
        :param d_ff: Dimensionality of the position-wise feed-forward inner layer
        """
        super().__init__()
        self.norm = SwiGLU(d_model, d_ff, device, dtype)
        self.multihead_attention = MultiHeadAttention(d_model, num_heads, device, dtype, theta, max_seq_len)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    # x.shape: (batch_size, seq_len, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.multihead_attention(self.norm(x))
        return y + self.ffn(self.norm(y))