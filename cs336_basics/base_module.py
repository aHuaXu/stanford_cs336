import math

import torch.cuda
from torch import nn
from torch.nn.init import trunc_normal_
from jaxtyping import Bool, Float, Int
from torch import Tensor

def init_params(x: torch.Tensor):
    trunc_normal_(
        x,
        mean=0.0,  # 均值
        std=0.02,  # 标准差（根据场景调整，如Transformer常用0.02）
        a=-2.0 * 0.02,  # 下界 = 均值 - 2*标准差
        b=2.0 * 0.02  # 上界 = 均值 + 2*标准差
    )

class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.cuda.device | None =None,
        dtype: torch.dtype  | None = None
    ):
        super(LinearLayer, self).__init__()
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        init_params(self.W)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return x @ self.W.T

def to_onehot(x: torch.Tensor, num_vocab: int) -> torch.Tensor:
    batch_size, num_queries = x.shape
    one_hot = torch.zeros(batch_size, num_queries, num_vocab, device=x.device, dtype=x.dtype)
    one_hot.scatter_(2, x.unsqueeze(2), 1)
    return one_hot

class EmbeddingLayer(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        """
        Construct an embedding module.

        Parameters:
            num_embeddings: int
                Size of the vocabulary

            embedding_dim: int
                Dimension of the embedding vectors, i.e., dmodel

            device: torch.device | None = None
                Device to store the parameters on

            dtype: torch.dtype | None = None
                Data type of the parameters
        """
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.num_vocab = num_embeddings
        init_params(self.W)

    # x.shape: [batch_size, num_queries]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = to_onehot(x, self.num_vocab).to(self.W.dtype)
        return y @ self.W

# Root Mean Square Layer Normalization
class RMSNormLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.g = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        init_params(self.g)
        self.eps = eps

    # x.shape: (batch_size, sequence_length, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.type(torch.float32)
        x_sq = x ** 2

        # (batch_size, sequence_length, 1)
        rms = torch.sqrt(x_sq.mean(dim=-1, keepdim=True) + self.eps)

        # broadcast
        x_norm = x / rms
        return (x_norm * self.g).to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        # base_dff = (8/3) * d_model
        # diff = round(base_dff/64) * 64
        # d_hidden: int = max(diff, 64)

        self.linear1 = LinearLayer(d_model, d_ff, device=device, dtype=dtype)
        self.linear3 = LinearLayer(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = LinearLayer(d_ff, d_model, device=device, dtype=dtype)
        self.siLu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.siLu(self.linear1(x)) * self.linear3(x)
        return self.linear2(hidden)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,   # Θ value for the RoPE
        d_k: int,   # int dimension of query and key vectors
        max_seq_len: int,   # Maximum sequence length that will be inputted
        device: torch.device | None = None #  Device to store the buffer on
    ):
        super().__init__()
        # (dk//2, ) md文档没括号
        thetas = theta ** (- (2 * torch.arange(d_k//2, device=device)) / d_k)
        # (max_seq_len, 1)
        pos = torch.arange(max_seq_len, device=device).unsqueeze(1)
        # (max_seq_len, dk//2)
        pos_theta = pos * thetas # broadcast

        self.register_buffer('rope_cos', torch.cos(pos_theta), persistent=False)
        self.register_buffer('rope_sin', torch.sin(pos_theta), persistent=False)

    # x.shape: (..., seq_len, d_k)
    # token_positions shape: (..., seq_len)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        *batch_dim, seq_len, d_k = x.shape

        x_even, x_odd = x[..., ::2], x[..., 1::2]

        # cos,sin shape: (..., seq_len, d_k//2)
        cos, sin = self.rope_cos[token_positions], self.rope_sin[token_positions]

        new_x_even, new_x_odd = cos * x_even - sin * x_odd, cos * x_odd + sin * x_even

        y = torch.zeros_like(x)
        y[..., ::2], y[..., 1::2] = new_x_even, new_x_odd
        return y

def softmax(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum

def scaled_dot_product_attention(
    q: Float[Tensor, " ... queries d_k"],
    k: Float[Tensor, " ... keys d_k"],
    v: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    queries, d_k = q.shape[-2:]
    keys, d_k = k.shape[-2:]
    values, d_v = v.shape[-2:]
    assert keys == values

    # scores.shape: (..., queries, keys)
    scores = torch.matmul(q, k.transpose(-1, -2))/math.sqrt(d_k)
    if mask is not None:
        scores[~mask] = -1e6
    softmax_scores = softmax(scores, dim=-1)

    return torch.matmul(softmax_scores, v)

