import torch.cuda
from torch import nn
from torch.nn.init import trunc_normal_


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
        trunc_normal_(
            self.W,
            mean=0.0,  # 均值
            std=0.02,  # 标准差（根据场景调整，如Transformer常用0.02）
            a=-2.0 * 0.02,  # 下界 = 均值 - 2*标准差
            b=2.0 * 0.02  # 上界 = 均值 + 2*标准差
        )

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
        trunc_normal_(
            self.W,
            mean=0.0,  # 均值
            std=0.02,  # 标准差（根据场景调整，如Transformer常用0.02）
            a=-2.0 * 0.02,  # 下界 = 均值 - 2*标准差
            b=2.0 * 0.02  # 上界 = 均值 + 2*标准差
        )

    # x.shape: [batch_size, num_queries]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = to_onehot(x, self.num_vocab).to(self.W.dtype)
        return y @ self.W
