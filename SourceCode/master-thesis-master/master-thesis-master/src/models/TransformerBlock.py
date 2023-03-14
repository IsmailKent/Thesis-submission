
import torch
import torch.nn as nn
from models.nn_utils import init_xavier_
from models import MultiHeadSelfAttention, DotProdCrossAttention


class TransformerBlock(nn.Module):
    """Tranformer decoder block."""

    def __init__(
                self,
                embed_dim: int,
                num_heads: int,
                qkv_size: int,
                mlp_size: int,
                pre_norm: bool = False,
                weight_init=None
            ):
        super().__init__()

        self.embed_dim = embed_dim
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        self.weight_init = weight_init

        assert num_heads >= 1
        assert qkv_size % num_heads == 0, "embed dim must be divisible by num_heads"
        self.head_dim = qkv_size // num_heads

        # submodules
        #  MHA
        self.attn = MultiHeadSelfAttention(
            emb_dim=embed_dim,
            num_heads=num_heads,
        )
        #  mlps
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, embed_dim),
        )
        #  layernorms
        self.layernorm_query = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        if self.num_heads > 1:
            self.dense_o = nn.Linear(qkv_size, embed_dim)
            self.multi_head = True
        else:
            self.multi_head = False
        self._init_model()
        return

    @torch.no_grad()
    def _init_model(self):
        init_xavier_(self)

    def forward(self, inputs):
        assert inputs.ndim == 3
        B, L, _ = inputs.shape

        if self.pre_norm:
            # Self-attention.
            x = self.layernorm_query(inputs)
            x = self.attn(x)
            x = x + inputs

            y = x

            # MLP
            z = self.layernorm_mlp(y)
            z = self.mlp(z)
            z = z + y
        else:
            # Self-attention on queries.
            x = self.attn(inputs)
            x = x + inputs
            x = self.layernorm_query(x)

            y = x

            # MLP
            z = self.mlp(y)
            z = z + y
            z = self.layernorm_mlp(z)
        return z
