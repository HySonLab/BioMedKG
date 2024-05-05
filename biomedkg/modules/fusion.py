import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from biomedkg.modules.utils import parameters_count


class AttentionFusion(nn.Module):
    def __init__(self,
                 embed_dim : int,
                 norm : bool = True,
                 ):
        super().__init__()

        self.norm = norm
        self.embed_dim = embed_dim

        self.pos_encoder = PositionalEncoding(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, 
                x : torch.tensor, 
                ) -> torch.tensor:

        batch_size = x.size(0)

        x = x.view(batch_size, -1, self.embed_dim)
        
        if self.norm:
            x = F.normalize(x, dim=-1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
        )
        
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x : torch.tensor) -> torch.tensor:
        """
        x of shape [batch_size, seq_len, embed_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

if __name__ == "__main__":
    embed_dim = 768
    num_modality = 2

    model = AttentionFusion(
        embed_dim=embed_dim,
        norm=True,
        aggr="mean",
    )

    print(model)

    print(f"Total parameters: {parameters_count(model):,}")

    x = torch.rand(1, embed_dim * num_modality)
    out = model(x)
    print(out.size())

    batch_size = 16
    x = torch.rand(batch_size, 1, embed_dim * num_modality)
    out = model(x)

    print(out.size())
