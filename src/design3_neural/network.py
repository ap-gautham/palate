"""Design 3 architecture: a residual tabular MLP with a genre embedding.

The 37 numeric features (standardized) are concatenated with a learned genre
embedding, projected to a hidden width, passed through pre-activation residual
blocks, and read out by a linear head.
"""
import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.BatchNorm1d(width), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width), nn.BatchNorm1d(width))
        self.act = nn.GELU()

    def forward(self, h):
        return self.act(h + self.net(h))


class TabularResNet(nn.Module):
    def __init__(self, n_numeric: int, n_genres: int, emb_dim: int,
                 width: int, depth: int, dropout: float):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(n_numeric)
        self.embedding = nn.Embedding(n_genres, emb_dim)
        self.proj = nn.Linear(n_numeric + emb_dim, width)
        self.blocks = nn.ModuleList([ResBlock(width, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, numeric, genre):
        x = torch.cat([self.input_norm(numeric), self.embedding(genre)], dim=1)
        h = self.proj(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(1)
