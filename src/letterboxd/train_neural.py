"""Neural-network training entry point for Letterboxd (intentionally idle).

Architecture and command-line guard are present now, but no model is trained
unless an operator explicitly supplies ``--train`` after the scale benchmark.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import MODELS, RATING_MAX, RATING_MIN, RATINGS_PARQUET, RESULTS, SEED


@dataclass(frozen=True)
class NeuralConfig:
    embedding_dim: int = 64
    width: int = 512
    depth: int = 4
    epochs: int = 30
    batch_size: int = 8192


class RatingMLP(nn.Module):
    def __init__(self, n_users: int, n_movies: int, config: NeuralConfig):
        super().__init__()
        self.user = nn.Embedding(n_users, config.embedding_dim)
        self.movie = nn.Embedding(n_movies, config.embedding_dim)
        layers: list[nn.Module] = []
        for _ in range(config.depth):
            layers += [nn.Linear(config.width if layers else config.embedding_dim * 2, config.width), nn.ReLU(), nn.Dropout(.1)]
        self.head = nn.Sequential(*layers, nn.Linear(config.width, 1))

    def forward(self, user: torch.Tensor, movie: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.user(user), self.movie(movie)], dim=1)).squeeze(1)


def train_model(config: NeuralConfig) -> dict:
    """Full opt-in Design 3 training path; never called without ``--train``.

    One rating per member is withheld before factorization. The remaining
    member/movie rows train a small embedding MLP in mini-batches; therefore
    no held-out target enters the embedding loss for that member.
    """
    ratings = pd.read_parquet(RATINGS_PARQUET).sort_values(["user_id", "movie_id"]).copy()
    held = ratings.groupby("user_id", group_keys=False).sample(n=1, random_state=SEED).index
    test = ratings.loc[held].copy()
    train = ratings.drop(index=held).copy()
    users = pd.Index(ratings.user_id.drop_duplicates())
    movies = pd.Index(ratings.movie_id.drop_duplicates())
    for frame in (train, test):
        frame["u"] = pd.Categorical(frame.user_id, categories=users).codes
        frame["m"] = pd.Categorical(frame.movie_id, categories=movies).codes
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = RatingMLP(len(users), len(movies), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    u = train.u.to_numpy(np.int64); m = train.m.to_numpy(np.int64); y = train.rating.to_numpy(np.float32)
    rng = np.random.default_rng(SEED)
    for _ in range(config.epochs):
        for start in range(0, len(train), config.batch_size):
            idx = rng.integers(0, len(train), size=min(config.batch_size, len(train)))
            optimizer.zero_grad()
            pred = model(torch.as_tensor(u[idx], device=device), torch.as_tensor(m[idx], device=device))
            loss_fn(pred, torch.as_tensor(y[idx], device=device)).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(test.u.to_numpy(np.int64), device=device),
                     torch.as_tensor(test.m.to_numpy(np.int64), device=device)).cpu().numpy()
    pred = np.clip(pred, RATING_MIN, RATING_MAX)
    report = {"model": "embedding_mlp", "rating_scale": [RATING_MIN, RATING_MAX],
              "train_rows": int(len(train)), "test_rows": int(len(test)),
              "rmse": float(np.sqrt(np.mean((pred - test.rating.to_numpy()) ** 2))),
              "architecture": config.__dict__, "device": str(device)}
    MODELS.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "users": len(users), "movies": len(movies),
                "config": config.__dict__}, MODELS / "letterboxd_neural.pt")
    (RESULTS / "neural_results.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Explicitly permit expensive neural training")
    args = parser.parse_args()
    if not args.train:
        print("Neural-network code is ready but training is intentionally skipped. Pass --train only when you explicitly want to begin that experiment.")
        return
    if not RATINGS_PARQUET.exists():
        raise FileNotFoundError("Run python -m letterboxd.preprocess first.")
    print(json.dumps(train_model(NeuralConfig()), indent=2))


if __name__ == "__main__":
    main()
