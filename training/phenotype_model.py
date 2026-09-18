"""Small, heavily-regularized phenotype embedder + regression head.

N is ~200-215 sessions after the EF-composite MIN_TASKS_REQUIRED gate --
tiny for deep learning. Kept deliberately small (2 hidden layers, width
64->32) with dropout and weight decay so overfitting is visible in the
train/val gap rather than papered over by capacity. LayerNorm (not
BatchNorm) because GroupKFold folds here are only ~40 samples, too few for
stable batch statistics.
"""

import torch
import torch.nn as nn


class PhenotypeEmbedder(nn.Module):
    def __init__(self, input_dim, embedding_dim=32, hidden_dim=64, dropout=(0.4, 0.3)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout[0]),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout[1]),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x):
        return self.net(x)


class PhenotypeRegressor(nn.Module):
    """Embedder + head, with covariates concatenated in at the head only
    (per the project's own decision: covariates are explicit, not routed
    through the embedder itself)."""

    def __init__(self, input_dim, n_covariates, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedder = PhenotypeEmbedder(input_dim, embedding_dim, hidden_dim)
        self.head = nn.Linear(embedding_dim + n_covariates, 1)

    def forward(self, x, covariates):
        embedding = self.embedder(x)
        return self.head(torch.cat([embedding, covariates], dim=1)).squeeze(-1)
