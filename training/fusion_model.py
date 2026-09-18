"""Image + phenotype late-fusion regressor.

MLP_img (ImageEmbedder) is a small trainable projection over the
precomputed 80-dim structural-MRI embedding -- there's no raw-image
pipeline in this repo, just a per-session feature vector already computed
upstream, so this plays the same role a CNN encoder would in the LMMNN
handoff spec without re-deriving image features from scratch.
"""

import torch
import torch.nn as nn

from phenotype_model import PhenotypeEmbedder

import config


def build_mlp_head(input_dim, hidden_dims, output_dim, dropout=0.3):
    """Linear -> ReLU -> Dropout stack, one block per entry in hidden_dims,
    ending in a plain Linear(*, output_dim). hidden_dims=[] gives a bare
    linear head (no hidden layer at all) -- lets architecture search vary
    head depth, not just width."""
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class ImageEmbedder(nn.Module):
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


class FusionRegressor(nn.Module):
    """concat(z_img, z_pheno) [+ covariates] -> MLP head -> n_targets.

    The fixed-effect part f_NN(x) of the LMMNN decomposition; the random-
    effect variance parameters live in LMMNNLoss (lmmnn_loss.py), not here.
    """

    def __init__(
        self,
        image_input_dim,
        phenotype_input_dim,
        n_covariates,
        n_targets=None,
        image_embedding_dim=None,
        phenotype_embedding_dim=32,
        embedder_hidden_dim=64,
        head_hidden_dims=None,
        head_dropout=0.3,
    ):
        super().__init__()
        n_targets = n_targets or len(config.TARGET_COLUMNS)
        image_embedding_dim = image_embedding_dim or config.IMAGE_EMBEDDING_OUTPUT_DIM
        head_hidden_dims = [config.FUSION_HIDDEN_DIM] if head_hidden_dims is None else head_hidden_dims

        self.image_embedder = ImageEmbedder(
            image_input_dim, embedding_dim=image_embedding_dim, hidden_dim=embedder_hidden_dim
        )
        self.phenotype_embedder = PhenotypeEmbedder(
            phenotype_input_dim, embedding_dim=phenotype_embedding_dim, hidden_dim=embedder_hidden_dim
        )

        fusion_input_dim = image_embedding_dim + phenotype_embedding_dim + n_covariates
        self.head = build_mlp_head(fusion_input_dim, head_hidden_dims, n_targets, dropout=head_dropout)

    def forward(self, image_x, phenotype_x, covariates):
        z_img = self.image_embedder(image_x)
        z_pheno = self.phenotype_embedder(phenotype_x)
        fused = torch.cat([z_img, z_pheno, covariates], dim=1)
        return self.head(fused)


class SingleModalityRegressor(nn.Module):
    """One embedder (image-only or phenotype-only) + covariates -> MLP
    head -> n_targets. Single-modality ablation counterpart to
    FusionRegressor, for comparing how much signal each modality carries on
    its own."""

    def __init__(
        self, embedder, embedding_dim, n_covariates, n_targets=None, head_hidden_dims=None, head_dropout=0.3
    ):
        super().__init__()
        n_targets = n_targets or len(config.TARGET_COLUMNS)
        head_hidden_dims = [config.FUSION_HIDDEN_DIM] if head_hidden_dims is None else head_hidden_dims

        self.embedder = embedder
        self.head = build_mlp_head(embedding_dim + n_covariates, head_hidden_dims, n_targets, dropout=head_dropout)

    def forward(self, x, covariates):
        z = self.embedder(x)
        return self.head(torch.cat([z, covariates], dim=1))
