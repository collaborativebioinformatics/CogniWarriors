"""The model that travels between the NVFlare server and the sites.

FedFusionModel = FusionRegressor (training/fusion_model.py)
               + global feature-scaling buffers (mean/std per input block)
               + optional LMMNN variance scalars (training/lmmnn_loss.py)

The scaling lives INSIDE the model, so:
  * every site uses the same global mean/std (filled in once by the server
    from per-site sums, see controller.py) -- not its own local statistics
  * the saved global model takes RAW features and is self-contained
  * FedAvg averages the buffers too, but they are identical at every site,
    so the average leaves them unchanged
"""

import torch
import torch.nn as nn

import config
from fusion_model import FusionRegressor
from lmmnn_loss import LMMNNLoss

BLOCKS = ("image", "phenotype", "covariates")


class FedFusionModel(nn.Module):
    def __init__(self, image_dim=80, phenotype_dim=91, n_covariates=2, n_targets=1, use_lmmnn=False):
        super().__init__()
        for block, dim in zip(BLOCKS, (image_dim, phenotype_dim, n_covariates)):
            self.register_buffer(f"{block}_mean", torch.zeros(dim))
            self.register_buffer(f"{block}_std", torch.ones(dim))
        self.net = FusionRegressor(
            image_dim, phenotype_dim, n_covariates, n_targets=n_targets, **config.FUSION_ARCHITECTURE
        )
        # the 2 variance scalars per target are learned + averaged like any weight
        self.lmmnn = LMMNNLoss(n_targets) if use_lmmnn else None

    def forward(self, image, phenotype, covariates):
        """Raw (unscaled) features in, predictions out."""
        return self.net(
            (image - self.image_mean) / self.image_std,
            (phenotype - self.phenotype_mean) / self.phenotype_std,
            (covariates - self.covariates_mean) / self.covariates_std,
        )
