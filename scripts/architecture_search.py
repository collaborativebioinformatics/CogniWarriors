#!/usr/bin/env python3
"""Architecture search for the fusion model: trains several candidate
FusionRegressor configurations (varying embedder width and fusion-head
depth/width) on the same subject-grouped train/val split, plain MSE, and
reports which one generalizes best.

Plain MSE (not the LMMNN loss) is used here deliberately -- this is an
architecture search, and comparing candidates on the model-selection metric
they'll ultimately be judged on (predictive MSE) is more informative than
comparing NLLs that also depend on architecture-independent variance
parameters. Once a winning architecture is picked, plug its
head_hidden_dims/embedder_hidden_dim/embedding_dim into train_fusion_lmmnn.py.

Runs on MPS (Apple GPU) / CUDA / CPU automatically via device_utils.get_device().
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import config
from device_utils import get_device
from fusion_model import FusionRegressor
from multimodal_data import MultimodalDataset, load_multimodal_dataframe
from train_fusion_lmmnn import BASELINE_MAX_EPOCHS, SEED, grouped_split, masked_mse, scale_columns

DEVICE = get_device()
SEEDS = [0, 1, 2]

# Candidate fusion-model architectures. "current" matches today's defaults
# (fusion_model.py / config.py) -- everything else is a genuine structural
# change, not just a hyperparameter nudge: linear_head has NO hidden layer
# in the fusion head at all, deep_head has two.
ARCHITECTURES = {
    "tiny": dict(image_embedding_dim=16, phenotype_embedding_dim=16, embedder_hidden_dim=32, head_hidden_dims=[32]),
    "current": dict(image_embedding_dim=32, phenotype_embedding_dim=32, embedder_hidden_dim=64, head_hidden_dims=[64]),
    "wide": dict(image_embedding_dim=64, phenotype_embedding_dim=64, embedder_hidden_dim=128, head_hidden_dims=[128]),
    "deep_head": dict(image_embedding_dim=32, phenotype_embedding_dim=32, embedder_hidden_dim=64, head_hidden_dims=[64, 32]),
    "linear_head": dict(image_embedding_dim=32, phenotype_embedding_dim=32, embedder_hidden_dim=64, head_hidden_dims=[]),
}


def load_split():
    df, phenotype_cols, covariate_cols, image_cols = load_multimodal_dataframe()
    trainval_idx, test_idx = grouped_split(df, test_size=0.2)
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    train_rel_idx, val_rel_idx = grouped_split(trainval_df, test_size=0.2, seed=SEED + 1)
    train_df = trainval_df.iloc[train_rel_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_rel_idx].reset_index(drop=True)

    train_df, (val_df, test_df), _ = scale_columns(train_df, [val_df, test_df], phenotype_cols)
    train_df, (val_df, test_df), _ = scale_columns(train_df, [val_df, test_df], image_cols)
    train_df, (val_df, test_df), _ = scale_columns(train_df, [val_df, test_df], covariate_cols)

    target_cols = config.TARGET_COLUMNS
    train_ds = MultimodalDataset(train_df, phenotype_cols, covariate_cols, image_cols, target_cols)
    val_ds = MultimodalDataset(val_df, phenotype_cols, covariate_cols, image_cols, target_cols)
    test_ds = MultimodalDataset(test_df, phenotype_cols, covariate_cols, image_cols, target_cols)
    return train_ds, val_ds, test_ds, len(image_cols), len(phenotype_cols), len(covariate_cols), len(target_cols)


def to_device_tensors(dataset):
    return (
        torch.from_numpy(dataset.image).to(DEVICE),
        torch.from_numpy(dataset.phenotype).to(DEVICE),
        torch.from_numpy(dataset.covariates).to(DEVICE),
        torch.from_numpy(dataset.targets).to(DEVICE),
    )


def train_one(arch_kwargs, train_tensors, val_tensors, image_dim, pheno_dim, n_covariates, n_targets, seed):
    torch.manual_seed(seed)
    model = FusionRegressor(image_dim, pheno_dim, n_covariates, n_targets=n_targets, **arch_kwargs).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    train_image, train_pheno, train_cov, train_targets = train_tensors
    val_image, val_pheno, val_cov, val_targets = val_tensors

    best_loss, patience_left = float("inf"), config.PATIENCE
    for _ in range(BASELINE_MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = masked_mse(model(train_image, train_pheno, train_cov), train_targets)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = masked_mse(model(val_image, val_pheno, val_cov), val_targets).item()
        if val_loss < best_loss:
            best_loss, patience_left = val_loss, config.PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    n_params = sum(p.numel() for p in model.parameters())
    return best_loss, n_params


def main():
    print(f"device: {DEVICE}")
    train_ds, val_ds, test_ds, image_dim, pheno_dim, n_covariates, n_targets = load_split()
    train_tensors = to_device_tensors(train_ds)
    val_tensors = to_device_tensors(val_ds)

    results = []
    for name, arch_kwargs in ARCHITECTURES.items():
        start = time.time()
        val_losses = []
        n_params = None
        for seed in SEEDS:
            val_loss, n_params = train_one(
                arch_kwargs, train_tensors, val_tensors, image_dim, pheno_dim, n_covariates, n_targets, seed
            )
            val_losses.append(val_loss)
        elapsed = time.time() - start
        results.append(
            {
                "name": name,
                "val_mse_mean": np.mean(val_losses),
                "val_mse_std": np.std(val_losses),
                "n_params": n_params,
                "seconds": elapsed,
                **arch_kwargs,
            }
        )

    results.sort(key=lambda r: r["val_mse_mean"])

    print(f"\n=== Fusion model architecture search ({len(SEEDS)} seeds each, {BASELINE_MAX_EPOCHS} max epochs, MSE) ===")
    header = f"{'rank':<5}{'name':<14}{'val_MSE':<18}{'params':<9}{'sec':<7}{'embed(img/pheno)':<18}{'embedder_hidden':<17}{'head_hidden_dims'}"
    print(header)
    for rank, r in enumerate(results, start=1):
        val_str = f"{r['val_mse_mean']:.4f}+/-{r['val_mse_std']:.4f}"
        embed_str = f"{r['image_embedding_dim']}/{r['phenotype_embedding_dim']}"
        print(
            f"{rank:<5}{r['name']:<14}{val_str:<18}{r['n_params']:<9}{r['seconds']:<7.1f}"
            f"{embed_str:<18}{r['embedder_hidden_dim']:<17}{r['head_hidden_dims']}"
        )

    best = results[0]
    print(f"\nBest architecture: '{best['name']}' (val MSE = {best['val_mse_mean']:.4f} +/- {best['val_mse_std']:.4f})")
    print("To adopt it: pass these kwargs to FusionRegressor(...) in train_fusion_lmmnn.py, "
          "or set them as the new defaults in fusion_model.py / config.py:")
    for k in ("image_embedding_dim", "phenotype_embedding_dim", "embedder_hidden_dim", "head_hidden_dims"):
        print(f"  {k} = {best[k]}")


if __name__ == "__main__":
    main()
