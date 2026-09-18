#!/usr/bin/env python3
"""Compares genuinely different model FAMILIES for the fusion task -- not
MLP architecture variants (that's architecture_search.py), but classical
regressors fit on the flat concat(image, phenotype, covariates) feature
vector: Ridge, linear/RBF SVR, Random Forest, gradient boosting, k-NN.
The best MLP fusion architectures are included too, purely as a reference
point, on the exact same subject-grouped train/val split (see
architecture_search.py / train_fusion_lmmnn.py for the split logic) so
every number below is directly comparable.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import config
from architecture_search import ARCHITECTURES, SEEDS, load_split, to_device_tensors, train_one

# Each entry takes a seed (ignored by deterministic models) so every family
# is evaluated across the same SEEDS as the MLP search for a fair spread.
SKLEARN_MODELS = {
    "ridge": lambda seed: Ridge(alpha=10.0),
    "svr_linear": lambda seed: SVR(kernel="linear", C=1.0),
    "svr_rbf": lambda seed: SVR(kernel="rbf", C=1.0, gamma="scale"),
    "knn": lambda seed: KNeighborsRegressor(n_neighbors=5),
    "random_forest": lambda seed: RandomForestRegressor(n_estimators=300, max_depth=4, random_state=seed),
    "gradient_boosting": lambda seed: HistGradientBoostingRegressor(max_depth=3, random_state=seed),
}

# MLP fusion architectures (from architecture_search.py) included as the
# reference point for "is a bigger/different model family actually better".
MLP_REFERENCE_ARCHITECTURES = ["current", "wide"]


def flat_features(dataset):
    return np.concatenate([dataset.image, dataset.phenotype, dataset.covariates], axis=1)


def evaluate_sklearn(model_fn, X_train, y_train, X_val, y_val, seeds):
    val_mses = []
    for seed in seeds:
        model = model_fn(seed)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        val_mses.append(np.mean((pred - y_val) ** 2))
    return float(np.mean(val_mses)), float(np.std(val_mses))


def main():
    train_ds, val_ds, test_ds, image_dim, pheno_dim, n_covariates, n_targets = load_split()
    X_train, X_val = flat_features(train_ds), flat_features(val_ds)
    target_cols = config.TARGET_COLUMNS

    results = []
    for t, col in enumerate(target_cols):
        y_train_full, y_val_full = train_ds.targets[:, t], val_ds.targets[:, t]
        train_mask, val_mask = ~np.isnan(y_train_full), ~np.isnan(y_val_full)
        Xt_train, yt_train = X_train[train_mask], y_train_full[train_mask]
        Xt_val, yt_val = X_val[val_mask], y_val_full[val_mask]

        for name, model_fn in SKLEARN_MODELS.items():
            mean_mse, std_mse = evaluate_sklearn(model_fn, Xt_train, yt_train, Xt_val, yt_val, SEEDS)
            results.append({"target": col, "family": name, "val_mse_mean": mean_mse, "val_mse_std": std_mse, "n_params": "-"})

    # MLP is inherently multi-output, so it's trained once (jointly over all
    # targets) rather than per-target -- with the default single-target
    # config the two framings coincide; flagged here for the 5-target case.
    train_tensors = to_device_tensors(train_ds)
    val_tensors = to_device_tensors(val_ds)
    joint_label = "+".join(target_cols)
    for name in MLP_REFERENCE_ARCHITECTURES:
        val_losses, n_params = [], None
        for seed in SEEDS:
            val_loss, n_params = train_one(
                ARCHITECTURES[name], train_tensors, val_tensors, image_dim, pheno_dim, n_covariates, n_targets, seed
            )
            val_losses.append(val_loss)
        results.append({
            "target": joint_label, "family": f"mlp_{name}",
            "val_mse_mean": float(np.mean(val_losses)), "val_mse_std": float(np.std(val_losses)),
            "n_params": n_params,
        })

    results.sort(key=lambda r: r["val_mse_mean"])

    print(f"=== Fusion model family comparison ({len(SEEDS)} seeds, flat concat(image, phenotype, covariates) features) ===")
    print(f"{'rank':<5}{'target':<14}{'family':<18}{'val_MSE':<20}{'params'}")
    for rank, r in enumerate(results, start=1):
        val_str = f"{r['val_mse_mean']:.4f}+/-{r['val_mse_std']:.4f}"
        print(f"{rank:<5}{r['target']:<14}{r['family']:<18}{val_str:<20}{r['n_params']}")

    best = results[0]
    print(f"\nBest: '{best['family']}' on target(s) '{best['target']}' (val MSE = {best['val_mse_mean']:.4f} +/- {best['val_mse_std']:.4f})")


if __name__ == "__main__":
    main()
