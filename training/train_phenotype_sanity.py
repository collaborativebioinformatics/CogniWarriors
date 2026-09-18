#!/usr/bin/env python3
"""Sanity-check the phenotype pipeline before it's fused with an image
embedder: can a model predict ef_composite from phenotype features at all,
and is the signal plausible (moderate correlation with diagnosis, not near-
zero or near-perfect)?

GroupKFold(5) by participant_id (a subject's sessions never split across
train/val, per the project's own splitting rule) comparing three models per
fold: mean-predictor, Ridge (StandardScaler + sklearn Ridge), and the
PhenotypeEmbedder+head MLP. Reporting Ridge alongside the MLP is the honest
check on whether the added complexity is earning its keep at N~200.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from phenotype_model import PhenotypeRegressor

N_FOLDS = 5
SEED = 0
MAX_EPOCHS = 300
PATIENCE = 20


def load_data():
    features = pd.read_csv(config.PHENOTYPE_FEATURES_TSV, sep="\t")
    composite = pd.read_csv(config.EF_COMPOSITE_TSV, sep="\t")
    df = features.merge(composite[config.ID_COLS + ["ef_composite"]], on=config.ID_COLS, how="inner")
    df = df.dropna(subset=["ef_composite"]).reset_index(drop=True)
    return df


def train_mlp(X_train, C_train, y_train, X_es, C_es, y_es):
    torch.manual_seed(SEED)
    model = PhenotypeRegressor(input_dim=X_train.shape[1], n_covariates=C_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    C_train_t = torch.tensor(C_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_es_t = torch.tensor(X_es, dtype=torch.float32)
    C_es_t = torch.tensor(C_es, dtype=torch.float32)
    y_es_t = torch.tensor(y_es, dtype=torch.float32)

    best_loss, best_state, patience_left = float("inf"), None, PATIENCE
    for _ in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t, C_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            es_loss = loss_fn(model(X_es_t, C_es_t), y_es_t).item()
        if es_loss < best_loss:
            best_loss, best_state, patience_left = es_loss, {k: v.clone() for k, v in model.state_dict().items()}, PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return model


def evaluate(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)


def main():
    df = load_data()
    with open(config.PHENOTYPE_FEATURE_MANIFEST.parent / "phenotype_feature_manifest.json") as f:
        import json
        manifest = json.load(f)
    feature_cols = manifest["embedder_input_columns"]
    covariate_cols = manifest["covariate_columns"]

    X = df[feature_cols].to_numpy(dtype=float)
    C = df[covariate_cols].to_numpy(dtype=float)
    y = df["ef_composite"].to_numpy(dtype=float)
    groups = df["participant_id"].to_numpy()

    print(f"N sessions with valid ef_composite: {len(df)} "
          f"({df['participant_id'].nunique()} unique subjects)")

    gkf = GroupKFold(n_splits=N_FOLDS)
    results = {"mean": [], "ridge": [], "mlp": []}
    oof_pred_mlp = np.full(len(df), np.nan)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        C_train, C_test = C[train_idx], C[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Mean-predictor baseline.
        pred_mean = np.full_like(y_test, y_train.mean())
        results["mean"].append(evaluate(y_test, pred_mean))

        # Ridge baseline (features + covariates together).
        xscaler, cscaler = StandardScaler(), StandardScaler()
        Xc_train = np.hstack([xscaler.fit_transform(X_train), cscaler.fit_transform(C_train)])
        Xc_test = np.hstack([xscaler.transform(X_test), cscaler.transform(C_test)])
        ridge = Ridge(alpha=10.0).fit(Xc_train, y_train)
        results["ridge"].append(evaluate(y_test, ridge.predict(Xc_test)))

        # MLP embedder + head, with an inner held-out split for early stopping.
        inner_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        sub_idx, es_idx = next(inner_split.split(X_train, y_train, groups[train_idx]))

        x_scaler_mlp, c_scaler_mlp = StandardScaler(), StandardScaler()
        X_sub = x_scaler_mlp.fit_transform(X_train[sub_idx])
        C_sub = c_scaler_mlp.fit_transform(C_train[sub_idx])
        X_es = x_scaler_mlp.transform(X_train[es_idx])
        C_es = c_scaler_mlp.transform(C_train[es_idx])
        X_test_scaled = x_scaler_mlp.transform(X_test)
        C_test_scaled = c_scaler_mlp.transform(C_test)

        model = train_mlp(X_sub, C_sub, y_train[sub_idx], X_es, C_es, y_train[es_idx])
        model.eval()
        with torch.no_grad():
            pred_mlp = model(
                torch.tensor(X_test_scaled, dtype=torch.float32),
                torch.tensor(C_test_scaled, dtype=torch.float32),
            ).numpy()
        results["mlp"].append(evaluate(y_test, pred_mlp))
        oof_pred_mlp[test_idx] = pred_mlp

        print(f"fold {fold}: n_train={len(train_idx)} n_test={len(test_idx)} "
              f"mean_MAE={results['mean'][-1][0]:.3f} "
              f"ridge_MAE={results['ridge'][-1][0]:.3f} ridge_R2={results['ridge'][-1][1]:.3f} "
              f"mlp_MAE={results['mlp'][-1][0]:.3f} mlp_R2={results['mlp'][-1][1]:.3f}")

    print("\n=== Cross-validated performance (mean +/- std across folds) ===")
    for name in ("mean", "ridge", "mlp"):
        maes = [m for m, _ in results[name]]
        r2s = [r for _, r in results[name]]
        print(f"{name:6s}  MAE={np.mean(maes):.3f}+/-{np.std(maes):.3f}  "
              f"R2={np.mean(r2s):.3f}+/-{np.std(r2s):.3f}")

    # Sanity check: correlate out-of-fold predicted/actual ef_composite
    # against diagnosis. study_group/dx_adhd/dx_psychosis are themselves
    # embedder inputs, so this checks the model hasn't collapsed onto
    # near-perfectly re-deriving diagnosis rather than learning EF signal.
    # dx_adhd/dx_psychosis already live in df as embedder-input columns;
    # only study_group is needed from participants.tsv (it was one-hot
    # encoded away in phenotype_features.tsv, so the raw label is gone).
    participants = pd.read_csv(config.PARTICIPANTS_TSV, sep="\t")[["participant_id", "study_group"]]
    df = df.merge(participants, on="participant_id", how="left")
    df["ef_composite_pred_oof"] = oof_pred_mlp

    print("\n=== Diagnosis correlation sanity check (out-of-fold MLP predictions) ===")
    for flag_col in ("dx_adhd", "dx_psychosis"):
        r_actual = df["ef_composite"].corr(df[flag_col])
        r_pred = df["ef_composite_pred_oof"].corr(df[flag_col])
        print(f"{flag_col}: r(actual)={r_actual:.3f}  r(predicted)={r_pred:.3f}")
    for group in df["study_group"].unique():
        indicator = (df["study_group"] == group).astype(int)
        r_actual = df["ef_composite"].corr(indicator)
        r_pred = df["ef_composite_pred_oof"].corr(indicator)
        print(f"study_group=={group}: r(actual)={r_actual:.3f}  r(predicted)={r_pred:.3f}")
    print("\nExpect moderate |r| (~0.2-0.5). Near 0 = model learned nothing useful; "
          ">0.8 = likely leakage / composite re-deriving diagnosis.")


if __name__ == "__main__":
    main()
