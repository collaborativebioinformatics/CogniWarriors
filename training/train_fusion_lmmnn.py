#!/usr/bin/env python3
"""Train the image+phenotype FusionRegressor with the LMMNN random-effects
loss (lmmnn_loss.py), using subject-grouped minibatches (multimodal_data.py)
so within-subject correlation from repeated imaging sessions is modeled
instead of assumed away.

Target(s): config.TARGET_COLUMNS (defaults to ["ef_composite"]; swap in the
5 per-task z-score columns there to go from 1 to 5 targets).

Acceptance checks (see handoff doc), printed at the end:
  1. no NaN/Inf in the loss or in sigma2_subject/sigma2_error during training
  2. subject-grouped batching verified (no subject split across batches)
  3. final sigma2_subject / sigma2_error per target
  4. within-subject residual ICC, LMMNN model vs. an MSE-only baseline
     (expected to decrease, not required to "look good")
"""

import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from checkpoint import predict, save_checkpoint
from fusion_model import FusionRegressor, ImageEmbedder, SingleModalityRegressor
from lmmnn_loss import LMMNNLoss
from multimodal_data import MultimodalDataset, SubjectGroupedBatchSampler, load_multimodal_dataframe
from phenotype_model import PhenotypeEmbedder

SEED = 0
BASELINE_MAX_EPOCHS = 60

# Reference palette (see project's dataviz conventions): categorical slot 1
# (blue) for train, slot 2 (orange) for val; neutral chart chrome.
COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"
COLOR_BEST_EPOCH = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"

# Categorical slots 1/2/3 (validated adjacent-pair order) -- one color per
# model in the modality-comparison plot; line style (not color) carries
# train vs val there.
MODEL_COLORS = {"image": "#2a78d6", "phenotype": "#eb6834", "fusion": "#1baf7a"}


def _style_axis(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(COLOR_TEXT_SECONDARY)
    ax.yaxis.label.set_color(COLOR_TEXT_SECONDARY)


def _mark_best_epoch(ax, best_epoch):
    if best_epoch is None:
        return
    ax.axvline(best_epoch, color=COLOR_BEST_EPOCH, linewidth=1.2, linestyle="--")


def plot_lmmnn_history(history, out_path):
    """Single axis, like the MSE plots: train and val are both the LMMNN
    NLL here (per-row, so the differently-sized train batches and the
    full val set are on a comparable footing) -- not val MSE, which is a
    different quantity used only for early stopping, not for this plot."""
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = np.arange(len(history["train_loss"]))

    ax.plot(epochs, history["train_loss"], color=COLOR_TRAIN, linewidth=2, linestyle="-", label="train")
    ax.plot(epochs, history["val_loss"], color=COLOR_VAL, linewidth=2, linestyle=":", label="val")
    _mark_best_epoch(ax, history["best_epoch"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("LMMNN NLL (per row)")
    ax.set_title("Fusion model (LMMNN loss)", color=COLOR_TEXT, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=COLOR_TEXT_SECONDARY)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def plot_baseline_history(history, out_path):
    """Train and val MSE share units, so one axis is correct here."""
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = np.arange(len(history["train_mse"]))

    ax.plot(epochs, history["train_mse"], color=COLOR_TRAIN, linewidth=2, label="train MSE")
    ax.plot(epochs, history["val_mse"], color=COLOR_VAL, linewidth=2, label="val MSE")
    _mark_best_epoch(ax, history["best_epoch"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Fusion model (MSE-only baseline)", color=COLOR_TEXT, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=COLOR_TEXT_SECONDARY)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def plot_modality_comparison(histories, out_path):
    """One figure, one axis (all three share MSE units): color = model
    (image/phenotype/fusion), line style = train (solid) vs val (dotted)."""
    fig, ax = plt.subplots(figsize=(7.5, 5))

    for name, history in histories.items():
        color = MODEL_COLORS[name]
        epochs = np.arange(len(history["train_mse"]))
        ax.plot(epochs, history["train_mse"], color=color, linewidth=2, linestyle="-")
        ax.plot(epochs, history["val_mse"], color=color, linewidth=2, linestyle=":")

    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Image vs. phenotype vs. fusion (MSE-only)", color=COLOR_TEXT, fontsize=12, loc="left")
    _style_axis(ax)

    color_handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS[name], linewidth=2, label=name)
        for name in histories
    ]
    style_handles = [
        plt.Line2D([0], [0], color=COLOR_TEXT_SECONDARY, linewidth=2, linestyle="-", label="train"),
        plt.Line2D([0], [0], color=COLOR_TEXT_SECONDARY, linewidth=2, linestyle=":", label="val"),
    ]
    legend1 = ax.legend(handles=color_handles, loc="upper right", frameon=False, fontsize=9,
                         labelcolor=COLOR_TEXT_SECONDARY, title="model", title_fontsize=9)
    ax.add_artist(legend1)
    ax.legend(handles=style_handles, loc="upper right", bbox_to_anchor=(1, 0.72), frameon=False,
              fontsize=9, labelcolor=COLOR_TEXT_SECONDARY, title="split", title_fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def grouped_split(df, test_size=0.2, seed=SEED):
    """Returns (keep_idx, held_out_idx), grouped by participant_id so a
    subject's rows never straddle the split. held_out_idx has size
    ~test_size of the groups."""
    groups = df["participant_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    keep_idx, held_out_idx = next(splitter.split(df, groups=groups))
    return keep_idx, held_out_idx


def scale_columns(train_df, other_dfs, columns):
    # fit on a bare ndarray (not the DataFrame) so the scaler doesn't track
    # feature names -- avoids a "does not have valid feature names" warning
    # later when checkpoint.predict() calls transform() on plain arrays.
    scaler = StandardScaler().fit(train_df[columns].to_numpy())
    train_df = train_df.copy()
    train_df[columns] = scaler.transform(train_df[columns].to_numpy())
    scaled_others = []
    for df in other_dfs:
        df = df.copy()
        df[columns] = scaler.transform(df[columns].to_numpy())
        scaled_others.append(df)
    return train_df, scaled_others, scaler


def masked_mse(preds, targets):
    mask = ~torch.isnan(targets)
    diff = (preds - targets)[mask]
    return (diff ** 2).mean()


def full_batch_tensors(dataset):
    image = torch.from_numpy(dataset.image)
    phenotype = torch.from_numpy(dataset.phenotype)
    covariates = torch.from_numpy(dataset.covariates)
    targets = torch.from_numpy(dataset.targets)
    return image, phenotype, covariates, targets


def assert_subject_grouping(batches, subject_ids):
    """Every subject's full set of row-indices in `subject_ids` must land
    entirely within a single batch -- the precondition the LMMNN loss
    depends on to build per-subject covariance blocks at all."""
    subject_to_indices = {}
    for idx, subject in enumerate(subject_ids):
        subject_to_indices.setdefault(subject, []).append(idx)
    index_to_batch = {}
    for b, batch in enumerate(batches):
        for idx in batch:
            index_to_batch[idx] = b
    for subject, indices in subject_to_indices.items():
        batch_ids = {index_to_batch[i] for i in indices}
        assert len(batch_ids) == 1, (
            f"subject {subject} split across batches {batch_ids} -- "
            "SubjectGroupedBatchSampler is broken"
        )


def residual_icc(residuals, subject_ids):
    """One-way ANOVA ICC(1): fraction of residual variance that's between-
    subject rather than within-subject, for subjects with >=2 observations.
    This is the textbook definition of "expected correlation between two
    same-subject residuals" under a compound-symmetry model -- exactly what
    the LMMNN loss is trying to explain away."""
    groups = {}
    for r, s in zip(residuals, subject_ids):
        groups.setdefault(s, []).append(r)
    groups = {s: np.array(v) for s, v in groups.items() if len(v) >= 2}
    if len(groups) < 2:
        return float("nan")

    all_r = np.concatenate(list(groups.values()))
    grand_mean = all_r.mean()
    N = len(all_r)
    k = len(groups)

    ss_between = sum(len(v) * (v.mean() - grand_mean) ** 2 for v in groups.values())
    ss_within = sum(((v - v.mean()) ** 2).sum() for v in groups.values())
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (N - k)

    sum_n2 = sum(len(v) ** 2 for v in groups.values())
    n0 = (N - sum_n2 / N) / (k - 1)

    denom = ms_between + (n0 - 1) * ms_within
    if denom == 0:
        return float("nan")
    return (ms_between - ms_within) / denom


def train_lmmnn(train_ds, val_ds, image_dim, pheno_dim, n_covariates, n_targets):
    torch.manual_seed(SEED)
    model = FusionRegressor(image_dim, pheno_dim, n_covariates, n_targets=n_targets, **config.FUSION_ARCHITECTURE)
    loss_fn = LMMNNLoss(n_targets=n_targets)
    optimizer = torch.optim.Adam(
        itertools.chain(model.parameters(), loss_fn.parameters()),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    sampler = SubjectGroupedBatchSampler(train_ds.subject_ids, batch_size=config.FUSION_BATCH_SIZE, seed=SEED)
    loader = DataLoader(train_ds, batch_sampler=sampler)

    val_image, val_pheno, val_cov, val_targets = full_batch_tensors(val_ds)

    best_loss, best_state, patience_left = float("inf"), None, config.PATIENCE
    grouping_checked = False
    history = {"train_loss": [], "val_loss": [], "val_mse": [], "best_epoch": None}

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        epoch_batches = []
        batch_losses = []
        for image_x, pheno_x, cov, targets, subject_ids in loader:
            epoch_batches.append(subject_ids)

            optimizer.zero_grad()
            preds = model(image_x, pheno_x, cov)
            loss, per_target = loss_fn(preds, targets, subject_ids)

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite LMMNN loss at epoch {epoch}: {loss.item()}")
            for info in per_target:
                if info is None:
                    continue
                if not (np.isfinite(info["sigma2_subject"]) and np.isfinite(info["sigma2_error"])):
                    raise RuntimeError(f"non-finite variance scalar at epoch {epoch}: {info}")

            loss.backward()
            optimizer.step()
            # per-row NLL so batches of different sizes (and the full-batch
            # val NLL below) are on a comparable footing for plotting.
            batch_losses.append(loss.item() / len(subject_ids))

        if not grouping_checked:
            batches_as_indices = list(sampler)
            assert_subject_grouping(batches_as_indices, train_ds.subject_ids)
            grouping_checked = True
            print("[check] subject-grouped batching verified: no subject split across batches")

        model.eval()
        with torch.no_grad():
            val_preds = model(val_image, val_pheno, val_cov)
            val_loss = masked_mse(val_preds, val_targets).item()
            val_nll, _ = loss_fn(val_preds, val_targets, val_ds.subject_ids)

        history["train_loss"].append(np.mean(batch_losses))
        history["val_loss"].append(val_nll.item() / len(val_ds.subject_ids))
        history["val_mse"].append(val_loss)

        if val_loss < best_loss:
            best_loss, best_state, patience_left = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, config.PATIENCE
            history["best_epoch"] = epoch
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return model, loss_fn, history


def train_baseline_mse(train_ds, val_ds, image_dim, pheno_dim, n_covariates, n_targets):
    """Same architecture, plain MSE, no subject grouping -- the comparison
    point for the residual-correlation sanity check."""
    torch.manual_seed(SEED)
    model = FusionRegressor(image_dim, pheno_dim, n_covariates, n_targets=n_targets, **config.FUSION_ARCHITECTURE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    train_image, train_pheno, train_cov, train_targets = full_batch_tensors(train_ds)
    val_image, val_pheno, val_cov, val_targets = full_batch_tensors(val_ds)

    best_loss, best_state, patience_left = float("inf"), None, config.PATIENCE
    history = {"train_mse": [], "val_mse": [], "best_epoch": None}
    for epoch in range(BASELINE_MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        preds = model(train_image, train_pheno, train_cov)
        loss = masked_mse(preds, train_targets)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = masked_mse(model(val_image, val_pheno, val_cov), val_targets).item()

        history["train_mse"].append(loss.item())
        history["val_mse"].append(val_loss)

        if val_loss < best_loss:
            best_loss, best_state, patience_left = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, config.PATIENCE
            history["best_epoch"] = epoch
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return model, history


def train_single_modality_mse(model, train_ds, val_ds, modality):
    """Same MSE training loop as train_baseline_mse, but for a
    SingleModalityRegressor over one modality only ('image' or
    'phenotype') -- the ablation comparison for the modality-comparison
    plot."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    train_x = torch.from_numpy(getattr(train_ds, modality))
    train_cov = torch.from_numpy(train_ds.covariates)
    train_targets = torch.from_numpy(train_ds.targets)
    val_x = torch.from_numpy(getattr(val_ds, modality))
    val_cov = torch.from_numpy(val_ds.covariates)
    val_targets = torch.from_numpy(val_ds.targets)

    best_loss, best_state, patience_left = float("inf"), None, config.PATIENCE
    history = {"train_mse": [], "val_mse": [], "best_epoch": None}
    for epoch in range(BASELINE_MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        preds = model(train_x, train_cov)
        loss = masked_mse(preds, train_targets)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = masked_mse(model(val_x, val_cov), val_targets).item()

        history["train_mse"].append(loss.item())
        history["val_mse"].append(val_loss)

        if val_loss < best_loss:
            best_loss, best_state, patience_left = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, config.PATIENCE
            history["best_epoch"] = epoch
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return model, history


def main():
    df, phenotype_cols, covariate_cols, image_cols = load_multimodal_dataframe()
    print(f"N sessions: {len(df)} ({df['participant_id'].nunique()} unique subjects)")

    trainval_idx, test_idx = grouped_split(df, test_size=0.2)
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    train_rel_idx, val_rel_idx = grouped_split(trainval_df, test_size=0.2, seed=SEED + 1)
    train_df = trainval_df.iloc[train_rel_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_rel_idx].reset_index(drop=True)
    raw_test_df = test_df.copy()  # pre-scaling, for the checkpoint round-trip check below
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)} sessions "
          f"({train_df['participant_id'].nunique()}/{val_df['participant_id'].nunique()}/"
          f"{test_df['participant_id'].nunique()} unique subjects)")

    train_df, (val_df, test_df), phenotype_scaler = scale_columns(train_df, [val_df, test_df], phenotype_cols)
    train_df, (val_df, test_df), image_scaler = scale_columns(train_df, [val_df, test_df], image_cols)
    train_df, (val_df, test_df), covariate_scaler = scale_columns(train_df, [val_df, test_df], covariate_cols)
    scalers = {"phenotype": phenotype_scaler, "image": image_scaler, "covariates": covariate_scaler}

    target_cols = config.TARGET_COLUMNS
    train_ds = MultimodalDataset(train_df, phenotype_cols, covariate_cols, image_cols, target_cols)
    val_ds = MultimodalDataset(val_df, phenotype_cols, covariate_cols, image_cols, target_cols)
    test_ds = MultimodalDataset(test_df, phenotype_cols, covariate_cols, image_cols, target_cols)

    n_targets = len(target_cols)
    model, loss_fn, lmmnn_history = train_lmmnn(
        train_ds, val_ds, len(image_cols), len(phenotype_cols), len(covariate_cols), n_targets
    )
    baseline_model, baseline_history = train_baseline_mse(
        train_ds, val_ds, len(image_cols), len(phenotype_cols), len(covariate_cols), n_targets
    )

    torch.manual_seed(SEED)
    image_model = SingleModalityRegressor(
        ImageEmbedder(len(image_cols), embedding_dim=config.IMAGE_EMBEDDING_OUTPUT_DIM),
        config.IMAGE_EMBEDDING_OUTPUT_DIM, len(covariate_cols), n_targets=n_targets,
    )
    image_model, image_history = train_single_modality_mse(image_model, train_ds, val_ds, "image")

    torch.manual_seed(SEED)
    phenotype_model = SingleModalityRegressor(
        PhenotypeEmbedder(len(phenotype_cols), embedding_dim=32),
        32, len(covariate_cols), n_targets=n_targets,
    )
    phenotype_model, phenotype_history = train_single_modality_mse(phenotype_model, train_ds, val_ds, "phenotype")

    config.FUSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lmmnn_plot_path = config.FUSION_OUTPUT_DIR / "loss_curve_lmmnn.png"
    baseline_plot_path = config.FUSION_OUTPUT_DIR / "loss_curve_baseline.png"
    comparison_plot_path = config.FUSION_OUTPUT_DIR / "loss_curve_modality_comparison.png"
    plot_lmmnn_history(lmmnn_history, lmmnn_plot_path)
    plot_baseline_history(baseline_history, baseline_plot_path)
    plot_modality_comparison(
        {"image": image_history, "phenotype": phenotype_history, "fusion": baseline_history},
        comparison_plot_path,
    )
    print(f"\nSaved loss curves to {lmmnn_plot_path}, {baseline_plot_path}, and {comparison_plot_path}")

    print("\n=== Final LMMNN variance scalars per target ===")
    for t, col in enumerate(target_cols):
        s2_subject = torch.exp(loss_fn.log_sigma2_subject[t]).item()
        s2_error = torch.exp(loss_fn.log_sigma2_error[t]).item()
        icc_from_variances = s2_subject / (s2_subject + s2_error)
        print(f"{col}: sigma2_subject={s2_subject:.4f} sigma2_error={s2_error:.4f} "
              f"(implied ICC={icc_from_variances:.3f})")

    print("\n=== Residual within-subject ICC: LMMNN vs. MSE-only baseline (train set) ===")
    train_image, train_pheno, train_cov, train_targets = full_batch_tensors(train_ds)
    model.eval()
    baseline_model.eval()
    with torch.no_grad():
        lmmnn_preds = model(train_image, train_pheno, train_cov)
        baseline_preds = baseline_model(train_image, train_pheno, train_cov)
    for t, col in enumerate(target_cols):
        mask = ~torch.isnan(train_targets[:, t])
        lmmnn_resid = (train_targets[:, t] - lmmnn_preds[:, t])[mask].numpy()
        baseline_resid = (train_targets[:, t] - baseline_preds[:, t])[mask].numpy()
        subjects = train_ds.subject_ids[mask.numpy()]
        print(f"{col}: baseline_ICC={residual_icc(baseline_resid, subjects):.3f} "
              f"lmmnn_ICC={residual_icc(lmmnn_resid, subjects):.3f} (expect a decrease)")

    print("\n=== Held-out test MSE vs. mean-baseline (fixed-effect predictions only) ===")
    test_image, test_pheno, test_cov, test_targets = full_batch_tensors(test_ds)
    with torch.no_grad():
        test_preds = model(test_image, test_pheno, test_cov)
    for t, col in enumerate(target_cols):
        train_mask = ~torch.isnan(train_targets[:, t])
        train_mean = train_targets[:, t][train_mask].mean().item()
        test_mask = ~torch.isnan(test_targets[:, t])
        y_test = test_targets[:, t][test_mask].numpy()
        pred_test = test_preds[:, t][test_mask].numpy()
        model_mse = float(np.mean((y_test - pred_test) ** 2))
        baseline_mse = float(np.mean((y_test - train_mean) ** 2))
        r2 = 1 - model_mse / baseline_mse if baseline_mse > 0 else float("nan")
        print(f"{col}: model_MSE={model_mse:.4f} mean_baseline_MSE={baseline_mse:.4f} "
              f"R2={r2:.3f} (n_test={test_mask.sum().item()})")

    model_kwargs = dict(
        image_input_dim=len(image_cols),
        phenotype_input_dim=len(phenotype_cols),
        n_covariates=len(covariate_cols),
        n_targets=n_targets,
        **config.FUSION_ARCHITECTURE,
    )
    save_checkpoint(
        config.FUSION_CHECKPOINT_PATH, model, scalers, phenotype_cols, covariate_cols, image_cols, target_cols, model_kwargs
    )
    print(f"\nSaved model checkpoint to {config.FUSION_CHECKPOINT_PATH}")

    # Round-trip sanity check: load the checkpoint back and confirm predict()
    # on raw (unscaled) test rows reproduces the in-memory model's predictions.
    from checkpoint import load_checkpoint

    reloaded_model, reloaded_checkpoint = load_checkpoint(config.FUSION_CHECKPOINT_PATH)
    raw_test_preds = predict(
        reloaded_model, reloaded_checkpoint,
        raw_test_df[image_cols].to_numpy(dtype="float64"),
        raw_test_df[phenotype_cols].to_numpy(dtype="float64"),
        raw_test_df[covariate_cols].to_numpy(dtype="float64"),
    )
    max_diff = np.abs(raw_test_preds - test_preds.numpy()).max()
    print(f"[check] checkpoint round-trip max prediction diff: {max_diff:.2e} (should be ~0)")


if __name__ == "__main__":
    main()
