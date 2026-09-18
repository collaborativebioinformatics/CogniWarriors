#!/usr/bin/env python3
"""FedAvg simulation for the image + phenotype FusionRegressor (fixed effects).

Simulates N hospitals ("sites") on one machine, reusing the real model, data
and loss code from training/. Nothing but model weights and summary
statistics ever crosses a site boundary.

What each round does
    1. hub sends the global weights to every site
    2. each site trains locally for a few epochs on its own sessions
    3. each site sends back its weights + its number of training sessions
    4. hub takes the weighted average (FedAvg) -> new global weights
    5. every site scores the new global model on its own validation data;
       hub keeps the best round (early stopping in rounds, not epochs)

Problems this handles (small data, multi-site)
    * sites are split BY PARTICIPANT, so one person's sessions never sit at
      two hospitals (needed for the LMMNN loss and to avoid leakage)
    * feature scaling uses GLOBAL mean/std built from per-site sums, not a
      separate scaler per site (sites would otherwise disagree on units)
    * FedAvg is weighted by session count, so a 20-session site doesn't
      count as much as a 60-session one
    * optional FedProx term (--mu) to stop tiny sites drifting apart
    * every run is compared against centralized training (all data pooled)
      and local-only training (each site alone), on the SAME held-out test
      set as training/train_fusion_lmmnn.py
    * several seeds, because at N=215 a single run is mostly noise

Usage (from the repo root)
    python federated/fedavg_simulation.py
    python federated/fedavg_simulation.py --split age --mu 0.01
    python federated/fedavg_simulation.py --loss lmmnn
"""

import argparse
import json
import warnings
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO / "training"
os.environ.setdefault("NBBH_DATA_ROOT", str(TRAINING_DIR / "data"))
sys.path.insert(0, str(TRAINING_DIR))

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import config
from checkpoint import save_checkpoint
from fusion_model import FusionRegressor
from lmmnn_loss import LMMNNLoss
from multimodal_data import MultimodalDataset, SubjectGroupedBatchSampler, load_multimodal_dataframe

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")


# --------------------------------------------------------------------------
# 1. Data: global test set + participant-level site assignment
# --------------------------------------------------------------------------

def grouped_split(df, test_size, seed):
    """Same logic/seed as training/train_fusion_lmmnn.py -> identical test set."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(df, groups=df["participant_id"].to_numpy()))


def assign_sites(df, n_sites, mode, seed):
    """Returns one DataFrame per site. Whole participants are assigned.

    iid : participants shuffled at random -> sites look alike
    age : participants sorted by mean age and cut into chunks -> sites
          differ (a realistic non-IID case, e.g. a children's hospital)
    """
    per_subject_age = df.groupby("participant_id")["age"].mean()
    subjects = per_subject_age.index.to_numpy()
    if mode == "iid":
        subjects = np.random.default_rng(seed).permutation(subjects)
    elif mode == "age":
        subjects = per_subject_age.sort_values().index.to_numpy()
    else:
        raise ValueError(f"unknown split mode {mode}")
    chunks = np.array_split(subjects, n_sites)
    return [df[df["participant_id"].isin(c)].reset_index(drop=True) for c in chunks]


def split_site_train_val(site_df, val_frac, seed):
    """Participant-grouped local train/val split, at least 1 val participant."""
    subjects = np.random.default_rng(seed).permutation(site_df["participant_id"].unique())
    n_val = max(1, int(round(val_frac * len(subjects))))
    val_subjects = set(subjects[:n_val])
    is_val = site_df["participant_id"].isin(val_subjects)
    return site_df[~is_val].reset_index(drop=True), site_df[is_val].reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Federated feature scaling (sites share sums, never rows)
# --------------------------------------------------------------------------

def local_stats(site_train_df, cols):
    x = site_train_df[cols].to_numpy(dtype=np.float64)
    return {"n": len(x), "sum": x.sum(0), "sumsq": (x ** 2).sum(0)}


def combine_stats(stats_list):
    n = sum(s["n"] for s in stats_list)
    mean = sum(s["sum"] for s in stats_list) / n
    var = np.maximum(sum(s["sumsq"] for s in stats_list) / n - mean ** 2, 0.0)
    std = np.sqrt(var)
    std[std < 1e-8] = 1.0  # constant column -> leave unscaled
    return mean, var, std, n


def make_sklearn_scaler(mean, var, std, n):
    """Wrap global stats in a StandardScaler so training/checkpoint.predict works."""
    scaler = StandardScaler()
    scaler.mean_, scaler.var_, scaler.scale_ = mean, var, std
    scaler.n_features_in_, scaler.n_samples_seen_ = len(mean), n
    return scaler


def apply_scaling(df, cols, mean, std):
    df = df.copy()
    df[cols] = (df[cols].to_numpy(dtype=np.float64) - mean) / std
    return df


# --------------------------------------------------------------------------
# 3. Model = FusionRegressor (+ the 2 LMMNN variance scalars if used)
# --------------------------------------------------------------------------

class FedModel(nn.Module):
    """One object whose state_dict is exactly what travels to the hub."""

    def __init__(self, dims, n_targets, use_lmmnn):
        super().__init__()
        self.net = FusionRegressor(*dims, n_targets=n_targets, **config.FUSION_ARCHITECTURE)
        self.lmmnn = LMMNNLoss(n_targets) if use_lmmnn else None

    def loss(self, preds, targets, subject_ids):
        if self.lmmnn is not None:
            return self.lmmnn(preds, targets, subject_ids)[0]
        return masked_mse(preds, targets)


def masked_mse(preds, targets):
    mask = ~torch.isnan(targets)
    return ((preds - targets)[mask] ** 2).mean()


def sq_error_sum(net, ds):
    """(sum of squared errors, count) -- summable across sites."""
    net.eval()
    with torch.no_grad():
        preds = net(*(torch.from_numpy(a) for a in (ds.image, ds.phenotype, ds.covariates)))
    targets = torch.from_numpy(ds.targets)
    mask = ~torch.isnan(targets)
    return float(((preds - targets)[mask] ** 2).sum()), int(mask.sum())


# --------------------------------------------------------------------------
# 4. Site (client) and hub (server) logic
# --------------------------------------------------------------------------

class Site:
    """Everything here stays at the hospital: data, optimizer state."""

    def __init__(self, name, train_ds, val_ds, args):
        self.name, self.train_ds, self.val_ds, self.args = name, train_ds, val_ds, args
        self.n_train = len(train_ds)
        self.optimizer_state = None  # Adam moments never leave the site

    def train_round(self, model, global_state, round_idx, seed):
        """Load global weights, train locally, return (weights, n_train, loss)."""
        model.load_state_dict(global_state)
        anchor = [p.detach().clone() for p in model.parameters()]  # for FedProx
        # weight decay on the network only: decaying the LMMNN log-variances
        # would pull both towards log(1)=0, i.e. force sigma2 -> 1
        groups = [{"params": model.net.parameters(), "weight_decay": config.WEIGHT_DECAY}]
        if model.lmmnn is not None:
            groups.append({"params": model.lmmnn.parameters(), "weight_decay": 0.0, "lr": self.args.variance_lr})
        opt = torch.optim.Adam(groups, lr=self.args.lr)
        if self.optimizer_state is not None:
            opt.load_state_dict(self.optimizer_state)

        sampler = SubjectGroupedBatchSampler(
            self.train_ds.subject_ids, batch_size=config.FUSION_BATCH_SIZE, seed=seed * 10_000 + round_idx
        )
        loader = DataLoader(self.train_ds, batch_sampler=sampler)
        losses = []
        for _ in range(self.args.local_epochs):
            model.train()
            for image_x, pheno_x, cov, targets, subject_ids in loader:
                opt.zero_grad()
                loss = model.loss(model.net(image_x, pheno_x, cov), targets, np.asarray(subject_ids))
                if self.args.mu > 0:
                    prox = sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))
                    loss = loss + 0.5 * self.args.mu * prox
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{self.name}: non-finite loss in round {round_idx}")
                loss.backward()
                opt.step()
                losses.append(loss.item())

        self.optimizer_state = opt.state_dict()
        weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return weights, self.n_train, float(np.mean(losses))

    def evaluate(self, model, global_state):
        model.load_state_dict(global_state)
        return sq_error_sum(model.net, self.val_ds)


def fedavg(updates):
    """Weighted average of state_dicts. updates = [(state_dict, n_samples), ...]."""
    total = sum(n for _, n in updates)
    avg = {}
    for key in updates[0][0]:
        if torch.is_floating_point(updates[0][0][key]):
            avg[key] = sum(sd[key] * (n / total) for sd, n in updates)
        else:  # integer buffers (none today) -- just take one
            avg[key] = updates[0][0][key].clone()
    return avg


def run_federation(sites, dims, n_targets, args, seed, verbose=False):
    """The hub loop. With one site and local_epochs=1 this is ordinary
    centralized training, which is how the baselines below are produced."""
    torch.manual_seed(seed)
    model = FedModel(dims, n_targets, use_lmmnn=args.loss == "lmmnn")
    global_state = {k: v.clone() for k, v in model.state_dict().items()}

    best = {"val_mse": float("inf"), "state": global_state, "round": -1}
    history, patience_left = [], args.patience
    for r in range(args.rounds):
        updates, train_losses = [], []
        for site in sites:
            weights, n, loss = site.train_round(model, global_state, r, seed)
            updates.append((weights, n))
            train_losses.append(loss)
        global_state = fedavg(updates)

        per_site = [site.evaluate(model, global_state) for site in sites]
        val_mse = sum(s for s, _ in per_site) / sum(c for _, c in per_site)
        history.append({
            "round": r + 1,
            "val_mse": val_mse,
            "site_val_mse": {site.name: s / c for site, (s, c) in zip(sites, per_site)},
            "site_train_loss": {site.name: l for site, l in zip(sites, train_losses)},
        })
        if verbose:
            print(f"  round {r + 1:3d}  val MSE {val_mse:.4f}")

        if val_mse < best["val_mse"]:
            best = {"val_mse": val_mse, "state": {k: v.clone() for k, v in global_state.items()}, "round": r + 1}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best["state"])
    return model, best, history


# --------------------------------------------------------------------------
# 5. Experiment driver
# --------------------------------------------------------------------------

def test_metrics(model, test_ds, train_mean):
    sse, count = sq_error_sum(model.net, test_ds)
    mse = sse / count
    y = test_ds.targets[~np.isnan(test_ds.targets)]
    baseline_mse = float(np.mean((y - train_mean) ** 2))
    return {"test_mse": mse, "test_r2": 1 - mse / baseline_mse}


def build_everything(args):
    df, pheno_cols, cov_cols, image_cols = load_multimodal_dataframe()
    targets = config.TARGET_COLUMNS
    trainval_idx, test_idx = grouped_split(df, test_size=0.2, seed=0)
    trainval_df, test_df = df.iloc[trainval_idx], df.iloc[test_idx].reset_index(drop=True)

    site_dfs = [split_site_train_val(s, args.val_frac, args.split_seed + i)
                for i, s in enumerate(assign_sites(trainval_df, args.n_sites, args.split, args.split_seed))]

    # federated scaling: each block's global mean/std from per-site sums
    scalers, scaled_sites = {}, [[tr, va] for tr, va in site_dfs]
    for block, cols in (("image", image_cols), ("phenotype", pheno_cols), ("covariates", cov_cols)):
        mean, var, std, n = combine_stats([local_stats(tr, cols) for tr, _ in site_dfs])
        scalers[block] = make_sklearn_scaler(mean, var, std, n)
        scaled_sites = [[apply_scaling(d, cols, mean, std) for d in pair] for pair in scaled_sites]
        test_df = apply_scaling(test_df, cols, mean, std)

    make_ds = lambda d: MultimodalDataset(d, pheno_cols, cov_cols, image_cols, targets)
    site_data = [(f"site-{i + 1}", make_ds(tr), make_ds(va)) for i, (tr, va) in enumerate(scaled_sites)]

    # pooled train-target mean (federated: sites send sum + count) for the R² baseline
    y_sum = sum(np.nansum(tr.targets) for _, tr, _ in site_data)
    y_n = sum(np.sum(~np.isnan(tr.targets)) for _, tr, _ in site_data)

    dims = (len(image_cols), len(pheno_cols), len(cov_cols))
    columns = {"image_cols": image_cols, "phenotype_cols": pheno_cols, "covariate_cols": cov_cols, "target_cols": targets}
    return site_data, make_ds(test_df), dims, len(targets), scalers, y_sum / y_n, columns


def pooled_site(site_data, args):
    """All sites' data in one place -- only for the centralized reference."""
    def cat(field, split):
        return np.concatenate([getattr(d[split], field) for d in site_data])
    pooled = []
    for split in (1, 2):
        ds = object.__new__(MultimodalDataset)
        for field in ("subject_ids", "image", "phenotype", "covariates", "targets"):
            setattr(ds, field, cat(field, split))
        pooled.append(ds)
    return Site("pooled", pooled[0], pooled[1], args)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-sites", type=int, default=4)
    p.add_argument("--split", choices=["iid", "age"], default="iid", help="how participants go to sites")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.2, help="fraction of each site's participants for local val")
    p.add_argument("--loss", choices=["mse", "lmmnn"], default="mse")
    p.add_argument("--rounds", type=int, default=150)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=20, help="rounds without val improvement before stopping")
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--mu", type=float, default=0.0, help="FedProx strength; 0 = plain FedAvg")
    p.add_argument("--variance-lr", type=float, default=0.02, help="lr for the 2 LMMNN variance scalars")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", type=Path, default=REPO / "results" / "fedavg")
    args = p.parse_args()

    site_data, test_ds, dims, n_targets, scalers, train_mean, columns = build_everything(args)
    print(f"{args.n_sites} sites ({args.split} split), loss={args.loss}, mu={args.mu}")
    for name, tr, va in site_data:
        print(f"  {name}: {len(tr)} train / {len(va)} val sessions "
              f"({len(set(tr.subject_ids))}/{len(set(va.subject_ids))} participants)")
    print(f"  global test set: {len(test_ds)} sessions\n")

    results = {"federated": [], "centralized": [], "local_only": []}
    best_fed = None
    for seed in args.seeds:
        sites = [Site(n, tr, va, args) for n, tr, va in site_data]
        fed_model, fed_best, fed_hist = run_federation(sites, dims, n_targets, args, seed)
        fed = {**test_metrics(fed_model, test_ds, train_mean), "best_round": fed_best["round"]}
        results["federated"].append(fed)
        if best_fed is None or fed_best["val_mse"] < best_fed[1]["val_mse"]:
            best_fed = (fed_model, fed_best, fed_hist, seed)

        # centralized reference: one "site" with all data, 1 epoch per "round"
        central_args = argparse.Namespace(**{**vars(args), "local_epochs": 1, "mu": 0.0,
                                             "rounds": config.MAX_EPOCHS, "patience": config.PATIENCE})
        c_model, _, _ = run_federation([pooled_site(site_data, central_args)], dims, n_targets, central_args, seed)
        results["centralized"].append(test_metrics(c_model, test_ds, train_mean))

        # local-only reference: each hospital trains alone
        local = []
        for n, tr, va in site_data:
            l_model, _, _ = run_federation([Site(n, tr, va, central_args)], dims, n_targets, central_args, seed)
            local.append(test_metrics(l_model, test_ds, train_mean))
        results["local_only"].append({k: float(np.mean([m[k] for m in local])) for k in ("test_mse", "test_r2")})

        print(f"seed {seed}: federated R²={fed['test_r2']:.3f}  "
              f"centralized R²={results['centralized'][-1]['test_r2']:.3f}  "
              f"local-only (avg) R²={results['local_only'][-1]['test_r2']:.3f}")

    summary = {}
    print(f"\n=== Test set, mean ± std over {len(args.seeds)} seeds ===")
    for setting, runs in results.items():
        mse, r2 = np.array([r["test_mse"] for r in runs]), np.array([r["test_r2"] for r in runs])
        summary[setting] = {"mse_mean": mse.mean(), "mse_std": mse.std(), "r2_mean": r2.mean(), "r2_std": r2.std()}
        print(f"{setting:12s}  MSE {mse.mean():.4f} ± {mse.std():.4f}   R² {r2.mean():.3f} ± {r2.std():.3f}")

    # save: summary, per-round history of the best federated run, and a
    # checkpoint that training/checkpoint.py can load and predict() with
    args.out.mkdir(parents=True, exist_ok=True)
    model, best, history, seed = best_fed
    with open(args.out / "summary.json", "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "summary": summary,
                   "runs": results, "best_seed": seed, "best_round": best["round"]}, f, indent=2, default=float)
    with open(args.out / "round_history.json", "w") as f:
        json.dump(history, f, indent=2)
    model_kwargs = dict(image_input_dim=dims[0], phenotype_input_dim=dims[1], n_covariates=dims[2],
                        n_targets=n_targets, **config.FUSION_ARCHITECTURE)
    save_checkpoint(args.out / "fedavg_global.pt", model.net, scalers, model_kwargs=model_kwargs, **columns)
    if model.lmmnn is not None:
        s2s, s2e = torch.exp(model.lmmnn.log_sigma2_subject), torch.exp(model.lmmnn.log_sigma2_error)
        print(f"\nglobal LMMNN variances: sigma2_subject={s2s.tolist()} sigma2_error={s2e.tolist()}")
    print(f"\nSaved to {args.out}/ (summary.json, round_history.json, fedavg_global.pt)")


if __name__ == "__main__":
    main()
