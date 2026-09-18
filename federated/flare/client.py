"""NVFlare client: what runs inside each hospital (site).

Uses the NVFlare Client API. The site reads ONLY its own folder:
    $SITE_DATA_DIR                      (set this in production), or
    <--data-root>/<site name>           (simulator / POC, e.g. data/site-1)

Each message from the server is one of two things:
  * phase "stats" (once, before round 1): reply with column sums, sums of
    squares and the row count of the local TRAIN data -- no rows leave the site
  * a normal FedAvg round: load the global model, score it on local
    validation data, train locally, send back weights + number of sessions
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel

import config
from cogni_model import BLOCKS, FedFusionModel
from multimodal_data import MultimodalDataset, SubjectGroupedBatchSampler

torch.set_num_threads(1)  # tiny model; avoids thread oversubscription with many sites


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data", help="folder holding one sub-folder per site")
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--variance-lr", type=float, default=0.02, help="lr for the LMMNN variance scalars")
    p.add_argument("--mu", type=float, default=0.0, help="FedProx strength; 0 = plain FedAvg")
    p.add_argument("--loss", choices=["mse", "lmmnn"], default="mse")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_split(data_dir, split, columns):
    df = pd.read_csv(data_dir / f"{split}.csv")
    return MultimodalDataset(df, columns["phenotype_cols"], columns["covariate_cols"],
                             columns["image_cols"], columns["target_cols"])


def tensors(ds):
    return [torch.from_numpy(np.ascontiguousarray(a)) for a in (ds.image, ds.phenotype, ds.covariates, ds.targets)]


def evaluate(model, ds):
    """MSE / MAE of `model` on a dataset (NaN targets ignored)."""
    image, pheno, cov, y = tensors(ds)
    model.eval()
    with torch.no_grad():
        diff = model(image, pheno, cov) - y
    diff = diff[~torch.isnan(y)]
    return float((diff ** 2).mean()), float(diff.abs().mean())


def local_stats(ds):
    """Per-block column sums / sums of squares / count. This is all the server
    needs to build a global mean and std."""
    stats = {"n": torch.tensor([float(len(ds))], dtype=torch.float64)}
    for block, arr in zip(BLOCKS, (ds.image, ds.phenotype, ds.covariates)):
        x = torch.from_numpy(np.asarray(arr, dtype=np.float64))
        stats[f"{block}_sum"] = x.sum(0)
        stats[f"{block}_sumsq"] = (x ** 2).sum(0)
    return stats


def masked_mse(preds, targets):
    mask = ~torch.isnan(targets)
    return ((preds - targets)[mask] ** 2).mean()


def main():
    args = parse_args()
    flare.init()
    site = flare.get_site_name()
    data_dir = Path(os.environ.get("SITE_DATA_DIR") or Path(args.data_root) / site)
    columns = json.loads((data_dir / "columns.json").read_text())
    train_ds, val_ds = load_split(data_dir, "train", columns), load_split(data_dir, "val", columns)
    print(f"[{site}] {len(train_ds)} train / {len(val_ds)} val sessions from {data_dir}")

    model = FedFusionModel(
        len(columns["image_cols"]), len(columns["phenotype_cols"]), len(columns["covariate_cols"]),
        n_targets=len(columns["target_cols"]), use_lmmnn=args.loss == "lmmnn",
    )
    # Adam state stays at the site and carries over between rounds
    groups = [{"params": model.net.parameters(), "weight_decay": config.WEIGHT_DECAY}]
    if model.lmmnn is not None:
        groups.append({"params": model.lmmnn.parameters(), "weight_decay": 0.0, "lr": args.variance_lr})
    optimizer = torch.optim.Adam(groups, lr=args.lr)

    while flare.is_running():
        input_model = flare.receive()
        meta = input_model.meta or {}

        # ---- phase 0: federated feature statistics ----
        if meta.get("phase") == "stats":
            flare.send(FLModel(params=local_stats(train_ds), meta={"phase": "stats"}))
            print(f"[{site}] sent feature statistics (sums and counts only)")
            continue

        model.load_state_dict({k: torch.as_tensor(v) for k, v in input_model.params.items()})

        # score the GLOBAL model on local validation data before training
        global_val_mse, global_val_mae = evaluate(model, val_ds)
        metrics = {"val_mse": global_val_mse, "val_mae": global_val_mae}

        if flare.is_evaluate():  # cross-site evaluation task: metrics only
            flare.send(FLModel(metrics=metrics))
            continue

        # ---- local training ----
        anchor = [p.detach().clone() for p in model.parameters()]  # for FedProx
        round_num = input_model.current_round or 0
        sampler = SubjectGroupedBatchSampler(train_ds.subject_ids, batch_size=config.FUSION_BATCH_SIZE,
                                             seed=args.seed * 10_000 + round_num)
        losses = []
        for _ in range(args.local_epochs):
            model.train()
            for batch in sampler:
                image, pheno, cov, y = (t[batch] for t in tensors(train_ds))
                optimizer.zero_grad()
                preds = model(image, pheno, cov)
                if model.lmmnn is not None:
                    loss = model.lmmnn(preds, y, train_ds.subject_ids[batch])[0]
                else:
                    loss = masked_mse(preds, y)
                if args.mu > 0:
                    loss = loss + 0.5 * args.mu * sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"[{site}] non-finite loss in round {round_num}")
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        metrics["train_loss"] = float(np.mean(losses))
        print(f"[{site}] round {round_num}: global val MSE {global_val_mse:.4f}, local train loss {metrics['train_loss']:.4f}")

        flare.send(FLModel(
            params={k: v.detach().cpu() for k, v in model.state_dict().items()},
            metrics=metrics,
            # FedAvg weight = number of local training sessions
            meta={"NUM_STEPS_CURRENT_ROUND": len(train_ds)},
        ))


if __name__ == "__main__":
    main()
