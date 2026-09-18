#!/usr/bin/env python3
"""Score the global model saved by the NVFlare server on the held-out test set.

The model takes RAW features (global scaling is inside it), so no scaler is
needed here.

    python federated/flare/evaluate_global.py \
        --model /tmp/nvflare/cogniwarriors/cogniwarriors_fedavg/server/simulate_job/app_server/best_FL_global_model.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
os.environ.setdefault("NBBH_DATA_ROOT", str(REPO / "data"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch

from cogni_model import FedFusionModel


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="FL_global_model.pt or best_FL_global_model.pt")
    p.add_argument("--test-dir", type=Path, default=HERE / "data" / "test")
    args = p.parse_args()

    columns = json.loads((args.test_dir / "columns.json").read_text())
    test = pd.read_csv(args.test_dir / "test.csv")

    saved = torch.load(args.model, map_location="cpu", weights_only=False)
    state = {k: torch.as_tensor(v) for k, v in saved.get("model", saved).items()}
    model = FedFusionModel(
        len(columns["image_cols"]), len(columns["phenotype_cols"]), len(columns["covariate_cols"]),
        n_targets=len(columns["target_cols"]), use_lmmnn=any(k.startswith("lmmnn.") for k in state),
    )
    model.load_state_dict(state)
    model.eval()

    as_tensor = lambda cols: torch.tensor(test[cols].to_numpy(np.float32))
    with torch.no_grad():
        preds = model(as_tensor(columns["image_cols"]), as_tensor(columns["phenotype_cols"]),
                      as_tensor(columns["covariate_cols"])).numpy()

    print(f"Global model: {args.model}")
    print(f"Test set: {len(test)} sessions, {test['participant_id'].nunique()} participants")
    for t, col in enumerate(columns["target_cols"]):
        y = test[col].to_numpy()
        mask = ~np.isnan(y)
        err = preds[mask, t] - y[mask]
        mse, mae = float(np.mean(err ** 2)), float(np.mean(np.abs(err)))
        r2 = 1 - mse / float(np.var(y[mask]))
        print(f"{col}: MSE={mse:.4f}  MAE={mae:.4f}  R²={r2:.3f}")
    if model.lmmnn is not None:
        print(f"LMMNN variances: sigma2_subject={torch.exp(model.lmmnn.log_sigma2_subject).tolist()} "
              f"sigma2_error={torch.exp(model.lmmnn.log_sigma2_error).tolist()}")


if __name__ == "__main__":
    main()
