#!/usr/bin/env python3
"""Split the multimodal dataset into one folder per NVFlare site.

Each site folder contains ONLY that site's rows (raw, unscaled features):

    <out>/site-1/train.csv, val.csv, columns.json
    <out>/site-2/...
    <out>/test/test.csv, columns.json      <- held-out test set

Rules:
  * whole participants go to one site (never split across sites)
  * the test set is the same 20% of participants as
    training/train_fusion_lmmnn.py, so results are directly comparable
  * each site's own validation set is also split by participant

In a real deployment each hospital already has its own data; this script
only simulates that from the pooled dataset in the repo.

    python federated/flare/prepare_site_data.py --n-sites 4
    python federated/flare/prepare_site_data.py --n-sites 4 --split age
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

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import config
from multimodal_data import load_multimodal_dataframe


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-sites", type=int, default=4)
    p.add_argument("--split", choices=["iid", "age"], default="iid",
                   help="iid = random participants per site; age = sites differ by age group")
    p.add_argument("--val-frac", type=float, default=0.2, help="fraction of each site's participants for validation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=HERE / "data")
    args = p.parse_args()

    df, pheno_cols, cov_cols, image_cols = load_multimodal_dataframe()
    columns = {"image_cols": image_cols, "phenotype_cols": pheno_cols,
               "covariate_cols": cov_cols, "target_cols": config.TARGET_COLUMNS}
    keep = config.ID_COLS + image_cols + pheno_cols + cov_cols + config.TARGET_COLUMNS

    # same held-out test participants as the centralized training script
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    trainval_idx, test_idx = next(splitter.split(df, groups=df["participant_id"]))
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]

    # participants -> sites
    subject_age = trainval.groupby("participant_id")["age"].mean()
    if args.split == "iid":
        subjects = np.random.default_rng(args.seed).permutation(subject_age.index.to_numpy())
    else:
        subjects = subject_age.sort_values().index.to_numpy()

    args.out.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(np.array_split(subjects, args.n_sites), start=1):
        site_df = trainval[trainval["participant_id"].isin(chunk)]
        site_subjects = np.random.default_rng(args.seed + i).permutation(site_df["participant_id"].unique())
        val_subjects = set(site_subjects[: max(1, round(args.val_frac * len(site_subjects)))])
        is_val = site_df["participant_id"].isin(val_subjects)

        site_dir = args.out / f"site-{i}"
        site_dir.mkdir(exist_ok=True)
        site_df.loc[~is_val, keep].to_csv(site_dir / "train.csv", index=False)
        site_df.loc[is_val, keep].to_csv(site_dir / "val.csv", index=False)
        (site_dir / "columns.json").write_text(json.dumps(columns, indent=2))
        print(f"site-{i}: {(~is_val).sum()} train / {is_val.sum()} val sessions "
              f"({len(chunk)} participants)")

    test_dir = args.out / "test"
    test_dir.mkdir(exist_ok=True)
    test[keep].to_csv(test_dir / "test.csv", index=False)
    (test_dir / "columns.json").write_text(json.dumps(columns, indent=2))
    print(f"test : {len(test)} sessions -> {test_dir}")


if __name__ == "__main__":
    main()
