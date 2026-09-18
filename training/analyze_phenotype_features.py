#!/usr/bin/env python3
"""Per-feature analysis of the phenotype embedder-input columns.

Two questions `train_phenotype_sanity.py` doesn't answer, since it only
scores the *whole* feature set: (1) which of the 161 embedder-input columns
actually associate with `ef_composite` on their own, and (2) which pairs of
columns are redundant with each other (multicollinear), before the
Phenotype Feature Extractor's final column list is locked in.

Row spine: same 217-session subset as train_phenotype_sanity.py (225
sessions with a valid ef_composite) for target-association stats, but the
full 225-session feature matrix for the feature-feature redundancy checks
(a property of X alone, independent of the target).

Outputs, alongside this script's --out-dir (default data/processed/):
- feature_target_association.tsv    : one row per embedder-input column,
  Pearson/Spearman r vs. ef_composite, sorted by |Pearson r| descending.
- feature_redundancy_pairs.tsv      : substantive (non-`_was_missing`)
  column pairs with |Pearson r| >= --redundancy-threshold.
- missingness_redundancy_pairs.tsv  : `_was_missing` indicator pairs above
  the same threshold, reported separately -- these reflect which CNB
  batteries/self-report forms tend to be missing *together* in a session,
  a different phenomenon from two value columns measuring the same thing,
  and would otherwise dominate the substantive-pairs list (in practice,
  ~95% of flagged pairs are `_was_missing`/`_was_missing` pairs).
- feature_vif.tsv                   : variance inflation factor per
  substantive continuous/binary column (one-hot dummies and `_was_missing`
  indicators excluded -- both are collinear by construction: one-hot sums
  to 1 within its group, and co-administered batteries' missingness flags
  are near-duplicates, so VIF on them just re-reports the pairwise finding
  above as a wall of `inf`).
"""

import argparse
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

REDUNDANCY_THRESHOLD_DEFAULT = 0.85
ONE_HOT_PREFIXES = tuple(f"{c}_" for c in config.CATEGORICAL_COLUMNS)


def load_target_joined():
    features = pd.read_csv(config.PHENOTYPE_FEATURES_TSV, sep="\t")
    composite = pd.read_csv(config.EF_COMPOSITE_TSV, sep="\t")
    df = features.merge(composite[config.ID_COLS + ["ef_composite"]], on=config.ID_COLS, how="inner")
    return df.dropna(subset=["ef_composite"]).reset_index(drop=True)


def one_hot_group_of(col):
    for prefix in ONE_HOT_PREFIXES:
        if col.startswith(prefix):
            return prefix.rstrip("_")
    return None


def feature_target_association(df, feature_cols):
    rows = []
    y = df["ef_composite"].to_numpy(dtype=float)
    for col in feature_cols:
        x = df[col].to_numpy(dtype=float)
        if np.std(x) == 0:
            pearson_r, pearson_p, spearman_r, spearman_p = np.nan, np.nan, np.nan, np.nan
        else:
            pearson_r, pearson_p = pearsonr(x, y)
            spearman_r, spearman_p = spearmanr(x, y)
        rows.append({
            "column": col,
            "one_hot_group": one_hot_group_of(col),
            "is_missingness_indicator": col.endswith("_was_missing"),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
        })
    out = pd.DataFrame(rows)
    out["abs_pearson_r"] = out["pearson_r"].abs()
    return out.sort_values("abs_pearson_r", ascending=False).drop(columns="abs_pearson_r").reset_index(drop=True)


def feature_redundancy_pairs(df, feature_cols, threshold):
    """Split into substantive pairs and `_was_missing`/`_was_missing` pairs
    -- the two report very different things and mixing them buries the
    substantive findings under co-administration patterns (see module
    docstring)."""
    corr = df[feature_cols].corr(method="pearson")
    substantive_rows, missingness_rows = [], []
    for col_a, col_b in combinations(feature_cols, 2):
        r = corr.loc[col_a, col_b]
        if pd.isna(r) or abs(r) < threshold:
            continue
        both_missingness = col_a.endswith("_was_missing") and col_b.endswith("_was_missing")
        group_a, group_b = one_hot_group_of(col_a), one_hot_group_of(col_b)
        row = {
            "column_a": col_a,
            "column_b": col_b,
            "pearson_r": r,
            "same_one_hot_group": group_a is not None and group_a == group_b,
        }
        (missingness_rows if both_missingness else substantive_rows).append(row)

    def to_sorted_df(rows):
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["abs_pearson_r"] = out["pearson_r"].abs()
        return out.sort_values("abs_pearson_r", ascending=False).drop(columns="abs_pearson_r").reset_index(drop=True)

    return to_sorted_df(substantive_rows), to_sorted_df(missingness_rows)


def variance_inflation_factors(df, columns):
    """VIF_j = 1 / (1 - R^2) from regressing column j on all other columns
    in `columns`, via closed-form OLS (np.linalg.lstsq) -- avoids adding a
    statsmodels dependency for one number per column."""
    X = df[columns].to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    rows = []
    for j, col in enumerate(columns):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        design = np.column_stack([np.ones(len(others)), others])
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        vif = np.inf if not np.isnan(r2) and r2 >= 1.0 else (1 / (1 - r2) if not np.isnan(r2) else np.nan)
        rows.append({"column": col, "r2_vs_other_features": r2, "vif": vif})
    out = pd.DataFrame(rows)
    return out.sort_values("vif", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(config.PROCESSED_DIR))
    parser.add_argument("--redundancy-threshold", type=float, default=REDUNDANCY_THRESHOLD_DEFAULT)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import json
    manifest = json.loads(config.PHENOTYPE_FEATURE_MANIFEST.read_text())
    feature_cols = manifest["embedder_input_columns"]

    full_features = pd.read_csv(config.PHENOTYPE_FEATURES_TSV, sep="\t")
    target_df = load_target_joined()

    print(f"Feature-target association: {len(target_df)} sessions with valid ef_composite, "
          f"{len(feature_cols)} embedder-input columns")
    assoc = feature_target_association(target_df, feature_cols)
    assoc_path = out_dir / "feature_target_association.tsv"
    assoc.to_csv(assoc_path, sep="\t", index=False)
    print(f"Wrote {assoc_path}")
    print("\nTop 15 by |Pearson r| vs. ef_composite:")
    print(assoc.head(15).to_string(index=False))

    print(f"\nFeature-feature redundancy: {len(full_features)} sessions, "
          f"threshold |r| >= {args.redundancy_threshold}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        substantive, missingness = feature_redundancy_pairs(full_features, feature_cols, args.redundancy_threshold)

    substantive_path = out_dir / "feature_redundancy_pairs.tsv"
    substantive.to_csv(substantive_path, sep="\t", index=False)
    print(f"Wrote {substantive_path} ({len(substantive)} substantive pairs)")
    if not substantive.empty:
        print("\nSubstantive redundant pairs:")
        print(substantive.to_string(index=False))

    missingness_path = out_dir / "missingness_redundancy_pairs.tsv"
    missingness.to_csv(missingness_path, sep="\t", index=False)
    print(f"\nWrote {missingness_path} ({len(missingness)} `_was_missing`/`_was_missing` pairs, "
          "reflects co-administered batteries -- not a feature-value redundancy)")

    vif_columns = [
        c for c in feature_cols
        if one_hot_group_of(c) is None
        and not c.endswith("_was_missing")
        and full_features[c].nunique() > 1
    ]
    print(f"\nVariance inflation factors: {len(vif_columns)} substantive columns "
          "(one-hot dummies and `_was_missing` indicators excluded)")
    vif = variance_inflation_factors(full_features, vif_columns)
    vif_path = out_dir / "feature_vif.tsv"
    vif.to_csv(vif_path, sep="\t", index=False)
    print(f"Wrote {vif_path}")
    print("\nTop 15 by VIF:")
    print(vif.head(15).to_string(index=False))
    print("\nRule of thumb: VIF > 5 = moderate collinearity, VIF > 10 = severe.")


if __name__ == "__main__":
    main()
