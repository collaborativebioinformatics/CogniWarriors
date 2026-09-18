#!/usr/bin/env python3
"""Build the wide per-session phenotype feature matrix for the embedder.

Row spine: the same 225 (participant_id, session_id) pairs as
build_ef_composite.py, derived from the downloaded sessions.tsv files.

Merges, in order: participants.tsv (subject-level, broadcast to every
session; baseline-only age/age_months dropped), per-session age +
derived session_index from sessions.tsv, every non-EF-cognition CNB task
(config.NON_EF_COGNITION_TASKS, gated by valid_code same as the EF
composite), every self-report scale (config.SELF_REPORT_SCALES), and a
sex-coalesced Tanner staging summary. config.EF_TASKS (the 5 files used by
build_ef_composite.py) are never touched here -- excluding whole files, not
just their primary column, avoids leaking EF-correlated columns like
cnb_pcet_cat through the back door.

Missing-value handling: any embedder-input column missing in more than
config.MISSING_DROP_THRESHOLD of the 225 sessions is dropped outright, then
every remaining column is imputed (median for numeric, mode for
categorical -- imputed BEFORE one-hot encoding) with a paired
`<col>_was_missing` indicator column.
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

# 225 rows / ~160 columns is tiny -- the column-by-column indicator
# assignments below trip pandas' fragmentation warning, but it's a
# performance-only concern that doesn't apply at this scale.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

SESSION_NUM_RE = re.compile(r"ses-(\d+)")


def load_session_spine():
    rows = []
    for path in sorted(config.SESSIONS_DIR.glob("sub-*_sessions.tsv")):
        participant_id = path.name.removesuffix("_sessions.tsv")
        df = pd.read_csv(path, sep="\t", na_values=["n/a"])
        df = df[["session_id"] + config.SESSIONS_ALLOWLIST_COLUMNS].copy()
        df["participant_id"] = participant_id
        df["_session_num"] = df["session_id"].str.extract(SESSION_NUM_RE).astype(int)
        df["session_index"] = df["_session_num"].rank(method="first").astype(int)
        rows.append(df.drop(columns="_session_num"))
    spine = pd.concat(rows, ignore_index=True)
    return spine[config.ID_COLS + ["session_index"] + config.SESSIONS_ALLOWLIST_COLUMNS]


def load_participants():
    df = pd.read_csv(config.PARTICIPANTS_TSV, sep="\t", na_values=["n/a"])
    return df.drop(columns=config.PARTICIPANTS_DROP_COLUMNS)


def load_gated_task(stem, spec):
    path = config.PHENOTYPE_DIR / f"{stem}.tsv"
    df = pd.read_csv(path, sep="\t", na_values=["n/a"])
    valid = df[spec["valid_code_column"]].isin(config.VALID_CODES_OK)
    out = df[config.ID_COLS].copy()
    for col in spec["columns"]:
        out[col] = pd.to_numeric(df[col], errors="coerce").where(valid)
    return out


def load_self_report(stem, columns):
    path = config.PHENOTYPE_DIR / f"{stem}.tsv"
    df = pd.read_csv(path, sep="\t", na_values=["n/a"])
    out = df[config.ID_COLS].copy()
    for col in columns:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return out


def load_tanner():
    """Sex-coalesced Tanner summary: off-sex rows are already all-NaN in the
    source files, so combine_first naturally picks whichever form applies."""
    summaries = {}
    for stem, item_prefix in config.TANNER_ITEM_PREFIXES.items():
        path = config.PHENOTYPE_DIR / f"{stem}.tsv"
        df = pd.read_csv(path, sep="\t", na_values=["n/a"])
        item_cols = [c for c in df.columns if c.startswith(item_prefix)]
        out = df[config.ID_COLS].copy()
        out["tanner_mean_stage"] = df[item_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        complete_col = config.TANNER_COMPLETE_COLUMN[stem]
        out["tanner_complete"] = pd.to_numeric(df[complete_col], errors="coerce")
        summaries[stem] = out.set_index(config.ID_COLS)

    boy, girl = summaries["tanner_boy"], summaries["tanner_girl"]
    combined = boy.combine_first(girl)
    return combined.reset_index()


def missing_fraction(series):
    return series.isna().mean()


def describe_column(col):
    """Human-readable description for one embedder-input/covariate column,
    for interpretability only. Expands `config.RAW_COLUMN_DESCRIPTIONS`
    (sourced from each phenotype file's own JSON sidecar) into the derived
    one-hot and `_was_missing` column names actually present in the final
    matrix. Returns None for anything not covered (should not happen for a
    column this pipeline itself produced -- surfaced as a warning in main())."""
    if col.endswith("_was_missing"):
        base = col[: -len("_was_missing")]
        if base in config.CATEGORICAL_COLUMNS:
            base_desc = config.CATEGORICAL_COLUMN_DESCRIPTIONS.get(base, base)
            return f"Missingness indicator: 1 if '{base}' ({base_desc}) was missing in participants.tsv and imputed with the mode; 0 otherwise."
        base_desc = config.RAW_COLUMN_DESCRIPTIONS.get(base)
        if base_desc is None:
            return None
        return f"Missingness indicator: 1 if this session's '{base}' value was missing and imputed; 0 otherwise. `{base}`: {base_desc}"

    for cat in config.CATEGORICAL_COLUMNS:
        prefix = f"{cat}_"
        if col.startswith(prefix):
            level = col[len(prefix):]
            levels = config.CATEGORICAL_LEVEL_DESCRIPTIONS.get(cat, {})
            if level in levels:
                base_desc = config.CATEGORICAL_COLUMN_DESCRIPTIONS.get(cat, cat)
                return f"One-hot indicator: {base_desc} = '{level}' ({levels[level]})"

    return config.RAW_COLUMN_DESCRIPTIONS.get(col)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(config.PHENOTYPE_FEATURES_TSV))
    parser.add_argument("--manifest-out", default=str(config.PHENOTYPE_FEATURE_MANIFEST))
    args = parser.parse_args()

    result = load_session_spine()
    result = result.merge(load_participants(), on="participant_id", how="left")

    for stem, spec in config.NON_EF_COGNITION_TASKS.items():
        result = result.merge(load_gated_task(stem, spec), on=config.ID_COLS, how="left")

    for stem, columns in config.SELF_REPORT_SCALES.items():
        result = result.merge(load_self_report(stem, columns), on=config.ID_COLS, how="left")

    result = result.merge(load_tanner(), on=config.ID_COLS, how="left")

    dx_columns = [c for c in result.columns if c.startswith("dx_")]
    numeric_candidate_columns = (
        ["age"]
        + [col for spec in config.NON_EF_COGNITION_TASKS.values() for col in spec["columns"]]
        + [col for cols in config.SELF_REPORT_SCALES.values() for col in cols]
        + ["tanner_mean_stage"]  # tanner_complete: p=0.138 vs ef_composite, cut
        + dx_columns
    )
    categorical_candidate_columns = [c for c in config.CATEGORICAL_COLUMNS if c in result.columns]

    # Drop columns exceeding the missingness threshold, before imputing.
    dropped_columns = {}
    for col in numeric_candidate_columns + categorical_candidate_columns:
        frac = missing_fraction(result[col])
        if frac > config.MISSING_DROP_THRESHOLD:
            dropped_columns[col] = round(float(frac), 4)
    numeric_candidate_columns = [c for c in numeric_candidate_columns if c not in dropped_columns]
    categorical_candidate_columns = [c for c in categorical_candidate_columns if c not in dropped_columns]
    result = result.drop(columns=list(dropped_columns))

    imputation_values = {}
    embedder_input_columns = []

    # Categorical: impute raw string column with mode, flag, THEN one-hot.
    for col in categorical_candidate_columns:
        was_missing = result[col].isna()
        if was_missing.any():
            mode_value = result[col].mode(dropna=True).iloc[0]
            result[col] = result[col].fillna(mode_value)
            imputation_values[col] = mode_value
            result[f"{col}_was_missing"] = was_missing.astype(int)
            embedder_input_columns.append(f"{col}_was_missing")
        dummies = pd.get_dummies(result[col], prefix=col, dtype=int)
        result = pd.concat([result.drop(columns=col), dummies], axis=1)
        embedder_input_columns.extend(dummies.columns.tolist())

    # Numeric: median impute + indicator. Covariates (age) get the same
    # treatment but are routed separately, not into embedder_input_columns.
    all_numeric_columns = numeric_candidate_columns + ["session_index"]
    for col in all_numeric_columns:
        was_missing = result[col].isna()
        if was_missing.any():
            median_value = result[col].median()
            result[col] = result[col].fillna(median_value)
            imputation_values[col] = round(float(median_value), 6)
            result[f"{col}_was_missing"] = was_missing.astype(int)
            if col not in config.COVARIATE_COLUMNS:
                embedder_input_columns.append(f"{col}_was_missing")
        if col not in config.COVARIATE_COLUMNS:
            embedder_input_columns.append(col)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, sep="\t", index=False)

    column_descriptions = {}
    undescribed = []
    for col in embedder_input_columns + config.COVARIATE_COLUMNS:
        desc = describe_column(col)
        if desc is None:
            undescribed.append(col)
        else:
            column_descriptions[col] = desc

    manifest = {
        "n_rows": len(result),
        "embedder_input_columns": embedder_input_columns,
        "covariate_columns": config.COVARIATE_COLUMNS,
        "dropped_columns": dropped_columns,
        "imputation_values": imputation_values,
        "column_descriptions": column_descriptions,
    }
    manifest_path = Path(args.manifest_out)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    print(f"Wrote {len(result)} rows, {len(embedder_input_columns)} embedder-input "
          f"columns to {out_path}")
    print(f"Dropped columns (>{config.MISSING_DROP_THRESHOLD:.0%} missing): "
          f"{list(dropped_columns.keys()) or 'none'}")
    print(f"Manifest written to {manifest_path}")
    if undescribed:
        print(f"WARNING: no column_descriptions entry for: {undescribed} "
              "-- add to config.RAW_COLUMN_DESCRIPTIONS")

    # Spot-check: one-hot groups should sum to 1 per row.
    for col in categorical_candidate_columns:
        one_hot_cols = [c for c in result.columns if c.startswith(f"{col}_") and c != f"{col}_was_missing"]
        row_sums = result[one_hot_cols].sum(axis=1)
        assert (row_sums == 1).all(), f"one-hot group {col} does not sum to 1 for all rows"
    print("One-hot group spot-check passed (each group sums to 1 per row).")


if __name__ == "__main__":
    main()
