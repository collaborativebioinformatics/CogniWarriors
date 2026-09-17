#!/usr/bin/env python3
"""Compute a per-session executive-function (EF) composite score.

Row spine: every (participant_id, session_id) found in the downloaded
sessions.tsv files (the BIDS ground truth for "a session exists") -- not the
CNB task files themselves, since ~6 of the 225 sessions have no CNB data at
all and should show up as explicit NaN rather than silently disappearing.

For each of the 5 EF tasks in config.EF_TASKS: keep a session's value only
if that task's own valid_code == 'V' (strict), z-score the resulting column
pooled across all sessions (not per age-band -- N~220 over an 8-16y range
is too small to estimate per-band mean/SD reliably), sign-flip if lower is
better (Trail Making B RT), then average whichever z-scores are non-null
for that session. Composite is NaN if fewer than MIN_TASKS_REQUIRED tasks
are available.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def load_session_spine():
    rows = []
    for path in sorted(config.SESSIONS_DIR.glob("sub-*_sessions.tsv")):
        participant_id = path.name.removesuffix("_sessions.tsv")
        df = pd.read_csv(path, sep="\t", dtype=str)
        for session_id in df["session_id"]:
            rows.append({"participant_id": participant_id, "session_id": session_id})
    spine = pd.DataFrame(rows)
    if spine.empty:
        raise SystemExit(f"No sessions.tsv files found under {config.SESSIONS_DIR}")
    return spine


def load_task_column(task_stem):
    spec = config.EF_TASKS[task_stem]
    path = config.PHENOTYPE_DIR / f"{task_stem}.tsv"
    df = pd.read_csv(path, sep="\t", na_values=["n/a"])
    valid = df[spec["valid_code_column"]].isin(config.VALID_CODES_OK)
    raw = pd.to_numeric(df[spec["column"]], errors="coerce")
    raw = raw.where(valid)
    out = df[config.ID_COLS].copy()
    out[f"raw_{task_stem}"] = raw
    return out


def zscore_signed(series, higher_is_better):
    z = (series - series.mean()) / series.std(ddof=0)
    return z if higher_is_better else -z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(config.EF_COMPOSITE_TSV))
    parser.add_argument("--min-tasks", type=int, default=config.MIN_TASKS_REQUIRED)
    args = parser.parse_args()

    result = load_session_spine()
    z_cols = []
    for task_stem, spec in config.EF_TASKS.items():
        task_df = load_task_column(task_stem)
        result = result.merge(task_df, on=config.ID_COLS, how="left")
        raw_col = f"raw_{task_stem}"
        z_col = f"z_{task_stem}"
        result[z_col] = zscore_signed(result[raw_col], spec["higher_is_better"])
        z_cols.append(z_col)
        n_valid = result[raw_col].notna().sum()
        print(f"{task_stem}: {n_valid}/{len(result)} sessions with valid data "
              f"(z mean={result[z_col].mean():.3f}, std={result[z_col].std():.3f})")

    result["n_tasks_valid"] = result[z_cols].notna().sum(axis=1)
    result["ef_composite"] = result[z_cols].mean(axis=1, skipna=True)
    result.loc[result["n_tasks_valid"] < args.min_tasks, "ef_composite"] = np.nan

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, sep="\t", index=False)

    print(f"\nWrote {len(result)} rows to {out_path}")
    print("n_tasks_valid distribution:")
    print(result["n_tasks_valid"].value_counts().sort_index().to_string())
    n_composite = result["ef_composite"].notna().sum()
    print(f"ef_composite non-null: {n_composite}/{len(result)}")


if __name__ == "__main__":
    main()
