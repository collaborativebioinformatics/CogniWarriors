#!/usr/bin/env python3
"""Create two center-local data roots for FLARE site simulations."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd


CENTER_IDS = ("center1", "center2")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data"), help="Source full data root.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "centers",
        help="Output directory that will contain center1/ and center2/.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic participant split seed.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def load_trainable_subjects(source: Path) -> List[str]:
    phenotype = pd.read_csv(source / "processed" / "phenotype_features.tsv", sep="\t")
    composite = pd.read_csv(source / "processed" / "ef_composite.tsv", sep="\t")
    embeddings = pd.concat(
        [
            pd.read_csv(source / "images" / "embeddings" / "train_embedding.csv"),
            pd.read_csv(source / "images" / "embeddings" / "test_embedding.csv"),
        ],
        ignore_index=True,
    )

    subjects = (
        set(phenotype["participant_id"].dropna())
        & set(composite["participant_id"].dropna())
        & set(embeddings["subject"].dropna())
    )
    if len(subjects) < 2:
        raise RuntimeError("need at least two trainable participants to create two centers")
    return sorted(subjects)


def split_subjects(subjects: List[str], seed: int) -> Dict[str, Set[str]]:
    shuffled = list(subjects)
    random.Random(seed).shuffle(shuffled)
    midpoint = len(shuffled) // 2
    return {
        "center1": set(sorted(shuffled[:midpoint])),
        "center2": set(sorted(shuffled[midpoint:])),
    }


def session_file_subject(path: Path) -> str | None:
    name = path.name
    if name.startswith("sub-") and "_sessions." in name:
        return name.split("_sessions.", 1)[0]
    return None


def table_id_column(columns: Iterable[str]) -> str | None:
    columns = set(columns)
    if "participant_id" in columns:
        return "participant_id"
    if "subject" in columns:
        return "subject"
    return None


def copy_or_subset_table(source_file: Path, destination_file: Path, subjects: Set[str]):
    sep = "\t" if source_file.suffix == ".tsv" else ","
    frame = pd.read_csv(source_file, sep=sep)
    id_column = table_id_column(frame.columns)
    if id_column is not None:
        frame = frame[frame[id_column].isin(subjects)]
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination_file, sep=sep, index=False)


def collect_source_files(source: Path, output: Path) -> List[Path]:
    files = []
    for source_file in source.rglob("*"):
        if source_file.is_dir():
            continue
        try:
            source_file.relative_to(output)
            continue
        except ValueError:
            files.append(source_file)
    return files


def copy_center(source: Path, destination: Path, subjects: Set[str], source_files: List[Path]):
    for source_file in source_files:

        relative = source_file.relative_to(source)
        session_subject = session_file_subject(source_file)
        if session_subject is not None and session_subject not in subjects:
            continue

        destination_file = destination / relative
        if source_file.suffix in {".tsv", ".csv"}:
            copy_or_subset_table(source_file, destination_file, subjects)
        else:
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination_file)


def write_manifest(output: Path, assignments: Dict[str, Set[str]], source: Path, seed: int):
    manifest = {
        "source": str(source.resolve()),
        "seed": seed,
        "centers": {
            center_id: {
                "num_participants": len(subjects),
                "participants": sorted(subjects),
            }
            for center_id, subjects in assignments.items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "data_split_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    for center_id, subjects in assignments.items():
        with open(output / center_id / "center_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "center_id": center_id,
                    "num_participants": len(subjects),
                    "participants": sorted(subjects),
                },
                handle,
                indent=2,
            )


def main():
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(f"source data root does not exist: {source}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
        shutil.rmtree(output)

    subjects = load_trainable_subjects(source)
    assignments = split_subjects(subjects, args.seed)
    source_files = collect_source_files(source, output)
    for center_id in CENTER_IDS:
        copy_center(source, output / center_id, assignments[center_id], source_files)
    write_manifest(output, assignments, source, args.seed)

    for center_id in CENTER_IDS:
        print(f"{center_id}: {len(assignments[center_id])} participants -> {output / center_id}")


if __name__ == "__main__":
    main()
