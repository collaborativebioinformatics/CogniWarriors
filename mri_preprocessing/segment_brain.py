#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


BUNDLE_NAME = "wholeBrainSeg_Large_UNEST_segmentation"
BUNDLE_VERSION = "0.2.2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MONAI Whole Brain Segmentation on T1w MRI files."
    )

    parser.add_argument(
        "--root_dir",
        type=Path,
        required=True,
        help="Root directory containing the PENN-LEAD dataset.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where segmentation outputs will be written.",
    )

    return parser.parse_args()


def find_t1_files(root_dir):
    """
    Find all T1w NIfTI files recursively.
    """

    files = sorted(root_dir.rglob("*_T1w.nii.gz"))

    # Ignore anything inside the output directory if it happens
    # to be located under root_dir.
    return files


def download_bundle(bundle_dir):
    """
    Download the MONAI Model Zoo bundle if it is not already present.
    """

    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Check whether a bundle already exists.
    existing = list(bundle_dir.glob(f"{BUNDLE_NAME}*"))

    if existing:
        print(f"[INFO] Bundle already present:")
        for path in existing:
            print(f"       {path}")
        return

    print()
    print("=" * 70)
    print("Downloading MONAI Whole Brain Segmentation bundle")
    print("=" * 70)

    cmd = [
        sys.executable,
        "-m",
        "monai.bundle",
        "download",
        BUNDLE_NAME,
        "--version",
        BUNDLE_VERSION,
        "--bundle_dir",
        str(bundle_dir),
    ]

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():

    args = parse_args()

    root_dir = args.root_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not root_dir.exists():
        raise FileNotFoundError(
            f"Root directory does not exist: {root_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep downloaded MONAI bundle outside the dataset.
    bundle_dir = output_dir / "_monai_bundle"

    print()
    print("=" * 70)
    print("MONAI Whole Brain Segmentation")
    print("=" * 70)
    print(f"Root directory : {root_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # ------------------------------------------------------------
    # Find T1 images
    # ------------------------------------------------------------

    t1_files = find_t1_files(root_dir)

    if not t1_files:
        raise RuntimeError(
            f"No '*_T1w.nii.gz' files found under {root_dir}"
        )

    print(f"[INFO] Found {len(t1_files)} T1w scans")

    for i, path in enumerate(t1_files, 1):
        print(f"  [{i:03d}] {path}")

    # ------------------------------------------------------------
    # Download MONAI bundle
    # ------------------------------------------------------------

    download_bundle(bundle_dir)

    # Locate extracted bundle.
    bundle_candidates = [
        p for p in bundle_dir.iterdir()
        if p.is_dir() and BUNDLE_NAME in p.name
    ]

    if not bundle_candidates:
        raise RuntimeError(
            f"Could not find extracted bundle in {bundle_dir}"
        )

    bundle_path = bundle_candidates[0]

    print()
    print(f"[INFO] Using bundle:")
    print(f"       {bundle_path}")

    # ------------------------------------------------------------
    # Prepare temporary inference directory
    # ------------------------------------------------------------

    inference_dir = output_dir / "_inference_input"
    inference_dir.mkdir(parents=True, exist_ok=True)

    # MONAI bundle inference is generally configured around a
    # dataset directory. We create a temporary flat input folder.
    #
    # Use symlinks so that we do NOT duplicate the MRI data.

    for t1 in t1_files:

        link = inference_dir / t1.name

        if link.exists() or link.is_symlink():
            continue

        link.symlink_to(t1)

    # ------------------------------------------------------------
    # Locate inference config
    # ------------------------------------------------------------

    configs = bundle_path / "configs"

    if not configs.exists():
        raise RuntimeError(
            f"Could not find configs directory: {configs}"
        )

    inference_configs = list(configs.glob("*inference*.json"))

    if not inference_configs:
        raise RuntimeError(
            f"No inference config found in {configs}"
        )

    inference_config = inference_configs[0]

    print()
    print(f"[INFO] Inference config:")
    print(f"       {inference_config}")

    # ------------------------------------------------------------
    # Run MONAI Bundle
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("Starting segmentation")
    print("=" * 70)

    cmd = [
        sys.executable,
        "-m",
        "monai.bundle",
        "run",
        "--config_file",
        str(inference_config),
        "--input_dir",
        str(inference_dir),
        "--output_dir",
        str(output_dir),
    ]

    print()
    print("Command:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)

    print()
    print("=" * 70)
    print("Segmentation completed")
    print("=" * 70)
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()