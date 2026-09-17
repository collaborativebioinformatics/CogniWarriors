#!/usr/bin/env python3
"""Worker script for Nvidia FLARE federated analysis.

Connects to the center and performs distributed training on structural MRI data.
"""

import argparse
from flare import FlareWorker


def main():
    parser = argparse.ArgumentParser(description="Nvidia FLARE Worker")
    parser.add_argument("--center", required=True, help="Center address (host:port)")
    parser.add_argument("--worker-id", required=True, help="Unique worker identifier")
    parser.add_argument("--data-dir", required=True, help="Path to structural MRI data")
    args = parser.parse_args()

    worker = FlareWorker(
        center_address=args.center,
        worker_id=args.worker_id,
        data_dir=args.data_dir,
    )
    worker.run()


if __name__ == "__main__":
    main()