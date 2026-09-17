#!/usr/bin/env python3
"""Center script for Nvidia FLARE federated analysis.

Distributes workers and coordinates the federated learning process.
"""

import argparse
import time
from flare import FlareApp


def main():
    parser = argparse.ArgumentParser(description="Nvidia FLARE Center")
    parser.add_argument("--config", required=True, help="Path to FLARE config file")
    parser.add_argument("--port", type=int, default=8080, help="Center port")
    args = parser.parse_args()

    app = FlareApp(config_path=args.config, center_port=args.port)
    app.start()


if __name__ == "__main__":
    main()