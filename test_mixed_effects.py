#!/usr/bin/env python3
"""Test script for mixed-effects federated learning.

Tests the hub and worker implementation with random data.
"""

import subprocess
import time
import sys
from pathlib import Path


def test_hub_worker():
    """Test hub and worker communication."""
    print("Testing mixed-effects federated learning...")
    
    # Start hub in background
    hub_cmd = [
        sys.executable, "center.py",
        "--port", "8080",
        "--n-workers", "2",
        "--n-rounds", "3",
        "--n-local-epochs", "5"
    ]
    
    hub_process = subprocess.Popen(hub_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Started hub process")
    
    # Wait for hub to start
    time.sleep(2)
    
    # Start workers
    worker_processes = []
    for i in range(2):
        worker_cmd = [
            sys.executable, "worker.py",
            "--hub-address", "localhost:8080",
            "--worker-id", f"worker_{i+1:02d}",
            "--data-dir", f"./data/center{i+1:02d}",
            "--n-local-epochs", "5",
            "--log-interval", "2"
        ]
        
        worker_process = subprocess.Popen(worker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        worker_processes.append(worker_process)
        print(f"Started worker_{i+1:02d} process")
    
    # Wait for all processes to complete
    print("\nWaiting for training to complete...")
    
    for i, process in enumerate(worker_processes):
        stdout, stderr = process.communicate()
        print(f"\nWorker {i+1} output:")
        print(stdout.decode())
        if stderr:
            print(f"Worker {i+1} errors:")
            print(stderr.decode())
    
    # Wait for hub to complete
    hub_stdout, hub_stderr = hub_process.communicate()
    print("\nHub output:")
    print(hub_stdout.decode())
    if hub_stderr:
        print("Hub errors:")
        print(hub_stderr.decode())
    
    # Check if result files were created
    result_files = [
        "hub_model.pt",
        "hub_metrics.json",
        "worker_01_report.json",
        "worker_02_report.json"
    ]
    
    print("\nChecking result files:")
    for file_name in result_files:
        if Path(file_name).exists():
            print(f"✓ {file_name} created")
        else:
            print(f"✗ {file_name} not found")
    
    print("\nTest completed!")


if __name__ == "__main__":
    test_hub_worker()