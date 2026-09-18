# Mixed-Effects Federated Learning Test Instructions

This guide explains how to run the mixed-effects federated learning system on a single machine for testing.

## Prerequisites

Make sure you have the required packages installed:
```bash
pip install torch numpy
```

## Quick Start

### Option 1: Run the test script (recommended)
```bash
./run_test.sh
```

### Option 2: Run manually

#### Step 1: Start the Hub Server
Open a terminal and run:
```bash
python3 center.py --port 8080 --n-workers 2 --n-rounds 3 --n-local-epochs 5
```

#### Step 2: Start Workers (in separate terminals)
Open two new terminals and run:

**Terminal 1:**
```bash
python3 worker.py --hub-address localhost:8080 --worker-id worker_01 --data-dir ./data/center01 --n-local-epochs 5 --log-interval 2
```

**Terminal 2:**
```bash
python3 worker.py --hub-address localhost:8080 --worker-id worker_02 --data-dir ./data/center02 --n-local-epochs 5 --log-interval 2
```

## Command Line Arguments

### Center/Hub Script (`center.py`)
- `--port`: Port number for the hub server (default: 8080)
- `--n-workers`: Number of workers to wait for (default: 2)
- `--n-rounds`: Number of federated training rounds (default: 5)
- `--n-local-epochs`: Number of local epochs per worker (default: 10)
- `--n-fixed-features`: Number of fixed effect features (default: 10)
- `--n-random-features`: Number of random effect features (default: 5)

### Worker Script (`worker.py`)
- `--hub-address`: Hub server address in format host:port (required)
- `--worker-id`: Unique identifier for this worker (required)
- `--data-dir`: Path to local data directory (required)
- `--n-local-epochs`: Number of local training epochs (default: 10)
- `--log-interval`: Metrics logging interval (default: 5)
- `--n-fixed-features`: Number of fixed effect features (default: 10)
- `--n-random-features`: Number of random effect features (default: 5)

## Expected Output

### Hub Output
```
[Hub] Starting server on port 8080
[Hub] Waiting for 2 workers...
[Hub] Worker worker_01 connected from ('127.0.0.1', 12345)
[Hub] Worker worker_02 connected from ('127.0.0.1', 12346)
[Hub] All 2 workers connected

[Hub] Round 1/3
[Hub] Collected updates from 2 workers
[Hub] Aggregated model updates (FedAvg)
[Hub] Broadcasted updated model to all workers
[Hub] Round 1 metrics: avg_loss=1.2345
...
[Hub] Model saved to hub_model.pt
[Hub] Metrics saved to hub_metrics.json
[Hub] Training complete
```

### Worker Output
```
[worker_01] Starting worker...
[worker_01] Loaded data: 100 samples
[worker_01] Connected to hub at ('localhost', 8080)
[worker_01] Received global model from hub

[worker_01] Round 1
[worker_01] Epoch 5/10: loss=1.2345
[worker_01] Epoch 10/10: loss=0.9876
[worker_01] Sent update to hub
...
[worker_01] Report saved to worker_01_report.json
[worker_01] Training complete
```

## Result Files

After training completes, you'll find these files:

1. **`hub_model.pt`**: Final global model with:
   - Fixed effects (shared across all workers)
   - Aggregated random effects (averaged from all workers)

2. **`hub_metrics.json`**: Training metrics including:
   - Round-by-round average loss
   - Number of workers per round

3. **`worker_01_report.json`** and **`worker_02_report.json`**: Worker-specific reports with:
   - Final loss values
   - Model state at end of training

## Architecture Overview

The system implements a mixed-effects model for federated learning:

- **Fixed Effects**: Global parameters learned at the hub level (shared across all workers)
- **Random Effects**: Local parameters learned at the worker level (specific to each site)

This architecture allows:
1. Privacy-preserving learning (raw data stays local)
2. Site-specific adaptation (random effects capture local patterns)
3. Global knowledge sharing (fixed effects capture common patterns)

## Troubleshooting

### "Connection refused" error
- Make sure the hub is running before starting workers
- Check that the port number matches between hub and workers

### "Address already in use" error
- Kill any existing processes using the port: `pkill -f "center.py"`
- Wait a few seconds before restarting

### Workers hang waiting for hub
- Ensure the hub is started with the correct number of workers (`--n-workers`)
- Check that all workers are connected before training begins

### Training takes too long
- Reduce `--n-rounds` or `--n-local-epochs` for faster testing
- Use smaller feature dimensions (`--n-fixed-features`, `--n-random-features`)