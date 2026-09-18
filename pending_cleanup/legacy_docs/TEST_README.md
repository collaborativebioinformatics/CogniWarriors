# Mixed-Effects Federated Learning Test

## Quick Start

Run the test with:
```bash
./run_test.sh
```

## Manual Testing

### Step 1: Start Hub
```bash
python3 center.py --port 8080 --n-workers 2 --n-rounds 3 --n-local-epochs 5
```

### Step 2: Start Workers (in separate terminals)

**Terminal 1:**
```bash
python3 worker.py --hub-address localhost:8080 --worker-id worker_01 --data-dir ./data/center01 --n-local-epochs 5 --n-rounds 3
```

**Terminal 2:**
```bash
python3 worker.py --hub-address localhost:8080 --worker-id worker_02 --data-dir ./data/center02 --n-local-epochs 5 --n-rounds 3
```

## Expected Output

Training will complete in ~30 seconds with output showing:
- Hub starting and waiting for workers
- Workers connecting and receiving model
- Local training with decreasing loss
- Model updates being sent to hub
- Final model and metrics saved

## Result Files

After training:
- `hub_model.pt` - Final global model
- `hub_metrics.json` - Training metrics
- `worker_01_report.json` - Worker 01 report
- `worker_02_report.json` - Worker 02 report

## Architecture

- **Hub**: Coordinates training, aggregates random effects
- **Workers**: Train locally on random data, send updates to hub
- **Model**: Mixed-effects with fixed (global) and random (local) effects

## Troubleshooting

If training hangs:
1. Kill existing processes: `pkill -f "center.py"`
2. Wait 2 seconds
3. Restart with `./run_test.sh`