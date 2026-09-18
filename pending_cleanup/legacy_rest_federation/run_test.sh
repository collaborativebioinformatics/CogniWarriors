#!/bin/bash
# Run script for testing mixed-effects federated learning
# This script runs the hub and workers on the same machine for testing

echo "=== Testing Mixed-Effects Federated Learning ==="
echo ""

# Clean up any existing processes
echo "Cleaning up existing processes..."
pkill -f "center.py" 2>/dev/null || true
pkill -f "worker.py" 2>/dev/null || true
sleep 1

# Remove old result files
echo "Removing old result files..."
rm -f hub_model.pt hub_metrics.json worker_*_report.json 2>/dev/null || true

echo ""
echo "=== Starting Hub Server ==="
echo "Starting hub on port 8080..."
python3 center.py --port 8080 --n-workers 2 --n-rounds 3 --n-local-epochs 5 &
HUB_PID=$!

# Wait for hub to start
sleep 2

echo ""
echo "=== Starting Workers ==="
echo "Starting worker_01..."
python3 worker.py --hub-address localhost:8080 --worker-id worker_01 --data-dir ./data/center01 --n-local-epochs 5 --log-interval 2 &
WORKER1_PID=$!

echo "Starting worker_02..."
python3 worker.py --hub-address localhost:8080 --worker-id worker_02 --data-dir ./data/center02 --n-local-epochs 5 --log-interval 2 &
WORKER2_PID=$!

echo ""
echo "=== Waiting for Training to Complete ==="
echo "Hub PID: $HUB_PID"
echo "Worker 1 PID: $WORKER1_PID"
echo "Worker 2 PID: $WORKER2_PID"

# Wait for all processes to complete
wait $WORKER1_PID 2>/dev/null
wait $WORKER2_PID 2>/dev/null
wait $HUB_PID 2>/dev/null

echo ""
echo "=== Training Complete ==="
echo ""
echo "Checking result files..."
if [ -f "hub_model.pt" ]; then
    echo "✓ hub_model.pt created"
else
    echo "✗ hub_model.pt not found"
fi

if [ -f "hub_metrics.json" ]; then
    echo "✓ hub_metrics.json created"
else
    echo "✗ hub_metrics.json not found"
fi

if [ -f "worker_01_report.json" ]; then
    echo "✓ worker_01_report.json created"
else
    echo "✗ worker_01_report.json not found"
fi

if [ -f "worker_02_report.json" ]; then
    echo "✓ worker_02_report.json created"
else
    echo "✗ worker_02_report.json not found"
fi

echo ""
echo "=== Test Complete ==="