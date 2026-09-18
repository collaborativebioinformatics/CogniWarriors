#!/usr/bin/env python3
"""Test federated learning with mixed-effects models.

Runs hub and workers in separate threads for testing.
"""

import threading
import time
import json
import socket
from pathlib import Path

import torch
import torch.nn as nn


class MixedEffectsModel(nn.Module):
    def __init__(self, n_fixed_features, n_random_features):
        super().__init__()
        self.fixed_weights = nn.Linear(n_fixed_features, 1, bias=False)
        self.random_weights = nn.Linear(n_random_features, 1, bias=False)
        
    def forward(self, x_fixed, x_random):
        fixed_output = self.fixed_weights(x_fixed)
        random_output = self.random_weights(x_random)
        return (fixed_output + random_output).squeeze(-1)


def run_hub(port=8080, n_workers=2, n_rounds=3):
    """Run hub server in a thread."""
    print(f"[Hub] Starting server on port {port}")
    
    # Create model
    model = MixedEffectsModel(n_fixed_features=10, n_random_features=5)
    
    # Create socket server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('localhost', port))
    server.listen(n_workers)
    
    # Accept worker connections
    workers = []
    for i in range(n_workers):
        client, address = server.accept()
        worker_id = client.recv(1024).decode()
        workers.append((worker_id, client))
        print(f"[Hub] Worker {worker_id} connected")
        
        # Send model to worker
        model_data = json.dumps({
            'fixed_weights': model.fixed_weights.weight.data.tolist(),
            'random_weights': model.random_weights.weight.data.tolist()
        }).encode()
        client.send(model_data)
    
    # Run training rounds
    for round_num in range(n_rounds):
        print(f"\n[Hub] Round {round_num + 1}/{n_rounds}")
        
        # Collect updates
        updates = []
        for worker_id, client in workers:
            # Wait for update
            update_data = b""
            while True:
                chunk = client.recv(4096)
                if chunk:
                    update_data += chunk
                    try:
                        update = json.loads(update_data.decode())
                        break
                    except json.JSONDecodeError:
                        continue
            
            updates.append(update)
            
        # Aggregate updates (average random effects)
        random_weights_list = [torch.tensor(u['random_weights']) for u in updates]
        avg_random_weights = torch.mean(torch.stack(random_weights_list), dim=0)
        model.random_weights.weight.data = avg_random_weights
        
        print(f"[Hub] Aggregated updates from {len(workers)} workers")
        
        # Send updated model to workers
        model_data = json.dumps({
            'fixed_weights': model.fixed_weights.weight.data.tolist(),
            'random_weights': model.random_weights.weight.data.tolist()
        }).encode()
        
        for worker_id, client in workers:
            client.send(model_data)
    
    # Save model
    torch.save(model.state_dict(), "hub_model.pt")
    print("[Hub] Model saved to hub_model.pt")
    
    # Close connections
    for worker_id, client in workers:
        client.close()
    server.close()


def run_worker(worker_id, port=8080, n_epochs=5):
    """Run worker in a thread."""
    print(f"[{worker_id}] Starting worker...")
    
    # Connect to hub
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', port))
    print(f"[{worker_id}] Connected to hub")
    
    # Send worker ID
    client.send(worker_id.encode())
    
    # Receive model from hub
    model_data = client.recv(4096)
    params = json.loads(model_data.decode())
    
    # Create model
    model = MixedEffectsModel(n_fixed_features=10, n_random_features=5)
    model.fixed_weights.weight.data = torch.tensor(params['fixed_weights'])
    model.random_weights.weight.data = torch.tensor(params['random_weights'])
    
    # Generate random data
    x_fixed = torch.randn(100, 10)
    x_random = torch.randn(100, 5)
    y_true = torch.randn(100)
    
    # Local training
    optimizer = torch.optim.Adam(model.random_weights.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        y_pred = model(x_fixed, x_random)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"[{worker_id}] Epoch {epoch + 1}/{n_epochs}: loss={loss.item():.4f}")
    
    # Send update to hub
    update = {
        'random_weights': model.random_weights.weight.data.tolist(),
        'metrics': {'loss': loss.item()},
        'worker_id': worker_id
    }
    client.send(json.dumps(update).encode())
    
    # Receive updated model
    model_data = client.recv(4096)
    params = json.loads(model_data.decode())
    model.random_weights.weight.data = torch.tensor(params['random_weights'])
    
    # Save report
    report = {
        'worker_id': worker_id,
        'final_loss': loss.item(),
        'model_state': {
            'fixed_weights': model.fixed_weights.weight.data.tolist(),
            'random_weights': model.random_weights.weight.data.tolist()
        }
    }
    
    with open(f"{worker_id}_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"[{worker_id}] Report saved to {worker_id}_report.json")
    
    client.close()


def main():
    print("Testing Federated Learning with Mixed-Effects Models...")
    
    # Start hub in a thread
    hub_thread = threading.Thread(target=run_hub, args=(8080, 2, 3))
    hub_thread.start()
    
    # Wait for hub to start
    time.sleep(2)
    
    # Start workers in threads
    worker_threads = []
    for i in range(2):
        worker_thread = threading.Thread(target=run_worker, args=(f"worker_{i+1:02d}", 8080, 5))
        worker_threads.append(worker_thread)
        worker_thread.start()
        time.sleep(0.5)  # Stagger worker starts
    
    # Wait for all threads to complete
    hub_thread.join()
    for thread in worker_threads:
        thread.join()
    
    print("\n✓ Test completed!")
    
    # Check result files
    result_files = ["hub_model.pt", "worker_01_report.json", "worker_02_report.json"]
    for file_name in result_files:
        if Path(file_name).exists():
            print(f"✓ {file_name} created")
        else:
            print(f"✗ {file_name} not found")


if __name__ == "__main__":
    main()