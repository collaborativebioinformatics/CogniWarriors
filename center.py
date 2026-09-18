#!/usr/bin/env python3
"""Center/Hub script for Nvidia FLARE federated analysis with mixed-effects models.

Distributes workers and coordinates the federated learning process.
Implements FedAvg aggregation for mixed-effects models with:
- Random effects at worker level (local)
- Fixed effects at hub level (global)

Usage:
    python center.py --port 8080 --n-workers 2 --n-rounds 5
"""

import argparse
import json
import socket
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn


class MixedEffectsModel(nn.Module):
    """Mixed-effects model with fixed effects (global) and random effects (local)."""
    
    def __init__(self, n_fixed_features, n_random_features):
        super().__init__()
        self.fixed_weights = nn.Linear(n_fixed_features, 1, bias=False)
        self.random_weights = nn.Linear(n_random_features, 1, bias=False)
        
    def forward(self, x_fixed, x_random):
        fixed_output = self.fixed_weights(x_fixed)
        random_output = self.random_weights(x_random)
        return (fixed_output + random_output).squeeze(-1)


class Hub:
    """Hub server for federated learning with mixed-effects models."""
    
    def __init__(self, port=8080, n_workers=2, n_fixed_features=10, 
                 n_random_features=5, n_rounds=5, n_local_epochs=10):
        self.port = port
        self.n_workers = n_workers
        self.n_fixed_features = n_fixed_features
        self.n_random_features = n_random_features
        self.n_rounds = n_rounds
        self.n_local_epochs = n_local_epochs
        
        # Global model with fixed effects
        self.global_model = MixedEffectsModel(n_fixed_features, n_random_features)
        
        # Store worker connections
        self.workers = {}
        self.worker_updates = {}
        self.round_metrics = []
        
    def start(self):
        """Start the hub server."""
        print(f"[Hub] Starting server on port {self.port}")
        print(f"[Hub] Waiting for {self.n_workers} workers...")
        
        # Create socket server
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('localhost', self.port))
        server.listen(self.n_workers)
        
        # Accept worker connections
        while len(self.workers) < self.n_workers:
            client, address = server.accept()
            worker_id = client.recv(1024).decode()
            self.workers[worker_id] = client
            print(f"[Hub] Worker {worker_id} connected from {address}")
            
            # Send global model to worker
            model_data = self.serialize_model()
            client.send(model_data)
        
        print(f"[Hub] All {self.n_workers} workers connected")
        
        # Run federated training
        for round_num in range(self.n_rounds):
            print(f"\n[Hub] Round {round_num + 1}/{self.n_rounds}")
            
            # Collect updates from all workers
            self.collect_updates()
            
            # Aggregate updates (FedAvg)
            self.aggregate_updates()
            
            # Send updated model to all workers
            self.broadcast_model()
            
            # Log round metrics
            self.log_round_metrics(round_num)
        
        # Save final model and metrics
        self.save_results()
        
        # Wait a moment for workers to receive final update
        time.sleep(1)
        
        # Close connections
        for worker_id, client in self.workers.items():
            try:
                client.close()
            except:
                pass
        server.close()
        
        print("[Hub] Training complete")
        
    def collect_updates(self):
        """Collect model updates from all workers."""
        self.worker_updates = {}
        
        for worker_id, client in self.workers.items():
            # Receive update from worker
            update_data = client.recv(4096)
            update = self.deserialize_update(update_data)
            self.worker_updates[worker_id] = update
            
        print(f"[Hub] Collected updates from {len(self.worker_updates)} workers")
        
    def aggregate_updates(self):
        """Aggregate worker updates using FedAvg."""
        # Average random effects from all workers
        random_weights_list = []
        for worker_id, update in self.worker_updates.items():
            random_weights_list.append(update['random_weights'])
        
        # Stack and average
        avg_random_weights = torch.mean(torch.stack(random_weights_list), dim=0)
        
        # Update global model
        self.global_model.random_weights.weight.data = avg_random_weights
        
        print("[Hub] Aggregated model updates (FedAvg)")
        
    def broadcast_model(self):
        """Send updated global model to all workers."""
        model_data = self.serialize_model()
        
        for worker_id, client in self.workers.items():
            client.send(model_data)
            
        print("[Hub] Broadcasted updated model to all workers")
        
    def serialize_model(self):
        """Serialize model to bytes."""
        return json.dumps({
            'fixed_weights': self.global_model.fixed_weights.weight.data.tolist(),
            'random_weights': self.global_model.random_weights.weight.data.tolist()
        }).encode()
        
    def deserialize_update(self, data):
        """Deserialize worker update from bytes."""
        update = json.loads(data.decode())
        return {
            'random_weights': torch.tensor(update['random_weights']),
            'metrics': update.get('metrics', {})
        }
        
    def log_round_metrics(self, round_num):
        """Log metrics for the current round."""
        # Calculate average metrics from workers
        avg_loss = 0
        for worker_id, update in self.worker_updates.items():
            if 'metrics' in update and 'loss' in update['metrics']:
                avg_loss += update['metrics']['loss']
        
        if self.worker_updates:
            avg_loss /= len(self.worker_updates)
        
        self.round_metrics.append({
            'round': round_num + 1,
            'avg_loss': avg_loss,
            'n_workers': len(self.worker_updates)
        })
        
        print(f"[Hub] Round {round_num + 1} metrics: avg_loss={avg_loss:.4f}")
        
    def save_results(self):
        """Save final model and training metrics."""
        # Save model
        model_path = Path("hub_model.pt")
        torch.save(self.global_model.state_dict(), model_path)
        print(f"[Hub] Model saved to {model_path}")
        
        # Save metrics
        metrics_path = Path("hub_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.round_metrics, f, indent=2)
        print(f"[Hub] Metrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Nvidia FLARE Hub for mixed-effects models")
    parser.add_argument("--port", type=int, default=8080, help="Hub port")
    parser.add_argument("--n-workers", type=int, default=2, help="Number of workers to wait for")
    parser.add_argument("--n-rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--n-local-epochs", type=int, default=10, help="Local epochs per worker")
    parser.add_argument("--n-fixed-features", type=int, default=10, help="Number of fixed effect features")
    parser.add_argument("--n-random-features", type=int, default=5, help="Number of random effect features")
    args = parser.parse_args()
    
    hub = Hub(
        port=args.port,
        n_workers=args.n_workers,
        n_fixed_features=args.n_fixed_features,
        n_random_features=args.n_random_features,
        n_rounds=args.n_rounds,
        n_local_epochs=args.n_local_epochs
    )
    
    hub.start()


if __name__ == "__main__":
    main()