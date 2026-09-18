#!/usr/bin/env python3
"""Worker script for Nvidia FLARE federated analysis with mixed-effects models.

Connects to the hub and performs distributed training on local data.
Implements mixed-effects model with:
- Random effects at worker level (local)
- Fixed effects at hub level (global, frozen during local training)

Usage:
    python worker.py --hub-address localhost:8080 --worker-id worker_01 --data-dir ./data/center01
"""

import argparse
import json
import socket
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


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


class Worker:
    """Worker client for federated learning with mixed-effects models."""
    
    def __init__(self, hub_address, worker_id, data_dir, n_fixed_features=10, 
                 n_random_features=5, n_local_epochs=10, log_interval=5, n_rounds=3):
        self.hub_address = hub_address
        self.worker_id = worker_id
        self.data_dir = Path(data_dir)
        self.n_fixed_features = n_fixed_features
        self.n_random_features = n_random_features
        self.n_local_epochs = n_local_epochs
        self.log_interval = log_interval
        self.n_rounds = n_rounds
        
        # Local model
        self.model = MixedEffectsModel(n_fixed_features, n_random_features)
        
        # Training metrics
        self.training_metrics = []
        
    def run(self):
        """Run the worker training loop."""
        print(f"[{self.worker_id}] Starting worker...")
        
        # Load local data
        x_fixed, x_random, y_true = self.load_data()
        print(f"[{self.worker_id}] Loaded data: {len(y_true)} samples")
        
        # Connect to hub
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(self.hub_address)
        print(f"[{self.worker_id}] Connected to hub at {self.hub_address}")
        
        # Send worker ID
        client.send(self.worker_id.encode())
        
        # Receive global model from hub
        model_data = client.recv(4096)
        self.load_model_from_hub(model_data)
        print(f"[{self.worker_id}] Received global model from hub")
        
        # Run federated training rounds
        for round_num in range(self.n_rounds):
            print(f"\n[{self.worker_id}] Round {round_num + 1}")
            
            # Local training
            metrics = self.local_train(x_fixed, x_random, y_true)
            self.training_metrics.append(metrics)
            
            # Send update to hub
            self.send_update(client, metrics)
            
            try:
                # Receive updated model from hub
                model_data = client.recv(4096)
                if model_data:
                    self.load_model_from_hub(model_data)
            except (ConnectionResetError, ConnectionAbortedError):
                print(f"[{self.worker_id}] Hub closed connection")
                break
            
        # Save result report
        self.save_report()
        
        # Close connection
        try:
            client.close()
        except:
            pass
        print(f"[{self.worker_id}] Training complete")
        
    def load_data(self):
        """Load local data (random data for testing)."""
        # Generate random data for testing
        n_samples = 100
        
        x_fixed = torch.randn(n_samples, self.n_fixed_features)
        x_random = torch.randn(n_samples, self.n_random_features)
        y_true = torch.randn(n_samples)  # Random target
        
        return x_fixed, x_random, y_true
        
    def load_model_from_hub(self, model_data):
        """Load model parameters from hub."""
        params = json.loads(model_data.decode())
        
        # Update model weights
        self.model.fixed_weights.weight.data = torch.tensor(params['fixed_weights'])
        self.model.random_weights.weight.data = torch.tensor(params['random_weights'])
        
    def local_train(self, x_fixed, x_random, y_true):
        """Perform local training on worker data."""
        # Freeze fixed effects (only train random effects)
        for param in self.model.fixed_weights.parameters():
            param.requires_grad = False
            
        # Enable gradients for random effects
        for param in self.model.random_weights.parameters():
            param.requires_grad = True
            
        # Optimizer for random effects only
        optimizer = torch.optim.Adam(self.model.random_weights.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        
        # Training loop
        for epoch in range(self.n_local_epochs):
            self.model.train()
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = self.model(x_fixed, x_random)
            loss = loss_fn(y_pred, y_true)
            
            # Backward pass (only updates random effects)
            loss.backward()
            optimizer.step()
            
            # Log metrics
            if (epoch + 1) % self.log_interval == 0:
                print(f"[{self.worker_id}] Epoch {epoch + 1}/{self.n_local_epochs}: loss={loss.item():.4f}")
                
        # Return final metrics
        return {
            'loss': loss.item(),
            'random_weights': self.model.random_weights.weight.data.clone(),
            'n_samples': len(y_true)
        }
        
    def send_update(self, client, metrics):
        """Send model update to hub."""
        update = {
            'random_weights': metrics['random_weights'].tolist(),
            'metrics': {'loss': metrics['loss']},
            'worker_id': self.worker_id
        }
        
        client.send(json.dumps(update).encode())
        print(f"[{self.worker_id}] Sent update to hub")
        
    def save_report(self):
        """Save training report to file."""
        # Convert tensors to lists for JSON serialization
        serializable_metrics = []
        for metric in self.training_metrics:
            serializable_metric = {
                'loss': metric['loss'],
                'random_weights': metric['random_weights'].tolist() if hasattr(metric['random_weights'], 'tolist') else metric['random_weights'],
                'n_samples': metric['n_samples']
            }
            serializable_metrics.append(serializable_metric)
        
        report = {
            'worker_id': self.worker_id,
            'n_local_epochs': self.n_local_epochs,
            'training_metrics': serializable_metrics,
            'final_model_state': {
                'fixed_weights': self.model.fixed_weights.weight.data.tolist(),
                'random_weights': self.model.random_weights.weight.data.tolist()
            }
        }
        
        report_path = Path(f"{self.worker_id}_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"[{self.worker_id}] Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Nvidia FLARE Worker for mixed-effects models")
    parser.add_argument("--hub-address", required=True, help="Hub address (host:port)")
    parser.add_argument("--worker-id", required=True, help="Unique worker identifier")
    parser.add_argument("--data-dir", required=True, help="Path to local data directory")
    parser.add_argument("--n-local-epochs", type=int, default=10, help="Local training epochs")
    parser.add_argument("--n-rounds", type=int, default=3, help="Number of federated rounds")
    parser.add_argument("--log-interval", type=int, default=5, help="Metrics logging interval")
    parser.add_argument("--n-fixed-features", type=int, default=10, help="Number of fixed effect features")
    parser.add_argument("--n-random-features", type=int, default=5, help="Number of random effect features")
    args = parser.parse_args()
    
    # Parse hub address
    host, port = args.hub_address.split(':')
    hub_address = (host, int(port))
    
    worker = Worker(
        hub_address=hub_address,
        worker_id=args.worker_id,
        data_dir=args.data_dir,
        n_fixed_features=args.n_fixed_features,
        n_random_features=args.n_random_features,
        n_local_epochs=args.n_local_epochs,
        log_interval=args.log_interval,
        n_rounds=args.n_rounds
    )
    
    worker.run()


if __name__ == "__main__":
    main()