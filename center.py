#!/usr/bin/env python3
"""Center/Hub script for Nvidia FLARE federated analysis with mixed-effects models.

Distributes workers and coordinates the federated learning process.
Implements FedAvg aggregation for mixed-effects models with:
- Random effects at worker level (local)
- Fixed effects at hub level (global)

Features:
- Global Model: Maintains centralized model with fixed effects
- FedAvg Aggregation: Averages random effects from all workers
- Round Management: Coordinates multiple training rounds
- Weight Distribution: Broadcasts updated model to all workers

Docker-ready: Binds to 0.0.0.0 for container networking

Usage:
    python center.py --port 8080 --n-workers 2 --n-rounds 5
"""

import argparse
import json
import socket
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class MixedEffectsModel(nn.Module):
    """Mixed-effects model with fixed effects (global) and random effects (local)."""
    
    def __init__(self, n_fixed_features: int, n_random_features: int):
        super().__init__()
        self.fixed_weights = nn.Linear(n_fixed_features, 1, bias=False)
        self.random_weights = nn.Linear(n_random_features, 1, bias=False)
        
    def forward(self, x_fixed: torch.Tensor, x_random: torch.Tensor) -> torch.Tensor:
        fixed_output = self.fixed_weights(x_fixed)
        random_output = self.random_weights(x_random)
        return (fixed_output + random_output).squeeze(-1)
    
    def get_fixed_weights(self) -> torch.Tensor:
        """Get fixed effect weights."""
        return self.fixed_weights.weight.data.clone()
    
    def get_random_weights(self) -> torch.Tensor:
        """Get random effect weights."""
        return self.random_weights.weight.data.clone()
    
    def set_random_weights(self, weights: torch.Tensor):
        """Set random effect weights."""
        self.random_weights.weight.data = weights.clone()


class GlobalModel:
    """Global model manager for federated learning."""
    
    def __init__(self, n_fixed_features: int, n_random_features: int):
        self.model = MixedEffectsModel(n_fixed_features, n_random_features)
        self.n_fixed_features = n_fixed_features
        self.n_random_features = n_random_features
        
    def get_model(self) -> MixedEffectsModel:
        """Get the global model."""
        return self.model
    
    def serialize(self) -> bytes:
        """Serialize model to bytes for transmission."""
        return json.dumps({
            'fixed_weights': self.model.get_fixed_weights().tolist(),
            'random_weights': self.model.get_random_weights().tolist(),
            'n_fixed_features': self.n_fixed_features,
            'n_random_features': self.n_random_features
        }).encode()
    
    def deserialize(self, data: bytes):
        """Deserialize model from bytes."""
        params = json.loads(data.decode())
        self.model.fixed_weights.weight.data = torch.tensor(params['fixed_weights'])
        self.model.random_weights.weight.data = torch.tensor(params['random_weights'])
        
    def save(self, path: Path):
        """Save model to file."""
        torch.save(self.model.state_dict(), path)
        
    def load(self, path: Path):
        """Load model from file."""
        self.model.load_state_dict(torch.load(path))


class FedAvgAggregator:
    """FedAvg aggregation for mixed-effects models."""
    
    def __init__(self):
        self.worker_updates: Dict[str, torch.Tensor] = {}
        
    def add_update(self, worker_id: str, random_weights: torch.Tensor):
        """Add a worker's update to the aggregation pool."""
        self.worker_updates[worker_id] = random_weights.clone()
        
    def aggregate(self) -> torch.Tensor:
        """Aggregate all worker updates using FedAvg (simple averaging)."""
        if not self.worker_updates:
            raise ValueError("No worker updates to aggregate")
        
        # Stack all random weights
        weights_stack = torch.stack(list(self.worker_updates.values()))
        
        # Average across workers (FedAvg)
        avg_weights = torch.mean(weights_stack, dim=0)
        
        return avg_weights
    
    def clear(self):
        """Clear accumulated updates."""
        self.worker_updates.clear()
        
    def get_n_workers(self) -> int:
        """Get number of workers that have submitted updates."""
        return len(self.worker_updates)


class RoundManager:
    """Manages federated training rounds."""
    
    def __init__(self, n_rounds: int, n_workers: int):
        self.n_rounds = n_rounds
        self.n_workers = n_workers
        self.current_round = 0
        self.round_metrics: List[dict] = []
        
    def start_round(self) -> int:
        """Start a new round and return round number."""
        self.current_round += 1
        print(f"\n[RoundManager] Starting round {self.current_round}/{self.n_rounds}")
        return self.current_round
    
    def complete_round(self, avg_loss: float, n_workers: int):
        """Complete current round and log metrics."""
        self.round_metrics.append({
            'round': self.current_round,
            'avg_loss': avg_loss,
            'n_workers': n_workers,
            'timestamp': time.time()
        })
        print(f"[RoundManager] Round {self.current_round} complete: avg_loss={avg_loss:.4f}")
        
    def is_training_complete(self) -> bool:
        """Check if all rounds are complete."""
        return self.current_round >= self.n_rounds
    
    def get_metrics(self) -> List[dict]:
        """Get all round metrics."""
        return self.round_metrics


class WeightDistributor:
    """Handles weight distribution to workers."""
    
    def __init__(self):
        self.worker_connections: Dict[str, socket.socket] = {}
        
    def register_worker(self, worker_id: str, connection: socket.socket):
        """Register a worker connection."""
        self.worker_connections[worker_id] = connection
        
    def broadcast_weights(self, model_data: bytes):
        """Broadcast model weights to all registered workers."""
        for worker_id, conn in self.worker_connections.items():
            try:
                conn.send(model_data)
            except Exception as e:
                print(f"[WeightDistributor] Error sending to {worker_id}: {e}")
                
    def get_n_workers(self) -> int:
        """Get number of registered workers."""
        return len(self.worker_connections)


class Hub:
    """Hub server for federated learning with mixed-effects models."""
    
    def __init__(self, 
                 port: int = 8080, 
                 host: str = "0.0.0.0",
                 n_workers: int = 2, 
                 n_fixed_features: int = 10, 
                 n_random_features: int = 5, 
                 n_rounds: int = 5, 
                 n_local_epochs: int = 10):
        self.port = port
        self.host = host
        self.n_workers = n_workers
        self.n_fixed_features = n_fixed_features
        self.n_random_features = n_random_features
        self.n_rounds = n_rounds
        self.n_local_epochs = n_local_epochs
        
        # Initialize components
        self.global_model = GlobalModel(n_fixed_features, n_random_features)
        self.aggregator = FedAvgAggregator()
        self.round_manager = RoundManager(n_rounds, n_workers)
        self.weight_distributor = WeightDistributor()
        
        # Worker connections
        self.workers: Dict[str, socket.socket] = {}
        
    def start(self):
        """Start the hub server."""
        print(f"[Hub] Starting server on {self.host}:{self.port}")
        print(f"[Hub] Waiting for {self.n_workers} workers...")
        print(f"[Hub] Configuration: {self.n_rounds} rounds, {self.n_local_epochs} local epochs")
        
        # Create socket server
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(self.n_workers)
        
        # Accept worker connections
        while len(self.workers) < self.n_workers:
            client, address = server.accept()
            worker_id = client.recv(1024).decode()
            self.workers[worker_id] = client
            self.weight_distributor.register_worker(worker_id, client)
            print(f"[Hub] Worker {worker_id} connected from {address}")
            
            # Send global model to worker
            model_data = self.global_model.serialize()
            client.send(model_data)
        
        print(f"[Hub] All {self.n_workers} workers connected")
        
        # Run federated training
        while not self.round_manager.is_training_complete():
            round_num = self.round_manager.start_round()
            
            # Collect updates from all workers
            self.collect_updates()
            
            # Aggregate updates (FedAvg)
            avg_weights = self.aggregate_updates()
            
            # Update global model
            self.global_model.get_model().set_random_weights(avg_weights)
            
            # Broadcast updated model to all workers
            self.broadcast_model()
            
            # Calculate and log round metrics
            avg_loss = self.calculate_round_metrics()
            self.round_manager.complete_round(avg_loss, self.aggregator.get_n_workers())
            
            # Clear aggregator for next round
            self.aggregator.clear()
        
        # Save final model and metrics
        self.save_results()
        
        # Wait a moment for workers to receive final update
        time.sleep(1)
        
        # Close connections
        self.shutdown()
        
        print("[Hub] Training complete")
        
    def collect_updates(self):
        """Collect model updates from all workers."""
        print(f"[Hub] Collecting updates from {len(self.workers)} workers...")
        
        for worker_id, client in self.workers.items():
            try:
                # Receive update from worker
                update_data = client.recv(8192)
                update = self.deserialize_update(update_data)
                
                # Add to aggregator
                self.aggregator.add_update(worker_id, update['random_weights'])
                
            except Exception as e:
                print(f"[Hub] Error collecting update from {worker_id}: {e}")
        
        print(f"[Hub] Collected {self.aggregator.get_n_workers()} updates")
        
    def aggregate_updates(self) -> torch.Tensor:
        """Aggregate worker updates using FedAvg."""
        print("[Hub] Aggregating updates (FedAvg)...")
        
        avg_weights = self.aggregator.aggregate()
        
        print(f"[Hub] Aggregated {self.aggregator.get_n_workers()} worker updates")
        
        return avg_weights
        
    def broadcast_model(self):
        """Send updated global model to all workers."""
        print("[Hub] Broadcasting updated model...")
        
        model_data = self.global_model.serialize()
        self.weight_distributor.broadcast_weights(model_data)
        
        print(f"[Hub] Broadcasted model to {self.weight_distributor.get_n_workers()} workers")
        
    def deserialize_update(self, data: bytes) -> dict:
        """Deserialize worker update from bytes."""
        update = json.loads(data.decode())
        return {
            'random_weights': torch.tensor(update['random_weights']),
            'metrics': update.get('metrics', {})
        }
        
    def calculate_round_metrics(self) -> float:
        """Calculate average loss for the current round."""
        # This is a placeholder - in real implementation, workers would send metrics
        # For now, we'll use a dummy value
        return 0.0
        
    def save_results(self):
        """Save final model and training metrics."""
        # Save model
        model_path = Path("hub_model.pt")
        self.global_model.save(model_path)
        print(f"[Hub] Model saved to {model_path}")
        
        # Save metrics
        metrics_path = Path("hub_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.round_manager.get_metrics(), f, indent=2)
        print(f"[Hub] Metrics saved to {metrics_path}")
        
    def shutdown(self):
        """Shutdown the hub server."""
        print("[Hub] Shutting down...")
        
        # Close worker connections
        for worker_id, client in self.workers.items():
            try:
                client.close()
            except:
                pass
        
        self.workers.clear()
        print("[Hub] Shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="Nvidia FLARE Hub for mixed-effects models")
    parser.add_argument("--port", type=int, default=8080, help="Hub port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Hub host (default: 0.0.0.0 for Docker)")
    parser.add_argument("--n-workers", type=int, default=2, help="Number of workers to wait for")
    parser.add_argument("--n-rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--n-local-epochs", type=int, default=10, help="Local epochs per worker")
    parser.add_argument("--n-fixed-features", type=int, default=10, help="Number of fixed effect features")
    parser.add_argument("--n-random-features", type=int, default=5, help="Number of random effect features")
    args = parser.parse_args()
    
    hub = Hub(
        port=args.port,
        host=args.host,
        n_workers=args.n_workers,
        n_fixed_features=args.n_fixed_features,
        n_random_features=args.n_random_features,
        n_rounds=args.n_rounds,
        n_local_epochs=args.n_local_epochs
    )
    
    hub.start()


if __name__ == "__main__":
    main()