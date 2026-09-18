#!/usr/bin/env python3
"""Simple sequential test for mixed-effects model."""

import json
import torch
import torch.nn as nn
from pathlib import Path


class MixedEffectsModel(nn.Module):
    def __init__(self, n_fixed_features, n_random_features):
        super().__init__()
        self.fixed_weights = nn.Linear(n_fixed_features, 1, bias=False)
        self.random_weights = nn.Linear(n_random_features, 1, bias=False)
        
    def forward(self, x_fixed, x_random):
        fixed_output = self.fixed_weights(x_fixed)
        random_output = self.random_weights(x_random)
        return (fixed_output + random_output).squeeze(-1)


def simulate_federated_training():
    """Simulate federated training without actual networking."""
    print("Simulating Federated Learning with Mixed-Effects Models...")
    
    # Create global model (hub)
    global_model = MixedEffectsModel(n_fixed_features=10, n_random_features=5)
    print(f"Global model created: {global_model}")
    
    # Simulate 2 workers
    n_workers = 2
    n_rounds = 3
    
    for round_num in range(n_rounds):
        print(f"\nRound {round_num + 1}/{n_rounds}")
        
        # Collect worker updates
        worker_updates = []
        
        for worker_id in range(n_workers):
            # Create worker model (copy of global)
            worker_model = MixedEffectsModel(n_fixed_features=10, n_random_features=5)
            worker_model.load_state_dict(global_model.state_dict())
            
            # Generate random data
            x_fixed = torch.randn(100, 10)
            x_random = torch.randn(100, 5)
            y_true = torch.randn(100)
            
            # Local training (only update random effects)
            optimizer = torch.optim.Adam(worker_model.random_weights.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()
            
            for epoch in range(5):
                worker_model.train()
                optimizer.zero_grad()
                
                y_pred = worker_model(x_fixed, x_random)
                loss = loss_fn(y_pred, y_true)
                loss.backward()
                optimizer.step()
            
            print(f"Worker {worker_id + 1}: final loss = {loss.item():.4f}")
            
            # Store worker's random weights
            worker_updates.append(worker_model.random_weights.weight.data.clone())
        
        # Aggregate updates (average random effects)
        avg_random_weights = torch.mean(torch.stack(worker_updates), dim=0)
        global_model.random_weights.weight.data = avg_random_weights
        
        print(f"Global model updated (aggregated from {n_workers} workers)")
    
    # Save final model
    torch.save(global_model.state_dict(), "hub_model.pt")
    print(f"\nFinal model saved to hub_model.pt")
    
    # Save metrics
    metrics = {
        "n_rounds": n_rounds,
        "n_workers": n_workers,
        "final_model": {
            "fixed_weights": global_model.fixed_weights.weight.data.tolist(),
            "random_weights": global_model.random_weights.weight.data.tolist()
        }
    }
    
    with open("hub_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Metrics saved to hub_metrics.json")
    
    # Create worker reports
    for worker_id in range(n_workers):
        report = {
            "worker_id": f"worker_{worker_id + 1:02d}",
            "n_rounds": n_rounds,
            "final_loss": loss.item()
        }
        
        with open(f"worker_{worker_id + 1:02d}_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Worker {worker_id + 1} report saved to worker_{worker_id + 1:02d}_report.json")
    
    print("\n✓ Simulation completed!")


if __name__ == "__main__":
    simulate_federated_training()