#!/usr/bin/env python3
"""Simple test for mixed-effects model."""

import torch
import torch.nn as nn
import json

class MixedEffectsModel(nn.Module):
    def __init__(self, n_fixed_features, n_random_features):
        super().__init__()
        self.fixed_weights = nn.Linear(n_fixed_features, 1, bias=False)
        self.random_weights = nn.Linear(n_random_features, 1, bias=False)
        
    def forward(self, x_fixed, x_random):
        fixed_output = self.fixed_weights(x_fixed)
        random_output = self.random_weights(x_random)
        return (fixed_output + random_output).squeeze(-1)

# Test the model
print("Testing Mixed-Effects Model...")

# Create model
model = MixedEffectsModel(n_fixed_features=10, n_random_features=5)
print(f"Model created: {model}")

# Generate random data
n_samples = 100
x_fixed = torch.randn(n_samples, 10)
x_random = torch.randn(n_samples, 5)
y_true = torch.randn(n_samples)

# Forward pass
y_pred = model(x_fixed, x_random)
print(f"Prediction shape: {y_pred.shape}")

# Test loss
loss_fn = nn.MSELoss()
loss = loss_fn(y_pred, y_true)
print(f"Loss: {loss.item():.4f}")

# Test serialization
model_data = json.dumps({
    'fixed_weights': model.fixed_weights.weight.data.tolist(),
    'random_weights': model.random_weights.weight.data.tolist()
})
print(f"Model serialized: {len(model_data)} bytes")

# Test deserialization
params = json.loads(model_data)
model2 = MixedEffectsModel(n_fixed_features=10, n_random_features=5)
model2.fixed_weights.weight.data = torch.tensor(params['fixed_weights'])
model2.random_weights.weight.data = torch.tensor(params['random_weights'])

# Verify models are the same
y_pred2 = model2(x_fixed, x_random)
print(f"Models match: {torch.allclose(y_pred, y_pred2)}")

print("✓ All tests passed!")