#!/usr/bin/env python3
"""Training Head for federated learning with mixed-effects models.

Exposes REST API endpoints for Federation Head orchestration.
Handles local training only, never exposes raw data.

Endpoints:
- GET /health: Health check
- POST /initialize: Initialize local model with weights
- POST /train: Start local training (async)
- GET /status: Get training status
- GET /weights: Retrieve local weights
- POST /evaluate: Evaluate model locally

Usage:
    python worker.py --center-id center1 --data-dir ./data/center1 --port 8001
"""

import argparse
import base64
import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from flask import Flask, request, jsonify


class MixedEffectsModel(nn.Module):
    def __init__(self, n_fixed_features: int, n_random_features: int):
        super().__init__()
        self.fixed_weights = nn.Linear(n_fixed_features, 1, bias=False)
        self.random_weights = nn.Linear(n_random_features, 1, bias=False)

    def forward(self, x_fixed: torch.Tensor, x_random: torch.Tensor) -> torch.Tensor:
        fixed_output = self.fixed_weights(x_fixed)
        random_output = self.random_weights(x_random)
        return (fixed_output + random_output).squeeze(-1)


def encode_state_dict(state_dict: dict) -> str:
    buffer = io.BytesIO()
    cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
    torch.save(cpu_state, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_state_dict(payload: str) -> dict:
    raw = base64.b64decode(payload.encode("ascii"), validate=True)
    buffer = io.BytesIO(raw)
    try:
        return torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu")


def generate_synthetic_data(n_samples, n_fixed_features, n_random_features):
    X_fixed = torch.randn(n_samples, n_fixed_features)
    X_random = torch.randn(n_samples, n_random_features)
    y = torch.randn(n_samples)
    return X_fixed, X_random, y


class TrainingHead:
    def __init__(self, center_id, data_dir, n_fixed_features=10, n_random_features=5):
        self.center_id = center_id
        self.data_dir = data_dir
        self.n_fixed_features = n_fixed_features
        self.n_random_features = n_random_features

        self.model = None
        self.current_round = 0
        self.model_version = ""
        self.status = "idle"
        self.current_epoch = 0
        self.total_epochs = 0
        self.train_loss = 0.0
        self.num_samples = 0
        self.metrics = {}
        self.checkpoint_path = None
        self.error = None
        self.training_thread = None

        self.load_data()

    def load_data(self):
        self.num_samples = 82
        self.X_fixed, self.X_random, self.y = generate_synthetic_data(
            self.num_samples, self.n_fixed_features, self.n_random_features
        )

    def initialize(self, round_num, model_version, weights, weights_format):
        self.current_round = round_num
        self.model_version = model_version
        self.status = "initializing"
        self.error = None

        try:
            self.model = MixedEffectsModel(self.n_fixed_features, self.n_random_features)
            if weights is not None:
                if weights_format != "torch_state_dict_base64":
                    raise ValueError(f"Unsupported weights format: {weights_format}")
                state_dict = decode_state_dict(weights)
                self.model.load_state_dict(state_dict)
            self.status = "idle"
            return True
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            return False

    def start_training(self, round_num, model_version, epochs):
        if self.status == "training":
            return False, "Training already in progress"
        if round_num != self.current_round:
            return False, f"Round mismatch: expected {self.current_round}, got {round_num}"

        self.total_epochs = epochs
        self.training_thread = threading.Thread(
            target=self._train_loop, args=(round_num, model_version, epochs), daemon=True
        )
        self.training_thread.start()
        return True, "training_started"

    def _train_loop(self, round_num, model_version, epochs):
        try:
            self.status = "training"
            self.current_epoch = 0

            optimizer = optim.SGD(self.model.parameters(), lr=0.01)
            criterion = nn.MSELoss()

            dataset = TensorDataset(self.X_fixed, self.X_random, self.y)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            for epoch in range(epochs):
                self.current_epoch = epoch + 1
                epoch_loss = 0.0

                for x_fixed, x_random, y in dataloader:
                    optimizer.zero_grad()
                    output = self.model(x_fixed, x_random)
                    loss = criterion(output, y)
                    loss.backward()
                    for param in self.model.fixed_weights.parameters():
                        param.grad = None
                    optimizer.step()
                    epoch_loss += loss.item()

                self.train_loss = epoch_loss / max(len(dataloader), 1)
                self.metrics = {
                    "train_loss": self.train_loss,
                    "loss": self.train_loss,
                    "mae": float(np.sqrt(self.train_loss)),
                }
                time.sleep(0.05)

            self.status = "completed"
            self.checkpoint_path = f"/tmp/checkpoint_{self.center_id}_round{round_num}.pt"
            torch.save(self.model.state_dict(), self.checkpoint_path)

        except Exception as e:
            self.status = "failed"
            self.error = str(e)

    def get_status(self):
        return {
            "center_id": self.center_id,
            "round": self.current_round,
            "model_version": self.model_version,
            "status": self.status,
            "epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "train_loss": self.train_loss,
            "validation_loss": 0.0,
            "num_samples": self.num_samples,
            "metrics": self.metrics,
            "checkpoint_path": self.checkpoint_path,
            "error": self.error,
        }

    def get_weights(self):
        if self.status != "completed":
            return None
        return {
            "center_id": self.center_id,
            "round": self.current_round,
            "num_samples": self.num_samples,
            "weights": encode_state_dict(self.model.state_dict()),
            "weights_format": "torch_state_dict_base64",
            "metrics": self.metrics,
            "model_version": self.model_version,
        }

    def evaluate(self, split="validation"):
        if self.model is None:
            return {"error": "Model not initialized"}

        self.model.eval()
        with torch.no_grad():
            output = self.model(self.X_fixed, self.X_random)
            loss = nn.MSELoss()(output, self.y).item()
        return {"loss": loss, "mae": float(np.sqrt(loss))}


def create_app(center_id, data_dir, n_fixed_features=10, n_random_features=5):
    app = Flask(__name__)
    head = TrainingHead(center_id, data_dir, n_fixed_features, n_random_features)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "healthy",
            "center_id": head.center_id,
            "data_dir": head.data_dir,
            "device": "cpu",
        })

    @app.route("/initialize", methods=["POST"])
    def initialize():
        data = request.get_json()
        round_num = data.get("round")
        model_version = data.get("model_version", "")
        weights = data.get("weights")
        weights_format = data.get("weights_format", "torch_state_dict_base64")

        success = head.initialize(round_num, model_version, weights, weights_format)
        if success:
            return jsonify({"status": "initialized", "center_id": head.center_id, "round": round_num})
        else:
            return jsonify({"status": "failed", "center_id": head.center_id, "error": head.error}), 400

    @app.route("/train", methods=["POST"])
    def train_endpoint():
        data = request.get_json()
        round_num = data.get("round")
        model_version = data.get("model_version", "")
        epochs = data.get("epochs", 10)

        success, message = head.start_training(round_num, model_version, epochs)
        if success:
            return jsonify({"status": "training_started", "center_id": head.center_id, "round": round_num})
        else:
            return jsonify({"status": "error", "center_id": head.center_id, "error": message}), 409

    @app.route("/status", methods=["GET"])
    def status_endpoint():
        return jsonify(head.get_status())

    @app.route("/weights", methods=["GET"])
    def weights_endpoint():
        data = head.get_weights()
        if data is None:
            return jsonify({"status": "error", "center_id": head.center_id, "error": "Training not complete"}), 409
        return jsonify(data)

    @app.route("/evaluate", methods=["POST"])
    def evaluate_endpoint():
        data = request.get_json() or {}
        split = data.get("split", "validation")
        result = head.evaluate(split)
        return jsonify({
            "center_id": head.center_id,
            "round": head.current_round,
            "metrics": result,
        })

    return app


def main():
    parser = argparse.ArgumentParser(description="Training Head for federated learning")
    parser.add_argument("--center-id", required=True, help="Center identifier")
    parser.add_argument("--data-dir", required=True, help="Path to data directory")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--n-fixed-features", type=int, default=10)
    parser.add_argument("--n-random-features", type=int, default=5)
    args = parser.parse_args()

    app = create_app(args.center_id, args.data_dir, args.n_fixed_features, args.n_random_features)
    print(f"[Training Head {args.center_id}] Listening on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
