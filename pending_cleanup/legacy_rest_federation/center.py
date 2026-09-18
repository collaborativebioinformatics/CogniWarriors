#!/usr/bin/env python3
"""Federation Head for federated learning with mixed-effects models.

Orchestrates Training Heads via HTTP REST API calls.
Implements FedAvg aggregation with sample-weighted averaging.

API Call Order per round:
1. Health Check (GET /health)
2. Initialize Local Models (POST /initialize)
3. Start Local Training (POST /train)
4. Poll Training Status (GET /status)
5. Retrieve Local Weights (GET /weights)
6. Aggregate With FedAvg
7. Start Next Round

Usage:
    python center.py --rounds 5 --epochs 10
"""

import argparse
import base64
import io
import json
import time
from pathlib import Path
from typing import List, Optional

import requests
import torch
import torch.nn as nn


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


def fedavg(weight_payloads: List[dict]) -> dict:
    decoded = [
        (decode_state_dict(item["weights"]), int(item["num_samples"]))
        for item in weight_payloads
    ]
    total_samples = sum(n for _, n in decoded)
    if total_samples <= 0:
        raise ValueError("FedAvg requires at least one sample")
    global_state = {}
    for key in decoded[0][0]:
        global_state[key] = sum(
            state[key].float() * (n / total_samples) for state, n in decoded
        )
    return global_state


class FederationHead:
    def __init__(self, training_head_urls, n_rounds=5, n_epochs=10,
                 n_fixed_features=10, n_random_features=5, poll_interval=1.0):
        self.training_head_urls = training_head_urls
        self.n_rounds = n_rounds
        self.n_epochs = n_epochs
        self.n_fixed_features = n_fixed_features
        self.n_random_features = n_random_features
        self.poll_interval = poll_interval
        self.global_model = MixedEffectsModel(n_fixed_features, n_random_features)
        self.current_round = 0
        self.model_version = "global_v0"
        self.round_metrics = []

    def run(self):
        print(f"[Federation Head] Training Heads: {self.training_head_urls}")
        print(f"[Federation Head] Rounds: {self.n_rounds}, Epochs: {self.n_epochs}")

        for round_num in range(1, self.n_rounds + 1):
            print(f"\n{'='*60}")
            print(f"[Federation Head] Round {round_num}/{self.n_rounds}")
            print(f"{'='*60}")

            self.current_round = round_num
            self.model_version = f"global_v{round_num}"

            if not self._health_check():
                print("[Federation Head] Health check failed. Aborting.")
                break
            if not self._initialize_models():
                print("[Federation Head] Initialize failed. Aborting.")
                break
            if not self._start_training():
                print("[Federation Head] Start training failed. Aborting.")
                break
            if not self._poll_training():
                print("[Federation Head] Training failed. Aborting.")
                break
            payloads = self._retrieve_weights()
            if not payloads:
                print("[Federation Head] Retrieve weights failed. Aborting.")
                break
            self._aggregate_weights(payloads)
            self._log_round(payloads)

        self._save_results()
        print(f"\n{'='*60}")
        print("[Federation Head] Training complete")
        print(f"{'='*60}")

    def _health_check(self) -> bool:
        print("\n[FH] Step 1: Health Check")
        ok = True
        for url in self.training_head_urls:
            try:
                r = requests.get(f"{url}/health", timeout=5)
                if r.status_code == 200:
                    print(f"  OK {url}: {r.json().get('status')}")
                else:
                    print(f"  FAIL {url}: HTTP {r.status_code}")
                    ok = False
            except requests.RequestException as e:
                print(f"  FAIL {url}: {e}")
                ok = False
        return ok

    def _initialize_models(self) -> bool:
        print("\n[FH] Step 2: Initialize Models")
        weights = None if self.current_round == 1 else encode_state_dict(self.global_model.state_dict())
        payload = {
            "round": self.current_round,
            "model_version": self.model_version,
            "weights": weights,
            "weights_format": "torch_state_dict_base64",
        }
        ok = True
        for url in self.training_head_urls:
            try:
                r = requests.post(f"{url}/initialize", json=payload, timeout=10)
                if r.status_code == 200:
                    print(f"  OK {url}: {r.json().get('status')}")
                else:
                    print(f"  FAIL {url}: HTTP {r.status_code}")
                    ok = False
            except requests.RequestException as e:
                print(f"  FAIL {url}: {e}")
                ok = False
        return ok

    def _start_training(self) -> bool:
        print("\n[FH] Step 3: Start Training")
        payload = {"round": self.current_round, "model_version": self.model_version, "epochs": self.n_epochs}
        ok = True
        for url in self.training_head_urls:
            try:
                r = requests.post(f"{url}/train", json=payload, timeout=10)
                if r.status_code == 200:
                    print(f"  OK {url}: {r.json().get('status')}")
                else:
                    print(f"  FAIL {url}: HTTP {r.status_code}")
                    ok = False
            except requests.RequestException as e:
                print(f"  FAIL {url}: {e}")
                ok = False
        return ok

    def _poll_training(self) -> bool:
        print("\n[FH] Step 4: Poll Status")
        status = {u: "training" for u in self.training_head_urls}
        pending = set(self.training_head_urls)

        while pending:
            time.sleep(self.poll_interval)
            for url in list(pending):
                try:
                    r = requests.get(f"{url}/status", timeout=5)
                    if r.status_code == 200:
                        d = r.json()
                        s = d.get("status", "unknown")
                        status[url] = s
                        epoch = d.get("epoch", "?")
                        total = d.get("total_epochs", "?")
                        loss = d.get("train_loss", "?")
                        if s == "training":
                            print(f"  {url}: epoch {epoch}/{total}, loss={loss}")
                        elif s == "completed":
                            print(f"  OK {url}: completed")
                            pending.discard(url)
                        elif s == "failed":
                            print(f"  FAIL {url}: {d.get('error')}")
                            pending.discard(url)
                except requests.RequestException as e:
                    print(f"  FAIL {url}: {e}")
                    status[url] = "failed"
                    pending.discard(url)

        return all(s == "completed" for s in status.values())

    def _retrieve_weights(self) -> List[dict]:
        print("\n[FH] Step 5: Retrieve Weights")
        payloads = []
        for url in self.training_head_urls:
            try:
                r = requests.get(f"{url}/weights", timeout=10)
                if r.status_code == 200:
                    d = r.json()
                    payloads.append(d)
                    print(f"  OK {url}: {d.get('num_samples')} samples")
                else:
                    print(f"  FAIL {url}: HTTP {r.status_code}")
            except requests.RequestException as e:
                print(f"  FAIL {url}: {e}")
        return payloads

    def _aggregate_weights(self, payloads: List[dict]):
        print("\n[FH] Step 6: FedAvg Aggregation")
        global_state = fedavg(payloads)
        self.global_model.load_state_dict(global_state)
        print(f"  Aggregated from {len(payloads)} Training Heads")

    def _log_round(self, payloads: List[dict]):
        total_samples = sum(p.get("num_samples", 0) for p in payloads)
        avg_metrics = {}
        for p in payloads:
            for k, v in p.get("metrics", {}).items():
                avg_metrics.setdefault(k, []).append(v)
        avg_metrics = {k: sum(v) / len(v) for k, v in avg_metrics.items()}
        self.round_metrics.append({
            "round": self.current_round,
            "total_samples": total_samples,
            "metrics": avg_metrics,
        })
        print(f"\n  Round {self.current_round} Summary: samples={total_samples}, metrics={avg_metrics}")

    def _save_results(self):
        model_path = Path("federation_model.pt")
        torch.save(self.global_model.state_dict(), model_path)
        print(f"\n[Federation Head] Model saved to {model_path}")
        metrics_path = Path("federation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.round_metrics, f, indent=2)
        print(f"[Federation Head] Metrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Federation Head")
    parser.add_argument("--training-heads", nargs="+",
                        default=["http://localhost:8001", "http://localhost:8002"],
                        help="Training Head URLs")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-fixed-features", type=int, default=10)
    parser.add_argument("--n-random-features", type=int, default=5)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    args = parser.parse_args()

    fh = FederationHead(
        training_head_urls=args.training_heads,
        n_rounds=args.rounds,
        n_epochs=args.epochs,
        n_fixed_features=args.n_fixed_features,
        n_random_features=args.n_random_features,
        poll_interval=args.poll_interval,
    )
    fh.run()


if __name__ == "__main__":
    main()
