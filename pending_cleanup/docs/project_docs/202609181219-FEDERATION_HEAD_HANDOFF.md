# Federation Head Handoff

This machine already runs two Dockerized Training Heads. The Federation Head should orchestrate them, but it must not access raw center data directly.

## What To Build

Build one Federation Head service that:

1. Stores the current global model weights and federated round number.
2. Calls both Training Heads for each federated round.
3. Starts local training on both centers.
4. Polls each center until local training finishes.
5. Retrieves local model weights and sample counts.
6. Aggregates local weights using FedAvg.
7. Sends the new global weights back to both centers on the next round.

The Federation Head owns aggregation. The Training Heads only train locally.

## Training Head URLs

If the Federation Head runs directly on this Windows host:

```text
Center 1: http://localhost:8001
Center 2: http://localhost:8002
```

If the Federation Head runs as a Docker container on the same Docker Compose network:

```text
Center 1: http://training-head-1:8001
Center 2: http://training-head-2:8002
```

Make these URLs configurable. Do not hard-code `localhost`.

## API Call Order

For every federated round, call the Training Heads in this order.

### 1. Health Check

Call both centers:

```http
GET /health
```

Example:

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
```

Expected response:

```json
{
  "status": "healthy",
  "center_id": "center1",
  "data_dir": "/data/center1",
  "device": "cpu"
}
```

Continue only if both centers are healthy.

### 2. Initialize Local Models

Send the current global model weights to both centers:

```http
POST /initialize
```

Round 1 can use `weights: null` to let each Training Head initialize the shared model architecture locally.

```json
{
  "round": 1,
  "model_version": "global_v1",
  "weights": null,
  "weights_format": "torch_state_dict_base64"
}
```

For later rounds, send the aggregated global weights:

```json
{
  "round": 2,
  "model_version": "global_v2",
  "weights": "<base64 encoded torch state_dict>",
  "weights_format": "torch_state_dict_base64"
}
```

### 3. Start Local Training

Call both centers:

```http
POST /train
```

Example payload:

```json
{
  "round": 1,
  "model_version": "global_v1",
  "epochs": 10
}
```

Expected response:

```json
{
  "status": "training_started",
  "center_id": "center1",
  "round": 1
}
```

Important: `/train` is asynchronous. It returns immediately while local training continues in the background.

### 4. Poll Training Status

Poll both centers until each reports `completed` or `failed`:

```http
GET /status
```

Example while training:

```json
{
  "center_id": "center1",
  "round": 1,
  "model_version": "global_v1",
  "status": "training",
  "epoch": 5,
  "total_epochs": 10,
  "train_loss": 0.124,
  "validation_loss": 0.141,
  "num_samples": 82,
  "metrics": {
    "train_loss": 0.124,
    "loss": 0.141,
    "mae": 0.29
  },
  "checkpoint_path": null,
  "error": null
}
```

Supported status values:

```text
idle
initializing
training
completed
failed
```

If either center returns `failed`, stop the round and surface the `error` field.

### 5. Retrieve Local Weights

After both centers complete:

```http
GET /weights
```

Example response:

```json
{
  "center_id": "center1",
  "round": 1,
  "num_samples": 82,
  "weights": "<base64 encoded torch state_dict>",
  "weights_format": "torch_state_dict_base64",
  "metrics": {
    "train_loss": 1.2661098683321919,
    "loss": 0.44610968232154846,
    "mae": 0.535914957523346
  },
  "model_version": "global_v1"
}
```

Use `num_samples` from this response for FedAvg. Do not assume the sample counts are fixed.

### 6. Aggregate With FedAvg

Decode each returned `weights` value into a PyTorch `state_dict`.

FedAvg:

```text
W_global = (N1 * W1 + N2 * W2) / (N1 + N2)
```

Where:

```text
W1 = center1 local state_dict
W2 = center2 local state_dict
N1 = center1 num_samples from /weights
N2 = center2 num_samples from /weights
```

Aggregate each tensor key independently.

### 7. Start Next Round

Encode the aggregated global `state_dict` as `torch_state_dict_base64`, increment the round number, and call `/initialize` again on both centers.

## Weight Serialization

The Training Heads use this format:

```text
weights_format = torch_state_dict_base64
```

Python helpers for the Federation Head:

```python
import base64
import io
import torch


def encode_state_dict(state_dict):
    buffer = io.BytesIO()
    cpu_state = {key: value.detach().cpu() for key, value in state_dict.items()}
    torch.save(cpu_state, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_state_dict(payload):
    raw = base64.b64decode(payload.encode("ascii"), validate=True)
    buffer = io.BytesIO(raw)
    try:
        return torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu")
```

FedAvg helper:

```python
def fedavg(weight_payloads):
    """
    weight_payloads: list of dictionaries from GET /weights.
    Each item must contain: weights, num_samples.
    """
    decoded = [
        (decode_state_dict(item["weights"]), int(item["num_samples"]))
        for item in weight_payloads
    ]
    total_samples = sum(num_samples for _, num_samples in decoded)
    if total_samples <= 0:
        raise ValueError("FedAvg requires at least one sample")

    global_state = {}
    first_state = decoded[0][0]
    for key in first_state:
        global_state[key] = sum(
            state[key].float() * (num_samples / total_samples)
            for state, num_samples in decoded
        )
    return global_state
```

## Optional Evaluation

The Federation Head can ask each center to evaluate locally:

```http
POST /evaluate
```

Payload:

```json
{
  "split": "validation"
}
```

Response:

```json
{
  "center_id": "center1",
  "round": 1,
  "metrics": {
    "loss": 0.44610968232154846,
    "mae": 0.535914957523346
  }
}
```

Allowed `split` values:

```text
validation
train
```

## Error Handling Requirements

The Federation Head should handle:

- A Training Head not reachable during `/health`.
- `/initialize` returning an error for invalid weights.
- `/train` returning `409` because training is already running.
- `/status` returning `failed`.
- `/weights` returning `409` because training is not complete.
- Round mismatch between centers.
- Missing or malformed `weights_format`.

## Local Smoke Test Commands

From this directory:

```bash
docker compose -f docker-compose.training-heads.yml ps
curl http://localhost:8001/health
curl http://localhost:8002/health
```

Start a one-epoch smoke round manually:

```bash
curl -X POST http://localhost:8001/initialize \
  -H "Content-Type: application/json" \
  -d '{"round":1,"model_version":"smoke_v1","weights":null,"weights_format":"torch_state_dict_base64"}'

curl -X POST http://localhost:8002/initialize \
  -H "Content-Type: application/json" \
  -d '{"round":1,"model_version":"smoke_v1","weights":null,"weights_format":"torch_state_dict_base64"}'

curl -X POST http://localhost:8001/train \
  -H "Content-Type: application/json" \
  -d '{"round":1,"model_version":"smoke_v1","epochs":1}'

curl -X POST http://localhost:8002/train \
  -H "Content-Type: application/json" \
  -d '{"round":1,"model_version":"smoke_v1","epochs":1}'
```

Then poll:

```bash
curl http://localhost:8001/status
curl http://localhost:8002/status
```

After both say `completed`:

```bash
curl http://localhost:8001/weights
curl http://localhost:8002/weights
```

## Privacy Boundary

The Federation Head must never read or mount:

```text
data/centers/center1
data/centers/center2
```

Those directories belong only to their corresponding Training Head containers. The Federation Head should only receive:

- center ID
- round
- model version
- sample count
- metrics
- model weights

