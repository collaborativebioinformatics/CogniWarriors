# Federated Training Heads

This directory now supports two independent Training Head containers:

- `training-head-1` serves Center 1 on port `8001`.
- `training-head-2` serves Center 2 on port `8002`.

Both services run the same `training_head` implementation. They differ only by environment configuration and mounted center-local data.

## Split Local Data

Create the two center data roots from the full local data directory:

```bash
python scripts/split_center_data.py --source data --output data/centers
```

The split is participant-based, so the same participant never appears in both centers.

## Run Both Heads

```bash
docker compose -f docker-compose.training-heads.yml up --build
```

Health checks:

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
```

## API

Each Training Head exposes:

- `GET /health`
- `POST /initialize`
- `POST /train`
- `GET /status`
- `GET /weights`
- `POST /evaluate`

Weights use `torch_state_dict_base64`. Raw center data is mounted only into the corresponding Training Head container and is never returned by the API.
