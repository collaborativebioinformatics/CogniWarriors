# CogniWarriors FLARE Training Monitor

Federated multimodal training for longitudinal imaging and phenotype data using
NVIDIA FLARE. The current runnable path is the FLARE dashboard in
`../training_docker_v1`.

## Current Architecture

```text
Browser dashboard
      |
      v
training-monitor container
      |
      v
NVIDIA FLARE simulator
      |
      +--> site-1 local training data
      +--> site-2 local training data
      +--> optional additional sites
      |
      v
FedAvg global model + held-out test evaluation
```

The dashboard controls the federation node. It chooses the number of sites,
rounds, local epochs, split strategy, loss mode, and FedProx strength. Each
FLARE run prepares site-local data, trains local models, aggregates a global
model, evaluates the global model on the held-out test set, and displays
global and per-site losses.

## Quick Start

Start Docker Desktop, then run:

```powershell
cd C:\Users\Lenovo\Desktop\nvidia-hackathon\Longitudinal_imaging_to_multimodality\training_docker_v1
docker compose up --build -d
```

Open:

```text
http://localhost:8080
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8080/api/flare/status
```

## Dashboard Workflow

1. Open `http://localhost:8080`.
2. Set:
   - `Sites`: number of FLARE centers to simulate
   - `Rounds`: FedAvg communication rounds
   - `Local epochs`: local training epochs per site per round
   - `Split`: `IID` or `Age`
   - `Loss`: `MSE` or `LMMNN`
   - `FedProx mu`: `0` for standard FedAvg
3. Click `Start FLARE Run`.
4. Review:
   - Global held-out test metrics
   - Global validation MSE by round
   - Separate train MSE for each site model
   - Final model paths and run summary

For LMMNN runs, dashboard `train_loss` is shown as nonnegative train MSE. The
LMMNN optimization objective is reported separately as `train_objective`.

## Repository Layout

```text
.
|-- README.md                  # Original project README kept at repo root
|-- LICENSE
|-- training_docker_v1/         # Active Docker + FLARE dashboard stack
|-- training/                   # Core model, data, and training utilities
|   `-- README.md              # This FLARE training overview
|-- mri_preprocessing/          # MRI preprocessing helpers
`-- pending_cleanup/            # Preserved legacy files and generated artifacts
```

## Active Runbook

For detailed commands, API order, smoke tests, output locations, restart
commands, and troubleshooting, use:

```text
../training_docker_v1/README.md
```

## Important Outputs

When started from `training_docker_v1`, FLARE outputs are written to:

```text
../training_docker_v1/flare_outputs
```

Important files:

- `flare_outputs/latest_summary.json`: latest dashboard summary
- `flare_outputs/latest_run.log`: latest run log
- `flare_outputs/runs/<run_id>/summary.json`: full run summary
- `flare_outputs/runs/<run_id>/workspace`: FLARE simulator workspace
- `FL_global_model.pt`: final global model
- `best_FL_global_model.pt`: best global model when emitted by FLARE

Generated FLARE outputs are ignored by Git.

## Pending Cleanup

Old REST federation prototypes, one-off reports, generated models, extra
Markdown notes, and duplicate top-level FLARE references were moved into:

```text
../pending_cleanup/
```

Nothing in `pending_cleanup` is part of the active Docker dashboard path. It is
kept for review before final removal.
