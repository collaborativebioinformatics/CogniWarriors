# FLARE Training Monitor Runbook

This folder runs the NVIDIA FLARE federation workflow and the web dashboard.
The active Docker stack is FLARE-only. The older REST federation-head and
training-head services are intentionally not part of this runnable stack.

## What Runs

- `training-monitor`: FastAPI dashboard and FLARE runner on port `8080`
- NVIDIA FLARE simulator: launched by the dashboard or by CLI smoke commands
- Local site data split: created per run under `flare_outputs/runs/<run_id>/site_data`
- FLARE workspace and global models: created per run under `flare_outputs/runs/<run_id>/workspace`

## Prerequisites

1. Start Docker Desktop.
2. Make sure at least one runnable copy exists.

Active demo folder:

```text
C:\Users\Lenovo\Desktop\nvidia-hackathon\training_docker_v1
```

Repo branch copy:

```text
C:\Users\Lenovo\Desktop\nvidia-hackathon\Longitudinal_imaging_to_multimodality\training_docker_v1
```

3. Make sure the selected folder has a data folder:

```text
.\data
```

The Docker image installs Python, PyTorch CPU, FastAPI, and `nvflare==2.9.0`.
You do not need a local Python install to run the dashboard.

## Start Everything

Run from the active demo folder:

```powershell
cd C:\Users\Lenovo\Desktop\nvidia-hackathon\training_docker_v1
docker compose up --build -d
```

Or run from the repo branch copy:

```powershell
cd C:\Users\Lenovo\Desktop\nvidia-hackathon\Longitudinal_imaging_to_multimodality\training_docker_v1
docker compose up --build -d
```

Open the dashboard:

```text
http://localhost:8080
```

Check that the container is running:

```powershell
docker ps --filter "name=training-monitor"
```

## Use the Dashboard

1. Open `http://localhost:8080`.
2. Set the run controls:
   - `Sites`: number of FLARE sites to split data into
   - `Rounds`: federated averaging rounds
   - `Local epochs`: local training epochs per site per round
   - `Split`: `IID` or `Age`
   - `Loss`: `MSE` or `LMMNN`
   - `FedProx mu`: use `0` for plain FedAvg
3. Click `Start FLARE Run`.
4. Watch:
   - `Global Model Evaluation`: held-out test metrics
   - `Losses by Round`: global validation MSE and each site train MSE
   - `Target Metrics`: final target MSE, MAE, and R2
   - `Details`: run id, model paths, and output folder

For LMMNN runs, the dashboard shows `train_loss` as nonnegative train MSE.
The LMMNN optimization objective is reported separately as `train_objective`.

## API Order

The dashboard calls the FLARE APIs in this order.

1. Check status:

```powershell
Invoke-RestMethod http://localhost:8080/api/flare/status
```

2. Start a run:

```powershell
$body = @{
  n_sites = 2
  rounds = 1
  local_epochs = 1
  split = "iid"
  loss = "mse"
  mu = 0
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://localhost:8080/api/flare/run `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

3. Poll status until `job.running` is `false`:

```powershell
Invoke-RestMethod http://localhost:8080/api/flare/status
```

## CLI Smoke Tests

Run a quick MSE smoke test:

```powershell
docker compose run --rm training-monitor python -m federated.flare.run_flare_sim `
  --n-sites 2 `
  --rounds 1 `
  --local-epochs 1 `
  --split iid `
  --loss mse `
  --mu 0
```

Run a quick LMMNN smoke test:

```powershell
docker compose run --rm training-monitor python -m federated.flare.run_flare_sim `
  --n-sites 2 `
  --rounds 1 `
  --local-epochs 2 `
  --split iid `
  --loss lmmnn `
  --mu 0
```

Check Python syntax inside the Docker image:

```powershell
docker run --rm nvidia-hackathon-flare-dashboard:latest python -m py_compile `
  monitor_app/main.py `
  federated/flare/client.py `
  federated/flare/controller.py `
  federated/flare/evaluate_global.py `
  federated/flare/job.py `
  federated/flare/prepare_site_data.py `
  federated/flare/run_flare_sim.py
```

## Output Files

Outputs are written under the folder you started Docker from.

Active demo folder:

```text
C:\Users\Lenovo\Desktop\nvidia-hackathon\training_docker_v1\flare_outputs
```

Repo branch copy:

```text
C:\Users\Lenovo\Desktop\nvidia-hackathon\Longitudinal_imaging_to_multimodality\training_docker_v1\flare_outputs
```

Important files:

- `flare_outputs/latest_summary.json`: latest run summary read by the dashboard
- `flare_outputs/latest_run.log`: latest run console output
- `flare_outputs/runs/<run_id>/summary.json`: full summary for one run
- `flare_outputs/runs/<run_id>/site_data`: generated site train/val/test splits
- `flare_outputs/runs/<run_id>/workspace`: FLARE simulator workspace
- `FL_global_model.pt`: final global model
- `best_FL_global_model.pt`: best global model when FLARE emits one

Generated outputs are ignored by Git.

## Stop or Restart

Stop the dashboard:

```powershell
docker compose down
```

Rebuild from scratch:

```powershell
docker compose build --no-cache training-monitor
docker compose up -d
```

View logs:

```powershell
docker compose logs -f training-monitor
```

## Troubleshooting

Port `8080` is already in use:

```powershell
docker ps --format "table {{.Names}}\t{{.Ports}}"
```

Then stop the conflicting container or change the compose port mapping.

Dashboard opens but no metrics appear:

```powershell
Invoke-RestMethod http://localhost:8080/api/flare/status
```

Check `summary.error` and `flare_outputs/latest_run.log`.

Docker build is stale:

```powershell
docker compose up --build -d --remove-orphans
```

Need a clean output folder:

```powershell
docker compose down
```

Then remove old files under the selected folder's `flare_outputs` directory.

```text
C:\Users\Lenovo\Desktop\nvidia-hackathon\training_docker_v1\flare_outputs
```

## Repo Branch Copy

The source repo copy lives here:

```text
C:\Users\Lenovo\Desktop\nvidia-hackathon\Longitudinal_imaging_to_multimodality\training_docker_v1
```

The refactor branch is:

```text
vb-refactor
```

The active runnable folder remains:

```text
C:\Users\Lenovo\Desktop\nvidia-hackathon\training_docker_v1
```
