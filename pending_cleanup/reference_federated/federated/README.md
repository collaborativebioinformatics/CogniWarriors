# Federated Training with NVIDIA FLARE

Trains the image + phenotype fusion model (`training/fusion_model.py`) across
sites with **FedAvg on NVIDIA FLARE 2.9**. No data leaves a site: sites only
send model weights, metrics, session counts and (once) column sums/counts for
global feature scaling.

## Files

| File | Role |
|---|---|
| `flare/prepare_site_data.py` | Splits the dataset into one folder per site (by participant) + a held-out test set |
| `flare/client.py` | Runs **at each site** (replaces `worker.py`): local training with the NVFlare Client API |
| `flare/controller.py` | Runs **on the server** (replaces `center.py`): global scaling step + NVFlare FedAvg |
| `flare/cogni_model.py` | The shared model: `FusionRegressor` + global scaling buffers (+ optional LMMNN variances) |
| `flare/job.py` | Builds the NVFlare job and runs it (simulator, POC, export, production) |
| `flare/evaluate_global.py` | Scores the saved global model on the held-out test set |
| `flare/project.yml` | Provisioning file for a real multi-machine deployment |
| `flare/Dockerfile` | Runtime image for server and sites (libraries only, no code, no data) |
| `fedavg_simulation.py` | Offline experiment: federated vs. centralized vs. single-site, several seeds |

## How it works

1. **Global scaling (once).** The server asks every site for per-column sums,
   sums of squares and row counts of its training data, and combines them into
   one global mean/std. These are stored inside the model, so every site scales
   features identically and the saved model takes raw features.
2. **FedAvg rounds.** Each round the server sends the global model to all sites.
   Each site scores it on its local validation data, trains locally for a few
   epochs, and sends back its weights. The server averages them, weighted by each
   site's number of training sessions.
3. **Best model and early stopping.** The server tracks the weighted validation
   MSE (`val_mse`) and keeps the best global model; it stops after `--patience`
   rounds without improvement.

Rules the data split follows: all sessions of a participant stay at one site,
and the test set is the same 20% of participants as `training/train_fusion_lmmnn.py`.

## Install

```bash
pip install -r requirements.txt
```

`nvflare==2.9.0` is pinned on purpose: `flare/job.py` extends NVFlare's FedAvg
recipe, whose internals can change between versions.

**Windows:** NVFlare's start scripts are bash scripts. Use WSL2 (Ubuntu) for POC
mode and real deployments. The simulator also runs natively.

All commands below run **from the repo root**.

## 1. Prepare the site data

```bash
python federated/flare/prepare_site_data.py --n-sites 4
```

Creates `federated/flare/data/site-1 ... site-4` (each with `train.csv`,
`val.csv`, `columns.json`) and `federated/flare/data/test/`.
Use `--split age` to give sites different age groups (a non-IID setting).

## 2. Run in the simulator (development)

All sites run as threads in one process:

```bash
python federated/flare/job.py --mode sim --n-sites 4 --rounds 100
```

The global models are saved in
`/tmp/nvflare/cogniwarriors/cogniwarriors_fedavg/server/simulate_job/app_server/`:
`FL_global_model.pt` (last round) and `best_FL_global_model.pt` (best round).
Change the location with `--workspace`.

## 3. Evaluate the global model

```bash
python federated/flare/evaluate_global.py --model /tmp/nvflare/cogniwarriors/cogniwarriors_fedavg/server/simulate_job/app_server/best_FL_global_model.pt
```

## 4. Run in POC mode (real processes, one machine)

A real server and one process per site, communicating over the network like a
real deployment. NVFlare prepares and starts everything itself:

```bash
python federated/flare/job.py --mode poc --n-sites 4 --rounds 100
```

## 5. Real deployment (separate machines)

**a. Build the runtime image** (on every machine, or push it to a registry):

```bash
docker build -t cogniwarriors-flare:latest -f federated/flare/Dockerfile .
```

**b. Provision.** Edit `federated/flare/project.yml` (server hostname, site names), then:

```bash
nvflare provision -p federated/flare/project.yml -w provision_workspace
```

This creates one startup kit per participant under
`provision_workspace/cogniwarriors/prod_00/`. Send each kit **only** to its owner.
Kits contain private keys: never commit them.

**c. Put the data at each site.** Hospital N keeps its data in a folder that
contains a `site-N/` sub-folder, for example `/srv/cogni/site-1/` with
`train.csv`, `val.csv`, `columns.json`.

**d. Start the server** (hub machine):

```bash
cd server1 && ./startup/docker.sh -d      # or ./startup/start.sh without Docker
```

**e. Start each site** (hospital N):

```bash
export MY_DATA_DIR=/srv/cogni             # mounted read-only as /data in the container
cd site-1 && ./startup/docker.sh -d       # or ./startup/start.sh without Docker
```

Without Docker, point the site at its data with `export SITE_DATA_DIR=/srv/cogni/site-1`
before `start.sh`.

**f. Submit the job** (admin machine):

```bash
python federated/flare/job.py --mode prod --startup-kit provision_workspace/cogniwarriors/prod_00/admin@cogniwarriors.org --data-root /data --rounds 100
```

The command prints the job status and the folder with the downloaded results.
Alternatively, `--mode export --job-dir jobs/` writes the job folder to submit
with the NVFlare admin console.

## Options (`flare/job.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `sim` | `sim`, `poc`, `export` or `prod` |
| `--n-sites` | 4 | Sites that must join every round |
| `--rounds` | 50 | Maximum number of rounds |
| `--local-epochs` | 3 | Local epochs per round (keep low for small sites) |
| `--patience` | 20 | Stop after N rounds without validation improvement |
| `--loss` | `mse` | `mse` (fixed effects) or `lmmnn` (adds the random-effects loss; its 2 variance terms are averaged too) |
| `--mu` | 0 | FedProx strength (0 = plain FedAvg) |
| `--data-root` | `federated/flare/data` | Folder holding one sub-folder per site (`/data` in Docker) |

## Results

Same 39-session test set as the centralized script (`ef_composite`, R²):

| Setting | R² |
|---|---|
| NVFlare FedAvg, 4 sites (simulator) | 0.26 |
| Offline FedAvg simulation, 5 seeds | 0.265 ± 0.042 |
| Centralized (all data pooled), 5 seeds | 0.273 ± 0.055 |
| Single site alone, 5 seeds | 0.116 ± 0.032 |

Federation recovers nearly all of the pooled-data performance without moving data.
Reproduce the comparison with `python federated/fedavg_simulation.py`.

## Notes

- The validation MSE a site reports in a round is for the global model it
  *received*, so NVFlare's "best model" choice lags one round. This is standard
  NVFlare behavior and makes little difference here.
- The global-scaling step shares per-column sums and counts per site, which is
  aggregate information, not individual records.
