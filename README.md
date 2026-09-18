# Longitudinal_imaging_to_multimodality
Name TBA, **Project Mind-blowing**

## Architecture Overview

The project follows a federated learning architecture for N-back score prediction across multiple sites, as documented in `doc/architecture_design_record.md`.

### Latest Architecture (11:10 Entry: 2026-09-17)
**Federated Learning for N-back Score Prediction (Multi-Site)**

This pipeline combines imaging and phenotype data across sites, aggregated into a global model that predicts N-back scores via regression.

**Sites:**
- **Site A** — Dataset 1 (e.g., ADNI): sMRI → hippocampus/structure segmentation → Volumes (hippocampus, other structures) + Phenotype data (age, sex, education)
- **Site B** — Dataset 2 (e.g., OASIS): Mirrors Site A's pipeline

**Federated Learning Flow:**
- Each site performs local model training on its own data (data never leaves the site)
- Both sites send model updates to a central FLARE server, which aggregates via FedAvg
- The server produces a Global Model — a neural network that predicts N-back score
- The Global Model outputs the final N-back score (Regression)

**Key Architectural Principle:**
Raw data (volumes, phenotype data) stays local to each site. Only model updates, not patient data, are shared with the central server — this is the core privacy-preserving mechanism of federated learning.

### Mixed-Effects Model Implementation

The system implements a **PyTorch mixed-effects model** for federated learning:

- **Fixed Effects**: Global parameters learned at hub level (shared across all workers)
- **Random Effects**: Local parameters learned at worker level (site-specific)
- **FedAvg Aggregation**: Averages random effects from all workers

**Model Architecture:**
```
MixedEffectsModel(
  (fixed_weights): Linear(in_features=10, out_features=1, bias=False)
  (random_weights): Linear(in_features=5, out_features=1, bias=False)
)
```

### Data Sources
- **Imaging data**: Penn LEAD MRI derivatives (structural MRI scans)
- **Phenotype data**: CNB tasks, self-report, demographics (age, sex, education)
- **Dataset**: https://openneuro.org/datasets/ds007089/versions/1.0.1

**Pipeline Components:**
1. **Image Feature Extractor**: ROI thickness, 68 DK regions (ACTIVE: bypass placeholder) → feeds into Model 1
   - Other imaging features (IN PROGRESS: vertex-wise / functional connectivity / raw volumes) → planned swap-in
2. **Phenotype Feature Extractor**: 97 features (selected via `src/analyze_phenotype_features.py`, down from an initial 161-column candidate matrix — see `doc/method.md` §3.2) — age, sex, group, dx flags, non-target CNB domains, a trimmed set of self-report scales → feeds into Model 2
3. **Embedding Models**: 
   - Model 1: Image embedder — consumes imaging features
   - Model 2: Phenotype embedder — consumes phenotype features
4. **Late Fusion**: Concatenate embeddings — combines Model 1 + Model 2 outputs
5. **Model 3: Prediction head**: Consumes the concatenated embedding to produce predictions
6. **Targets (y — TBD)**: EF composite (placeholder), N-back score — 2-back minus 0-back (placeholder)

**Status Notes / Open Items:**
- Imaging features: currently using ROI thickness (68 DK regions) as an active bypass; planned swap to vertex-wise, functional connectivity, or raw volume features once ready
- Phenotype features: finalized at 97 columns (pruned from 161 by univariate significance + redundancy/VIF analysis, see `doc/results.md`); sanity-check numbers there are re-run against this final set
- Prediction targets: still TBD between EF composite and N-back score (2-back minus 0-back)
- Pre-trained model initialization: REJECTED (no pre-training anymore)
- Visualization dashboard: WIP, starting with CLI/Pythonic approaches

### How to Use

#### Option 1: Docker (Recommended)
```bash
# Run with Docker Compose
./run_docker.sh

# Or manually
docker-compose up --build
```

#### Option 2: Local Testing
```bash
# Run local test
./run_test.sh
```

#### Option 3: Manual Setup

**Start the Hub:**
```bash
python3 center.py --port 8080 --n-workers 2 --n-rounds 3 --n-local-epochs 5
```

**Start Workers (in separate terminals):**
```bash
# Worker 1
python3 worker.py --hub-address localhost:8080 --worker-id worker_01 --data-dir ./data/center01 --n-local-epochs 5 --n-rounds 3

# Worker 2
python3 worker.py --hub-address localhost:8080 --worker-id worker_02 --data-dir ./data/center02 --n-local-epochs 5 --n-rounds 3
```

#### Monitor Training Progress
- Check `hub_metrics.json` for round-by-round metrics
- Check `worker_01_report.json` and `worker_02_report.json` for worker-specific reports
- View container logs: `docker-compose logs -f`

---

## Checklist

### Implemented ✅
- [x] Hub script (`center.py`) - distributes workers and coordinates FL process
- [x] Worker script (`worker.py`) - connects to hub, performs distributed training
- [x] Mixed-effects model with fixed and random effects
- [x] FedAvg aggregation for random effects
- [x] Round management for multi-round training
- [x] Weight distribution to workers
- [x] Docker support with docker-compose
- [x] Network endpoints for container networking
- [x] Training metrics logging
- [x] Model checkpoint saving
- [x] Worker report generation

### Outstanding 📋
- [ ] Integrate with real MRI/phenotype data
- [ ] FLARE configuration file (`flare_config.yaml`) - detailed FL setup
- [ ] Structural MRI loading and preprocessing pipeline
- [ ] Phenotypical data integration with training progress
- [ ] Visualization dashboard for training progress
- [ ] Multi-center coordination and data governance protocols
- [ ] Model validation and cross-validation
- [ ] Authentication and security for worker-hub communication

### Archived Design History (from `doc/architecture_design_record.md`)
The project has evolved through several architecture designs, documented in the architecture design record with timestamps from 10:25 to 11:10. Key evolutions include:
- Transition from center-worker pattern to federated multi-site architecture
- Addition of PENN LEAD v1.0 as origin dataset with multimodal MRI (T1, rs-fMRI, DWI)
- Refinement of N-back score as primary output prediction
- Implementation of privacy-preserving data governance (raw data stays local)
- Mixed-effects model with random effects at worker level, fixed effects at hub level

---

## Project Directory Structure

```
longitudinal_imaging_to_multimodality/
├── center.py                  # Hub script with Global Model, FedAvg, Round Management
├── worker.py                  # Worker script with local training
├── docker-compose.yml         # Docker orchestration
├── Dockerfile.hub             # Docker image for hub
├── Dockerfile.worker          # Docker image for workers
├── requirements.txt           # Python dependencies
├── run_docker.sh              # Docker run script
├── run_test.sh                # Local test script
├── doc/
│   ├── agents.md              # Agents administration guide
│   ├── architecture_design_record.md  # Architecture decisions
│   ├── dataset_description.md
│   ├── method.md
│   ├── problem.md
│   └── results.md
├── src/
│   ├── config.py              # Configuration and constants
│   ├── phenotype_model.py     # Phenotype embedder model
│   ├── train_phenotype_sanity.py  # Training script
│   └── ...
├── data/
│   ├── processed/             # Processed data
│   └── ...
├── hub_model.pt               # Saved global model (after training)
├── hub_metrics.json           # Training metrics (after training)
├── worker_01_report.json      # Worker 01 report (after training)
└── worker_02_report.json      # Worker 02 report (after training)
```

---

## Docker Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Network                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Hub Container                      │   │
│  │  - Global Model (Fixed Effects)                 │   │
│  │  - FedAvg Aggregator                            │   │
│  │  - Round Manager                                │   │
│  │  - Weight Distributor                           │   │
│  │  - Port: 8080                                   │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│         ┌───────────────┼───────────────┐               │
│         │               │               │               │
│  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐       │
│  │ Worker 01   │ │ Worker 02   │ │ Worker N    │       │
│  │ - Local     │ │ - Local     │ │ - Local     │       │
│  │   Training  │ │   Training  │ │   Training  │       │
│  │ - Random    │ │ - Random    │ │ - Random    │       │
│  │   Effects   │ │   Effects   │ │   Effects   │       │
│  └─────────────┘ └─────────────┘ └─────────────┘       │
└─────────────────────────────────────────────────────────┘
```

---

## Requirements

* Team 9: Integrating longitudinal imaging data (from different data sources) with phenotype and genotype analysis  
* Imaging Data  
* Time Data  
* Omics Data  

**Python Dependencies:**
- torch>=2.0.0
- numpy>=1.24.0

---

## Resources

- [https://github.com/IBM/comical/tree/main](https://github.com/IBM/comical/tree/main) (IBM, 2024\)  
- ADNI  
- [https://github.com/collaborativebioinformatics/Longitudinal\_imaging\_to\_multimodality](https://github.com/collaborativebioinformatics/Longitudinal_imaging_to_multimodality)  
- [NBBH\_attendance\_confirmation\_and\_group\_assignment](https://docs.google.com/spreadsheets/d/104H5TKJJpT7IsP2lMZRVPLlJf7pW9inCzA1hD_KDTWc/edit?gid=719203122#gid=719203122)  
- [https://data.dpuk.ukserp.ac.uk/cohortdirectory/Item?fingerPrintID=GENFI](https://data.dpuk.ukserp.ac.uk/cohortdirectory/Item?fingerPrintID=GENFI)  
- [https://atlaslongitudinaldatasets.ac.uk/datasets/ppmi-pd](https://atlaslongitudinaldatasets.ac.uk/datasets/ppmi-pd)
- [https://openneuro.org/datasets/ds007116/versions/1.0.6](https://openneuro.org/datasets/ds007116/versions/1.0.6)
- [https://openneuro.org/datasets/ds007089/versions/1.0.1](https://openneuro.org/datasets/ds007089/versions/1.0.1)

**Data**: ~1.5 GB (223 T1w volumes + phenotype tables) — organized into 4 centers per `doc/architecture_design_record.md` entry 12:44.

---

## Quick Reference

### Docker Commands
```bash
# Start all containers
docker-compose up --build

# Stop all containers
docker-compose down

# View logs
docker-compose logs -f

# Rebuild containers
docker-compose down && docker-compose up --build
```

### Local Commands
```bash
# Start hub
python3 center.py --port 8080 --n-workers 2 --n-rounds 3

# Start worker
python3 worker.py --hub-address localhost:8080 --worker-id worker_01 --data-dir ./data/center01

# Run automated test
./run_test.sh
```

### Configuration Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--port` | 8080 | Hub port |
| `--host` | 0.0.0.0 | Hub host (0.0.0.0 for Docker) |
| `--n-workers` | 2 | Number of workers |
| `--n-rounds` | 5 | Federated rounds |
| `--n-local-epochs` | 10 | Local epochs per worker |
| `--n-fixed-features` | 10 | Fixed effect features |
| `--n-random-features` | 5 | Random effect features |

![Workflow](workflow.png)