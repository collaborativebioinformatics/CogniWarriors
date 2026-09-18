# Architecture Design Record

## Record Entry: 2026-09-18 09:29

### Dataset
- **Source**: https://openneuro.org/datasets/ds007089/versions/1.0.1
- **Files**: sourcedata>freesurfer>any>mri>*.mgz

### FLARE Configuration (flare_config.yaml)
- **Mixed-Effects Models**: Set up to accommodate mixed-effects models in training
- **Random Effects**: Local (worker) level
- **Fixed Effects**: Federated (center/hub) level

### Pre-trained Model Initialization and Fine-tuning Pipeline
- **Status**: REJECTED
- **Reason**: No pre-training anymore

### Structural MRI Loading and Preprocessing Pipeline
- **Status**: New embeddings processed but not fully integrated yet

### Phenotypical Data Integration with Training Progress
- **Status**: New embeddings processed but not fully integrated yet

### Visualization Dashboard for Training Progress
- **Status**: WIP (Work in Progress)
- **Approach**: Start with CLI/Pythonic approaches before anything more advanced

### Multi-center Coordination and Data Governance Protocols
- **Status**: Single-center, single-worker first before anything more complicated
- **Data Governance**: No real data governance protocols
- **Scaffold**: Leave some scaffold for data validation / schema checking

## Record Entry: 2026-09-17 17:52

- EF scores: here the original plan was to have the outcome be a composite score 'Executive Function'.
- But now, I thin you are leaning on splitting the outcome into 5: Nback, PLEY, AIM, CPT, TRAILMAKING
- Late-fusion is really cool to combine pre-trained models.

## Record Entry: 2026-09-17b (Worker Structure Design)

### Worker CLI/Client Architecture
- **Interface**: CLI-based (can later upgrade to UI), designed as a command-line tool for data scientists and site administrators
- **Installation**: Assumed pre-installed (NVFLARE, PyTorch, dependencies); placeholder for future install steps/wizard
- **Data Input**: User specifies local file path via CLI (NOT S3/cloud storage for privacy compliance)
  ```
  # Placeholder for future install wizard
  # Assume NVFLARE and dependencies are already installed
  ```
- **Data Preprocessing**: Load local data file, validate schema, preprocess to match model input requirements (feature extraction, normalization, etc.)
- **Network Test**: Verify connection to hub server before training begins
- **Training**: Execute `worker.run()` routines for local model training
- **Local Evaluation**: Log evaluation metrics every few epochs during training (in tandem with training logs)
- **Result Reporting**: Generate and save result report file upon successful completion

### Worker Workflow
1. **Initialization**: CLI parses arguments (hub address, data path, worker ID)
2. **Data Loading**: Read local data file, validate and preprocess to model schema
3. **Network Check**: Test connection to hub server
4. **Training Loop**: Execute local training via `worker.run()`
5. **Evaluation**: Log metrics every N epochs during training
6. **Reporting**: Save result report file with training metrics, model performance, and convergence status

### Notes
- All data stays local (no cloud/S3 upload) for privacy compliance
- Worker can run independently once connected to hub
- Future UI upgrade path preserved in design

## Record Entry: 2026-09-17b (Phenotype pipeline: EF composite + embedder)

### Model Assumptions
- **Target (y)**: continuous EF composite score (z-score average of 5 CNB
  tasks), not diagnostic category -- diagnosis doesn't harmonize across
  differently-instrumented biobanks, a continuous trait does.
- **Session-level modeling**: each of the 225 Penn LEAD imaging sessions is
  one training row (own EF composite from that session's CNB scores). No
  per-patient aggregation. Splits (train/val/test, later the institute
  simulation) are grouped by `participant_id` so a subject's sessions never
  straddle a split.
- **Phenotype embedder input** = everything the EF composite doesn't use:
  demographics/diagnosis flags (`participants.tsv`), non-EF CNB cognition,
  full self-report battery, sex-coalesced Tanner staging. `age` and
  `session_index` are kept as explicit head-level covariates, not routed
  through the embedder.
- **Structural-MRI QC (Euler number) is explicitly out of scope here** --
  deferred to the image-side track; `derivatives/freesurfer-post/` was not
  downloaded for this pipeline.

### Data Shapes
- **Phenotype embedder input**: 225 sessions x 161 features (one-hot
  demographics, non-EF CNB accuracy/RT, self-report totals/subscales,
  Tanner stage, `_was_missing` indicators). See
  `data/processed/phenotype_feature_manifest.json` for the exact column
  list, dropped columns, and imputation values used.
- **EF composite**: 225 sessions, `ef_composite` non-null for 217 (8
  sessions have fewer than `MIN_TASKS_REQUIRED=2` valid EF tasks).
- **Covariates**: `age` (per-session, from `sessions.tsv`), `session_index`
  (1/2/3 per subject).

### Accepted
- [x] 5-task EF composite: N-back (`cnb_lnb_mcr`), PCET (`cnb_pcet_cr`),
      AIM (`cnb_aim_aimtot`), CPT (`cnb_cptnl_total_sen`), Trail Making B
      (`cnb_trails_rtcr`, sign-flipped) -- pooled z-score, `valid_code` in
      {V, VC, F, 0} treated as usable (see Arguments below), averaged
      across whichever tasks are present, NaN if fewer than 2.
- [x] Trail Making A and Digit Symbol excluded from the composite
      (processing-speed baseline / borderline EF construct) but retained
      as phenotype-embedder input.
- [x] Missing values: median/mode imputation + `<col>_was_missing`
      indicator columns; columns >50% missing dropped first (none actually
      crossed that threshold in this dataset).
- [x] Small, regularized MLP embedder (Linear->LayerNorm->ReLU->Dropout,
      64->32) + linear head on [embedding, covariates] -- validated against
      a Ridge baseline via GroupKFold(5), MLP R^2=0.43+/-0.05 vs Ridge
      R^2=0.26+/-0.05 vs mean-predictor R^2~=0.

### Rejected
- [ ] Per-age-band z-scoring for the EF composite -- N~220 over an 8-16y
      range leaves ~25-30 per band, too unstable to estimate mean/SD.
- [ ] `valid_code == 'V'` as a strict QC filter -- initially implemented,
      but verified against the data that '0' ("no autovalidation rule
      available") is the *dominant* code for AIM/Trails/Digit Symbol
      (~98% of rows), not a failure code; a strict-V rule discarded nearly
      all data for those three tasks. Corrected to treat {V, VC, F, 0} as
      usable, {S, N, C, X, n/a} as not.
- [ ] Item-level self-report features -- only summary/subscale columns are
      used as embedder input, to keep dimensionality sane at N~225.
- [ ] `substance.tsv` as embedder input -- no summary score exists for it
      (sparse multi-select checkboxes per its own sidecar); would explode
      dimensionality for negligible signal at this N.

### Outstanding
- [x] Fuse with an image embedder (structural MRI, deferred/owned by a
      separate track) and the NVFLARE federated training loop.
- [ ] Revisit whether Euler number / other structural QC should re-enter
      as a covariate once the image side lands.
- [ ] EF composite vs. diagnosis correlation came out on the low end of
      the expected moderate range (|r| ~0.02-0.22 against
      dx_adhd/dx_psychosis/study_group, out-of-fold) -- not indicative of
      leakage (nowhere near 0.8), but worth a second look once more data
      or the fused model is available.

### Arguments/Reasons for Changes
- EF task selection and z-scoring/QC decisions are detailed above; see
  `repo/src/config.py` (`EF_TASKS`, `VALID_CODES_OK`,
  `NON_EF_COGNITION_TASKS`, `SELF_REPORT_SCALES`) for the exact,
  single-source-of-truth column lists both `build_ef_composite.py` and
  `build_phenotype_input.py` import, so the two scripts can't desync.
- Ridge-vs-MLP comparison is reported specifically because N~200-215 is
  small for deep learning; the MLP's win here is a real (if modest) signal,
  not a foregone conclusion, and worth re-checking as more data arrives.

## Record Entry: 2026-09-17 14:00

## Inputs

1) Phenotype MLP ──────> Embedding Vector
   └──> Different Phenotypes

2) Image Analysis feature Vector
   └──> Contains Aseg + Aparc stats

## Output

EF Score ──────> Nback, PCET, AIM, CPT, TRAILMAKING B

---

1) Prepare data
   ├──> Image Vector
   │     → Create FS stats
   │     → Make Vector
   │     → Check
   │         EF vs Image Vector
   │     → Finalize Vector
   └──> Phenotype Vector

2) Finalize Arch

3) Train Single Device

4) Wrap it in FL

5) Prepare Analysis

## Record Entry: 2026-09-17 12:44

- The dataset has been organized and split into 4 centers
 - i.e. workers, we do not mean you create 4 'center' scripts. In fact, MAYBe we should rename center to 'Hub' or 'Pool', or something similar to avoid confusion.
-  ~1.5 GB (223 T1w volumes + phenotype tables). 
- DATA_DESCRIPTION.md in the folder that covers everything like layout, columns, and the gotchas to watch for.

## Record Entry: 2026-09-17 11:49

- Review w/ Henrique & Ben
- Clinic -> Images
- Images -> Presence/Classification
- Presence/Classification -> Value for Biobanks
- Biobanks -> Clinic, Virtuous Cycle!
- Very powerful showcase, because 2027 might have a Clinical Hackathon in the summer!
- Assume: Geographic separation! So how do we build an app to connect different countries! That's what the hub and worker scripts should aim to achieve. Clinics, health centers etc. across the nordics, EU etc.
  - secondary: we adopt for discrepancies between the different participants.
- Features matter! Skip over the genomics focus on patient features, phenotypes.

## Record Entry: 2026-09-17 11:10

Architecture: Federated Learning for N-back Score Prediction (Multi-Site)

A conceptual/high-level diagram showing federated learning across two example sites, each with their own MRI-derived structural and phenotype data, aggregated into a global model that predicts N-back scores via regression.


Sites
Site A — Dataset 1 (e.g., ADNI)
sMRI (structural MRI scan)
→ Extract hippocampus and other structures (segmentation step)
→ produces two data streams:
Volumes (e.g., hippocampus, other structures)
Phenotype data (e.g., age, sex, education)
Site B — Dataset 2 (e.g., OASIS)


Mirrors Site A's pipeline:


sMRI
→ Extract hippocampus and other structures
→ produces:
Volumes (e.g., hippocampus, other structures)
Phenotype data (e.g., age, sex, education)
Federated Learning Flow
Each site performs local model training on its own data (Site A trains on Site A data only; Site B trains on Site B data only — data never leaves the site)
Both sites send model updates to a central Federated Learning Server, which aggregates model updates (depicted with a database/aggregation icon)
The server produces a Global Model — a neural network that predicts N-back score
The Global Model outputs the final N-back score (Regression)
Key Architectural Principle


Raw data (volumes, phenotype data) stays local to each site. Only model updates, not patient data, are shared with the central server — this is the core privacy-preserving mechanism of federated learning.

## Record Entry: 2026-09-17 11:06

Architecture: Federated Multimodal Model for Executive Function Prediction

This pipeline combines imaging and phenotype data through separate feature extractors and embedding models, fuses them via late fusion, and trains the joint model using federated learning (NVIDIA FLARE) across sites.


Data Sources
Imaging data — Penn LEAD MRI derivatives
Phenotype data — CNB tasks, self-report, demographics
Pipeline Components
1. Image Feature Extractor
ROI thickness, 68 DK regions (ACTIVE — currently a bypass placeholder) → feeds into Model 1
Other imaging features (IN PROGRESS — vertex-wise / functional connectivity / raw volumes) → planned swap-in to replace the ROI thickness placeholder
2. Phenotype Feature Extractor
X features (TBD) (IN PROGRESS) — age, sex, group, self-report, non-target CNB domains → feeds into Model 2
3. Embedding Models
Model 1: Image embedder — consumes imaging features
Model 2: Phenotype embedder — consumes phenotype features
4. Late Fusion
Concatenate embeddings — combines Model 1 + Model 2 outputs
Model 3: Prediction head — consumes the concatenated embedding to produce predictions
5. Targets (y — TBD)
EF composite (placeholder)
N-back score — 2-back minus 0-back (placeholder)
Federated Training Loop


All three models (Image embedder, Phenotype embedder, Prediction head) are trained jointly at each site:


Local training per site — all 3 models updated jointly using local data
FLARE server — aggregates via FedAvg
Global model — updated image embedder + phenotype embedder + prediction head
Global model is redistributed to sites → next round → repeat
Status Notes / Open Items
Imaging features: currently using ROI thickness (68 DK regions) as an active bypass; planned swap to vertex-wise, functional connectivity, or raw volume features once ready
Phenotype features: exact feature set still TBD (candidates: age, sex, group, self-report, non-target CNB domains)
Prediction targets: still TBD between EF composite and N-back score (2-back minus 0-back)

## Record Entry: 2026-09-17 10:33

### Additional Design Notes
- **1) 100-150 images t1 mri**: T1 MRI dataset size range for federated learning experiments
- **2) hippocampus segmentation**: Using Hippodeep PyTorch model - https://github.com/bthyreau/hippodeep_pytorch for automated hippocampal segmentation
- **3) cognitive test**: Cognitive assessment integration for N-back and Trail B test scores

## Record Entry: 2026-09-17 10:25

### Model Assumptions
- **FLARE Framework**: Using Nvidia FLARE for federated learning coordination
- **Hub-Worker Pattern**: Central coordinator distributes tasks to multiple workers
- **PENN LEAD Origin Dataset**: Primary data source with MRI and Cognitive components
- **Data Shape**: 3D volumes with dimensions (H, W, D) typically 180x240x180mm FOV, variable voxel resolution
- **T1 hippocampal volume + age regression**: T1 MRI analysis for hippocampal volume measurement and age-related regression modeling
- **N back score**: Working memory task performance score as primary output prediction

### Data Structure (PENN LEAD v1.0)
- **1.0 Origin Dataset**: Contains two main components:
  - **1.1 MRI Data**: Broken down as follows:
    - **1.1.1 T1 MRI**: Structural imaging input
    - **1.1.2 rs-fMRI**: Resting-state functional MRI
    - **1.1.3 n-back**: Task-based fMRI
    - **1.1.1.1 Sub-branches**: N-back prediction and Trail B prediction
  - **1.2 Cognitive Data**: Broken down as:
    - **1.2.1 N-back**: Working memory task performance
    - **1.2.2 Trail B**: Trail Making Test Part B performance

### Model Structure
- **2.1 Input Modalities**:
  - T1 MRI as input
  - rs-fMRI as input
  - DWI (Diffusion Weighted Imaging) as input
- **2.2 Machine Learning Model**: Federated learning pipeline with hub-worker coordination
- **2.3 Output**: N-back score prediction

### Accepted
- [x] Hub-worker architecture for federated analysis
- [x] PENN LEAD v1.0 as origin dataset
- [x] T1 MRI, rs-fMRI, DWI as input modalities
- [x] N-back score as output prediction
- [x] FLARE framework integration

### Rejected
- [ ] Centralized data storage (privacy-preserving constraint)
- [ ] Single-center processing (requires multi-center distribution)
- [ ] Raw DICOM files without preprocessing (NIfTI preferred)

### Outstanding
- [ ] FLARE configuration file (`flare_config.yaml`) detailed setup
- [ ] Pre-trained model initialization pipeline implementation
- [ ] Structural MRI loading and preprocessing pipeline
- [ ] Phenotypical data integration with training progress
- [ ] Visualization dashboard for training progress
- [ ] Multi-center coordination and data governance protocols

### Arguments/Reasons for Changes
- PENN LEAD v1.0 selected as origin dataset due to availability of multimodal MRI (T1, rs-fMRI, DWI) and cognitive scores (N-back, Trail B)
- Hub-worker pattern chosen over peer-to-peer for clearer coordination and easier debugging with multi-center data
- FLARE chosen over other FL frameworks due to Nvidia ecosystem compatibility and documentation availability
- N-back prediction as output aligns with primary clinical question of working memory assessment