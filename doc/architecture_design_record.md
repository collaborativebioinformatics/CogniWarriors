# Architecture Design Record

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
- [ ] Fuse with an image embedder (structural MRI, deferred/owned by a
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

## Record Entry: 2026-09-17 (Latest)

### Model Assumptions
- **FLARE Framework**: Using Nvidia FLARE for federated learning coordination
- **Center-Worker Pattern**: Central coordinator distributes tasks to multiple workers
- **Structural MRI Input**: Pre-processed NIfTI (.nii.gz) images as primary data modality
- **Data Shape**: 3D volumes with dimensions (H, W, D) typically 180x240x180mm FOV, variable voxel resolution
- **Phenotypical Output**: Training progress metrics and model performance indicators
- **Pre-trained Fine-tuning**: Starting with pre-trained model weights and fine-tuning on distributed data

### Data Shapes
- **Structural MRI**: 3D NIfTI volumes, typical shape (182, 218, 182) for MPRAGE, intensity range [0, 1] after normalization
- **Phenotypical Data**: Tabular format with columns [age, sex, diagnosis, center_id, followup_time], variable number of phenotypes per patient
- **Pre-trained Model**: weights shape (num_classes, channels, height, width) initialized on ImageNet, fine-tuned for MRI classification

### Accepted
- [x] Center-worker architecture for federated analysis
- [x] Structural MRI as input modality
- [x] Training progress visualization via output file
- [x] Pre-trained model initialization with fine-tuning
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
- Center-worker pattern selected over peer-to-peer for clearer coordination and easier debugging
- FLARE chosen over other FL frameworks due to Nvidia ecosystem compatibility and documentation availability
- Pre-trained fine-tuning approach selected to reduce data requirements and accelerate convergence
- NIfTI format chosen over DICOM for easier processing in deep learning pipelines