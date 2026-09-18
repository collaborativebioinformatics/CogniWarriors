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

### Data Sources
- **Imaging data**: Penn LEAD MRI derivatives (structural MRI scans)
- **Phenotype data**: CNB tasks, self-report, demographics (age, sex, education)

**Pipeline Components:**
1. **Image Feature Extractor**: ROI thickness, 68 DK regions (ACTIVE: bypass placeholder) → feeds into Model 1
   - Other imaging features (IN PROGRESS: vertex-wise / functional connectivity / raw volumes) → planned swap-in
2. **Phenotype Feature Extractor**: 97 features (selected via `preprocessing/analyze_phenotype_features.py`, down from an initial 161-column candidate matrix — see `doc/method.md` §3.2) — age, sex, group, dx flags, non-target CNB domains, a trimmed set of self-report scales → feeds into Model 2
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

### How to Use

#### 1. Start the Center
```bash
python center.py --config flare_config.yaml --port 8080
```

#### 2. Start Workers
```bash
python worker.py --center localhost:8080 --worker-id worker_01 --data-dir /data/center01
python worker.py --center localhost:8080 --worker-id worker_02 --data-dir /data/center02
```

#### 3. Monitor Training Progress
Check `progress_output.txt` for training metrics.

#### 4. Input MRI Files
Place structural MRI `.nii.gz` files in the specified directories and update `mri_input.txt` with paths.

---

## Checklist

### Implemented ✅
- [x] Center script (`center.py`) - distributes workers and coordinates FL process
- [x] Worker script (`worker.py`) - connects to center, performs distributed training
- [x] MRI input file (`mri_input.txt`) - structural MRI image paths
- [x] Progress output file (`progress_output.txt`) - training metrics visualization
- [x] README updated with setup confirmation
- [x] FLARE framework integration
- [x] Center-worker architecture for federated analysis

### Outstanding 📋
- [ ] FLARE configuration file (`flare_config.yaml`) - detailed FL setup
- [ ] Pre-trained model initialization and fine-tuning pipeline
- [ ] Structural MRI loading and preprocessing pipeline
- [ ] Phenotypical data integration with training progress
- [ ] Visualization dashboard for training progress
- [ ] Multi-center coordination and data governance protocols

### Archived Design History (from `doc/architecture_design_record.md`)
The project has evolved through several architecture designs, documented in the architecture design record with timestamps from 10:25 to 11:10. Key evolutions include:
- Transition from center-worker pattern to federated multi-site architecture
- Addition of PENN LEAD v1.0 as origin dataset with multimodal MRI (T1, rs-fMRI, DWI)
- Refinement of N-back score as primary output prediction
- Implementation of privacy-preserving data governance (raw data stays local)

---

## Project Directory Structure

```
 /Users/ahmet/Desktop/Longitudinal_imaging_to_multimodality/
 ├── .git/
 ├── LICENSE
 ├── README.md              # Updated with how-to guide and checklist
 ├── center.py              # FLARE center script
 ├── doc/
 │   ├── agents.md          # Agents administration guide
 │   ├── architecture_design_record.md  # Architecture design and decisions with full timestamp history
 │   ├── dataset_description.md
 │   ├── method.md
 │   ├── problem.md
 │   └── results.md
 ├── mri_input.txt          # Structural MRI input file
 ├── progress_output.txt    # Training progress visualization
 ├── workflow.png           # Project workflow diagram
 └── worker.py              # FLARE worker script
```

---

## Requirements

* Team 9: Integrating longitudinal imaging data (from different data sources) with phenotype and genotype analysis  
* Imaging Data  
* Time Data  
* Omics Data  

---

## Resources

- [https://github.com/IBM/comical/tree/main](https://github.com/IBM/comical/tree/main) (IBM, 2024\)  
- ADNI  
- [https://github.com/collaborativebioinformatics/Longitudinal\_imaging\_to\_multimodality](https://github.com/collaborativebioinformatics/Longitudinal_imaging_to_multimodality)  
- [NBBH\_attendance\_confirmation\_and\_group\_assignment](https://docs.google.com/spreadsheets/d/104H5TKJJpT7IsP2lMZRVPLlJf7pW9inCzA1hD_KDTWc/edit?gid=719203122#gid=719203122)  
- [https://data.dpuk.ukserp.ac.uk/cohortdirectory/Item?fingerPrintID=GENFI](https://data.dpuk.ukserp.ac.uk/cohortdirectory/Item?fingerPrintID=GENFI)  
- [https://atlaslongitudinaldatasets.ac.uk/datasets/ppmi-pd](https://atlaslongitudinaldatasets.ac.uk/datasets/ppmi-pd)
- [https://openneuro.org/datasets/ds007116/versions/1.0.6](https://openneuro.org/datasets/ds007116/versions/1.0.6)

**Data**: ~1.5 GB (223 T1w volumes + phenotype tables) — organized into 4 centers per `doc/architecture_design_record.md` entry 12:44.

![Workflow](workflow.png)
