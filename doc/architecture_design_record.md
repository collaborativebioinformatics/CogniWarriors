# Architecture Design Record

## Record Entry: 2026-09-17 11:06 (Latest)

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

## Record Entry: 2026-09-17 10:33 (Latest)

### Additional Design Notes
- **1) 100-150 images t1 mri**: T1 MRI dataset size range for federated learning experiments
- **2) hippocampus segmentation**: Using Hippodeep PyTorch model - https://github.com/bthyreau/hippodeep_pytorch for automated hippocampal segmentation
- **3) cognitive test**: Cognitive assessment integration for N-back and Trail B test scores

## Record Entry: 2026-09-17 10:25 (Previous)

### Model Assumptions
- **FLARE Framework**: Using Nvidia FLARE for federated learning coordination
- **Center-Worker Pattern**: Central coordinator distributes tasks to multiple workers
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
- **2.2 Machine Learning Model**: Federated learning pipeline with center-worker coordination
- **2.3 Output**: N-back score prediction

### Accepted
- [x] Center-worker architecture for federated analysis
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
- Center-worker pattern chosen over peer-to-peer for clearer coordination and easier debugging with multi-center data
- FLARE chosen over other FL frameworks due to Nvidia ecosystem compatibility and documentation availability
- N-back prediction as output aligns with primary clinical question of working memory assessment