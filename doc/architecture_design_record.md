# Architecture Design Record

## Record Entry: 2026-09-17 11:10 (Latest)

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

## Record Entry: 2026-09-17 11:06 (Previous)

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