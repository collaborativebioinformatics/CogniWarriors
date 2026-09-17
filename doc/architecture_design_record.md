# Architecture Design Record

## Record Entry: 2026-09-17 (Latest)

### Model Assumptions
- **FLARE Framework**: Using Nvidia FLARE for federated learning coordination
- **Center-Worker Pattern**: Central coordinator distributes tasks to multiple workers
- **Structural MRI Input**: Pre-processed NIfTI (.nii.gz) images as primary data modality
- **Phenotypical Output**: Training progress metrics and model performance indicators
- **Pre-trained Fine-tuning**: Starting with pre-trained model weights and fine-tuning on distributed data

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