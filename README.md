# Longitudinal_imaging_to_multimodality
:checkmark: madpro's opencode is now set up and running

## How to Use

### 1. Start the Center
```bash
python center.py --config flare_config.yaml --port 8080
```

### 2. Start Workers
```bash
python worker.py --center localhost:8080 --worker-id worker_01 --data-dir /data/center01
python worker.py --center localhost:8080 --worker-id worker_02 --data-dir /data/center02
```

### 3. Monitor Training Progress
Check `progress_output.txt` for training metrics (loss, accuracy, learning rate per step).

### 4. Input MRI Files
Place structural MRI `.nii.gz` files in the specified directories and update `mri_input.txt` with paths.

---

## Checklist

### Implemented ✅
- [x] Center script (`center.py`) - distributes workers and coordinates FL process
- [x] Worker script (`worker.py`) - connects to center, performs distributed training
- [x] MRI input file (`mri_input.txt`) - structural MRI image paths
- [x] Progress output file (`progress_output.txt`) - training metrics visualization
- [x] README updated with setup confirmation

### Outstanding 📋
- [ ] FLARE configuration file (`flare_config.yaml`) - detailed FL setup
- [ ] Pre-trained model initialization and fine-tuning pipeline
- [ ] Structural MRI loading and preprocessing pipeline
- [ ] Phenotypical data integration with training progress
- [ ] Visualization dashboard for training progress
- [ ] Multi-center coordination and data governance protocols Establish a privacy-preserving infrastructure (such as DataSHIELD or federated learning protocols) to securely connect and query data across multiple hospital centers and disparate sources without centralizing raw sensitive patient data.  
2. **Exploratory Multi-Modal Data Profiling**: Perform exploratory data analysis across heterogeneous data types (imaging, omics, phenotypes, time-series) to map out structural variations, missingness patterns, and shape disparities between different clinical centers.  
3. **Flexible Schema & Generalized Feature Representation**: Design a robust, extensible schema and embedding strategy that accommodates evolving data shapes, varying feature dimensions, and asynchronous time points as new data and centers are incorporated.  
4. **Active-Learning & Uncertainty-Driven Sampling**: Implement an active learning loop where the model queries domain experts or prioritizes informative/uncertain samples (e.g., novel imaging phenotypes or discordant clinical outcomes) for annotation and refinement.  
5. **Generalized Multi-Modal Learning & Domain Adaptation**: Train robust multi-modal architectures equipped with domain adaptation or regularized ensemble techniques to generalize across shifting data distributions and center-specific biases.  
6. **Interpretability & Iterative Validation**: Integrate explainable AI techniques (such as uncertainty estimates and feature attribution) to evaluate model performance iteratively across centers, ensuring clinical trustworthiness and seamless deployment.

---

## Project Directory Structure

```
/Users/ahmet/Desktop/Longitudinal_imaging_to_multimodality/
├── .git/
├── LICENSE
├── README.md              # Updated with how-to guide and checklist
├── center.py              # FLARE center script
├── doc/
│   ├── dataset_description.md
│   ├── method.md
│   ├── problem.md
│   └── results.md
├── mri_input.txt          # Structural MRI input file
├── progress_output.txt    # Training progress visualization
├── workflow.png           # Project workflow diagram
└── worker.py              # FLARE worker script
```

Requirements:

* Team 9: Integrating longitudinal imaging data (from different data sources) with phenotype and genotype analysis  
* Imaging Data  
* Time Data  
* Omics Data

Resources:

- [https://github.com/IBM/comical/tree/main](https://github.com/IBM/comical/tree/main) (IBM, 2024\)  
- ADNI  
- [https://github.com/collaborativebioinformatics/Longitudinal\_imaging\_to\_multimodality](https://github.com/collaborativebioinformatics/Longitudinal_imaging_to_multimodality)  
- [NBBH\_attendance\_confirmation\_and\_group\_assignment](https://docs.google.com/spreadsheets/d/104H5TKJJpT7IsP2lMZRVPLlJf7pW9inCzA1hD_KDTWc/edit?gid=719203122#gid=719203122)  
- [https://data.dpuk.ukserp.ac.uk/cohortdirectory/Item?fingerPrintID=GENFI](https://data.dpuk.ukserp.ac.uk/cohortdirectory/Item?fingerPrintID=GENFI)  
- [https://atlaslongitudinaldatasets.ac.uk/datasets/ppmi-pd]

![Workflow](workflow.png)