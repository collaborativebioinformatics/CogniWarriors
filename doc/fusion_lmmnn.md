# Method: Fusion Model + LMMNN Random-Effects Loss

## Status

**Implemented and running end-to-end**: image embedder, phenotype embedder,
late fusion, LMMNN random-effects loss, subject-grouped train/val/test
split, model checkpointing + inference, an architecture search, and a
model-family comparison (MLP vs. classical regressors). This picks up
where `doc/method.md` left off ("not yet implemented: image embedder, late
fusion") -- that gap is now closed for the local (non-federated) model.
NVFLARE federation across simulated institutes is still a separate,
unstarted track.

## 0. Environment

Everything here needs `torch`, `pandas`, `numpy`, `scikit-learn`,
`matplotlib` (see `requirements.txt`). If `python3 <script>.py` fails with
`ModuleNotFoundError: No module named 'torch'`, your active Python env
doesn't have the stack installed -- either `pip install -r
repo/requirements.txt`, or use an existing env that already has it
(`conda run -n <env> python <script>.py`).

All scripts auto-detect the fastest available device (`device_utils.py`):
MPS on Apple Silicon, else CUDA, else CPU -- via `torch.backends.mps.is_available()`
/ `torch.cuda.is_available()`, not OS-name checks, so it degrades correctly
on hardware without a GPU too.

## 1. Data

Row spine: the 215 `(participant_id, session_id)` sessions that have a
valid `ef_composite`, phenotype features, **and** a precomputed image
embedding (inner join -- see `multimodal_data.load_multimodal_dataframe()`).
Three sources, joined on `config.ID_COLS`:

- `data/processed/phenotype_features.tsv` -- 91 phenotype columns (see
  `doc/method.md` for how these were built/selected) + 2 covariates
  (`age`, `session_index`).
- `data/processed/ef_composite.tsv` -- target column(s), `config.TARGET_COLUMNS`
  (default `["ef_composite"]`; the 5 per-task z-score columns are also
  already in this file if you want to switch to 5 targets -- see the
  commented-out list in `config.py`).
- `data/images/embeddings/{train,test}_embedding.csv` -- 80-dim precomputed
  structural-MRI feature vectors per session (both files concatenated; the
  "train/test" split in that folder is by session-count, not an ML split
  -- ignore it for modeling, real splitting happens fresh below).

Train/val/test: two chained `GroupShuffleSplit` calls, both grouped by
`participant_id` so a subject's sessions never straddle a split --
**143 train / 33 val / 39 test** sessions. Fixed `random_state`
(`SEED=0`/`1`), so every script in this pipeline sees the *exact same* rows
in each split -- results across different scripts/architectures are
directly comparable.

## 2. Model architecture

Three small MLPs (`fusion_model.py`, `phenotype_model.py`), all sharing the
same `Linear -> LayerNorm -> ReLU -> Dropout -> Linear -> ReLU -> Dropout`
embedder shape:

```
ImageEmbedder:      Linear(80 -> 64) -> LayerNorm -> ReLU -> Dropout(0.4)
                     -> Linear(64 -> 32) -> ReLU -> Dropout(0.3)
PhenotypeEmbedder:   Linear(91 -> 64) -> LayerNorm -> ReLU -> Dropout(0.4)
                     -> Linear(64 -> 32) -> ReLU -> Dropout(0.3)
FusionRegressor:     concat(z_img[32], z_pheno[32], covariates[2]) = 66
                     -> Linear(66 -> 64) -> ReLU -> Dropout(0.3)
                     -> Linear(64 -> n_targets)
```

**19,841 parameters** total (n_targets=1). This is deliberately the
smaller of two candidates found by `scripts/architecture_search.py` -- a "wide"
variant (64/64 embeddings, 128 hidden, 56k params) scored better on val
MSE but not by a robust margin once run-to-run MPS noise and seed variance
were accounted for, and 2.8x the parameters on 143 training rows is real
overfitting risk. The chosen architecture is centralized in
**`config.FUSION_ARCHITECTURE`** -- a single dict that both training and
checkpoint metadata read, so there's one source of truth for "what
architecture is this."

`ImageEmbedder` is a small trainable projection over the *precomputed*
80-dim structural-MRI vector, not a raw-image CNN -- there's no raw-image
pipeline in this repo, just an already-extracted per-session feature
vector, so this plays the role a CNN encoder would in the LMMNN framing
without re-deriving image features from scratch.

## 3. LMMNN loss (`lmmnn_loss.py`)

Replaces MSE with the negative log-likelihood of a Gaussian whose
covariance has a per-subject random-intercept term plus i.i.d. error:
`V = sigma2_subject * Z @ Z.T + sigma2_error * I`. Everything (network
weights + `log_sigma2_subject`/`log_sigma2_error`, one pair per target) is
optimized jointly by one Adam optimizer -- no EM, no alternating steps.

Because minibatches are subject-grouped (`multimodal_data.SubjectGroupedBatchSampler`
greedily packs each subject's rows into the same batch), each subject's
block of `V` is exactly a compound-symmetry matrix (`sigma2_error*I + sigma2_subject*J`),
which has a closed-form inverse/log-determinant (Sherman-Morrison) --
computed per group via `scatter_add`, no dense matrix inversion, no
per-group Python loop.

Default inference is the fixed-effect part only (`f_NN(x)`, no BLUP
subject-specific correction) -- see `checkpoint.predict()`.

## 4. Running the pipeline

```bash
cd repo/src
python train_fusion_lmmnn.py
```

This: loads + splits the data, trains the LMMNN fusion model (subject-grouped
minibatches) and an MSE-only baseline fusion model (for the residual-ICC
sanity check) and two single-modality ablation models (image-only,
phenotype-only, for the modality-comparison plot), then saves everything
to `repo/results/`:

| File | What it is |
|---|---|
| `loss_curve_lmmnn.png` | LMMNN model: train vs. val, both as per-row LMMNN NLL (same units -- not val MSE, which is a different quantity used only for early stopping) |
| `loss_curve_baseline.png` | MSE-only fusion baseline: train vs. val MSE |
| `loss_curve_modality_comparison.png` | image-only vs. phenotype-only vs. fusion, MSE units, solid=train / dotted=val |
| `fusion_lmmnn.pt` | the trained LMMNN fusion model checkpoint (see below) |

Printed acceptance checks: no NaN/Inf in the loss or variance scalars,
subject-grouped batching verified (no subject split across a batch), final
`sigma2_subject`/`sigma2_error` per target, within-subject residual ICC
(LMMNN vs. baseline -- note this is *expected* to go up, not down: LMMNN
offloads subject-level mean structure into `sigma2_subject` rather than
forcing the fixed effect to explain it, so `f_NN(x)`-only residuals show
*more* subject clustering than a plain-MSE model's, not less -- recovering
that structure at prediction time is the optional BLUP step, not yet
implemented), held-out test MSE vs. a mean-baseline (R² framing), and a
checkpoint round-trip verification (load the just-saved checkpoint, predict
on raw test rows, confirm it matches the in-memory model exactly).

To predict the 5 EF sub-scores instead of the single composite, uncomment
the 5-column list for `config.TARGET_COLUMNS` in `config.py` -- everything
else (model output size, loss, batching, plots) adapts automatically.

## 5. Checkpoint / inference API (`checkpoint.py`)

A checkpoint bundles the trained weights, the exact architecture kwargs,
the fitted `StandardScaler`s, and the training-time column order -- it's
self-describing and doesn't depend on `config.py` still matching what it
was trained with.

```python
from checkpoint import load_checkpoint, predict

model, checkpoint = load_checkpoint("../results/fusion_lmmnn.pt")

# image_features / phenotype_features / covariates: 2D arrays (n_rows, n_cols),
# RAW (unscaled) values, columns in the order checkpoint["image_cols"] /
# checkpoint["phenotype_cols"] / checkpoint["covariate_cols"] expect.
preds = predict(model, checkpoint, image_features, phenotype_features, covariates)
# -> (n_rows, n_targets) numpy array, already in the target's natural units
```

`predict_from_checkpoint_path(path, ...)` is a one-shot convenience version
for callers that don't want to hold a loaded model in memory.

## 6. Architecture & model-family search tools

- **`scripts/architecture_search.py`** -- trains several `FusionRegressor` width/depth
  variants (same data split, plain MSE, 3 seeds each) and ranks them by val
  MSE. Useful if you change the data (more sessions, more image features)
  and want to re-check whether a bigger network is now justified.
- **`scripts/model_family_search.py`** -- compares genuinely different model
  *families* on the flat `concat(image, phenotype, covariates)` feature
  vector: Ridge, linear/RBF SVR, Random Forest, gradient boosting, k-NN,
  plus the MLP fusion architectures as a reference point. **The classical
  models here are untuned defaults** (one hyperparameter setting each) --
  treat this as a rough sanity check, not a verdict; a linear SVR in
  particular can blow up badly (seen: val MSE 4x worse than a mean
  baseline) if its regularization isn't matched to the feature count vs.
  sample size, which says more about needing a hyperparameter search than
  about linear models being bad here.

Both run on the same auto-detected device as the main training script.

## 7. Known limitations (read before scaling this up)

- **N=215 sessions / 130 subjects is small.** Several results in this repo
  are plausibly small-N noise, not signal: SVR's blowup, the "wide"
  architecture's uncertain win, the counterintuitive ICC increase. None of
  this is a sign the *logic* is wrong -- the LMMNN math, subject-grouped
  splitting, and fixed/random-effect decomposition are all N-invariant --
  but don't over-read single-run numbers at this sample size.
- **Image signal is currently weak.** The image-only ablation barely beats
  a mean-baseline (R²≈0.12) vs. phenotype-only's much stronger signal
  (R²≈0.6+) -- plausible given precomputed structural volumes are a more
  indirect predictor of a behavioral EF composite than some of the
  phenotype inputs (which include other cognitive-task scores).
- **Full-batch training in three places** (`train_baseline_mse`,
  `train_single_modality_mse`, both search scripts) won't scale to a much
  larger dataset (e.g. biobank-scale) -- only the actual LMMNN training
  loop uses real minibatches today. Val/test evaluation is also one
  full-batch forward pass everywhere. `load_multimodal_dataframe()` loads
  everything into memory with a Python-level `ast.literal_eval` per image
  vector -- fine at N=215, not at biobank scale.
- **Single random effect (subject) only.** If this is ever pointed at a
  genuinely multi-site/federated dataset (the actual NVFLARE goal this
  project sits inside), site/scanner heterogeneity is a second natural
  random-effect candidate that isn't modeled here -- "accounts for
  within-subject correlation" and "accounts for cross-site heterogeneity"
  are different claims; only the first is implemented.
- **MPS (Apple GPU) training is not bit-reproducible** run-to-run even with
  a fixed seed (kernel scheduling / float summation order differs) --
  expect single-run numbers to move a bit; trust the multi-seed mean/std
  in the search scripts over any one run.

## Reference: pipeline scripts

| Script | Purpose |
|---|---|
| `config.py` | Paths, target columns, chosen architecture (`FUSION_ARCHITECTURE`), all hyperparameters |
| `device_utils.py` | `get_device()` -- MPS/CUDA/CPU auto-detection |
| `multimodal_data.py` | Joins phenotype + EF targets + image embeddings; `MultimodalDataset`; `SubjectGroupedBatchSampler` |
| `fusion_model.py` | `ImageEmbedder`, `FusionRegressor`, `SingleModalityRegressor`, `build_mlp_head` |
| `lmmnn_loss.py` | `LMMNNLoss` -- the random-effects loss |
| `checkpoint.py` | `save_checkpoint` / `load_checkpoint` / `predict` -- the I/O boundary |
| `train_fusion_lmmnn.py` | Main training entrypoint; produces the plots + checkpoint in `results/` |
| `scripts/architecture_search.py` | MLP width/depth search for the fusion model |
| `scripts/model_family_search.py` | MLP vs. classical regressors, same split |
