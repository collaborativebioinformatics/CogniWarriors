# Method: Phenotype Pipeline (EF Composite + Phenotype Embedder)

## Status

**Implemented and validated**: data download, EF composite computation,
phenotype feature matrix construction, phenotype MLP embedder +
sanity-check regression.

**Not yet implemented** (separate track): structural/functional image
embedder, late fusion of image + phenotype embeddings, NVFLARE federated
training loop across simulated institutes.

## 1. Data

- Source: OpenNeuro `ds007116` (Penn LEAD) — 132 adolescents, 225 imaging
  sessions (59 subjects with 1 session, 53 with 2, 20 with 3), 3 diagnostic
  groups (TD/NC, ADHD, PRO/CHR).
- Downloaded: `participants.tsv` (demographics + diagnosis flags),
  `sub-*_sessions.tsv` (per-session age + in-scanner n-back performance),
  `phenotype/` (42 CNB cognitive-task + self-report measure files).
  Structural-MRI derivatives were **not** downloaded — out of scope for
  this half of the pipeline (see Section 5).
- Row spine for every downstream table: the 225 `(participant_id,
  session_id)` pairs found in `sessions.tsv`. **Each imaging session is one
  modeling unit, not each subject** — a subject with 3 sessions
  contributes 3 rows, each with its own EF composite and (eventually) its
  own structural features. Train/val/test and institute splits are grouped
  by `participant_id` so a subject's sessions never straddle a split.

## 2. EF composite score

Target variable: a continuous executive-function (EF) composite, one value
per session, built from 5 Penn Computerized Neurobehavioral Battery (CNB)
tasks:

| Task | Column | Construct |
|---|---|---|
| N-back | `cnb_lnb_mcr` | working memory |
| PCET | `cnb_pcet_cr` | abstraction / set-shifting |
| AIM | `cnb_aim_aimtot` | abstraction / inhibition / working memory |
| CPT | `cnb_cptnl_total_sen` | sustained attention / inhibition |
| Trail Making B | `cnb_trails_rtcr` (sign-flipped) | cognitive flexibility |

Excluded from the composite: Trail Making A (a processing-speed baseline,
not EF-specific) and Digit Symbol (borderline EF construct) — both are
still included as phenotype-embedder *input*.

**Z-scoring.** Each task's raw score is standardized to
`z = (x - mean(x)) / std(x)` across all sessions with a QC-valid value for
that task (pooled across the full sample, not per age-band — see
Limitations in `doc/results.md`). This puts every task on the same scale
("SDs from this sample's average") despite wildly different raw units
(e.g. PCET correct-count 0–48 vs. Trail Making RT in milliseconds).
Trail Making B's z is sign-flipped so higher-z always means
better performance, consistently across all 5 tasks. Per session,
`ef_composite` = the mean of whichever task z-scores are available for
that session (a session needs ≥2 of the 5 tasks to get a composite;
otherwise `NaN`).

**QC filtering.** Every CNB task file carries its own `valid_code` column.
A session's score for a task is used only if `valid_code` is one of
`{V, VC, F, 0}` (Valid / Valid-with-comment / Likely valid / No
automated-check-available) and dropped if it's `{S, N, C, X, n/a}`
(Skipped / Not valid / bare comment / Excluded / missing). Note: `'0'`
means *"no autovalidation rule was run"*, not *"invalid"* — it's the
dominant code (~98% of rows) for AIM/Trails-B/Digit Symbol, so treating it
as invalid would have discarded almost all data for those three tasks.

**Result**: 217/225 sessions (96.4%) received a composite; 187/225 had all
5 tasks available (full breakdown in `doc/results.md`).

## 3. Phenotype feature matrix

One row per session (225 rows), 161 embedder-input columns, built from:

- `participants.tsv`: `study_group`, `sex`, `race`, `ethnicity` (one-hot
  encoded — see rationale below), 10 `dx_*` diagnostic flags (kept as
  plain 0/1 inputs; diagnosis is a phenotype *input*, not the target).
- `sessions.tsv`: per-session `age` (used instead of `participants.tsv`'s
  baseline-only age, which is invalid for later sessions) and a derived
  `session_index` (1/2/3) — both kept as explicit covariates concatenated
  in at the regression head, **not** fed through the embedder itself.
- 14 non-EF CNB cognition tasks + Trail Making A + Digit Symbol: one
  accuracy + one RT column each, same QC gating as the EF composite.
- 17 self-report scales (PANAS, STAI pre/post, BIS/BAS, ALS-18, ALES, BDI,
  PPA, ARI, ASRM, RPAS, RSAS, MAP-SR, Wolf IM/EM, E-SWAN ADHD/DMDD, PRIME):
  summary/subscale totals only — item-level responses are excluded to
  keep dimensionality sane at N≈225.
- Tanner pubertal staging: the boy (6-item) and girl (8-item) forms are
  not item-parallel, so each is summarized as a mean stage + completion
  flag, then sex-coalesced into one `tanner_mean_stage`/`tanner_complete`
  pair (off-sex rows are already all-`NaN` in the source files, so this
  coalescing is exact, not approximate).
- Excluded entirely: `substance.tsv` — it has no summary score (sparse
  multi-select checkboxes per its own sidecar; would explode
  dimensionality for negligible signal at this N).

**Categorical encoding**: one-hot for `study_group`/`sex`/`race`/`ethnicity`.
These are nominal categories with no natural order — a single numeric
column (e.g. `study_group = 0/1/2`) would force the model to treat the
categories as evenly spaced on an invented scale. One-hot avoids that at
the cost of a few extra columns, which is cheap at this N.

### Missing-value handling

1. Any candidate column missing in more than 50% of sessions would be
   dropped outright *before* imputation. In practice, none crossed this
   threshold (see `doc/results.md` for exact per-column missingness).
2. Every remaining column is imputed — median for continuous, mode for
   categorical (imputed **before** one-hot encoding) — each paired with a
   `<col>_was_missing` binary indicator column, so the model can use the
   missingness pattern itself as signal rather than have it silently
   disappear. This preserves all 225 sessions rather than dropping
   incomplete rows.

Exact missingness numbers (which columns, how much) are reported as a
limitation in `doc/results.md`.

## 4. Phenotype embedder

`PhenotypeEmbedder` (`repo/src/phenotype_model.py`):
```
Linear(in_dim -> 64) -> LayerNorm(64) -> ReLU -> Dropout(0.4)
-> Linear(64 -> 32) -> ReLU -> Dropout(0.3)
```
producing a 32-dimensional embedding per session. Deliberately small and
heavily regularized given only ≈200–215 usable sessions — LayerNorm (not
BatchNorm) is used because grouped cross-validation folds are only ≈40
samples, too few for stable batch statistics.

`PhenotypeRegressor` (sanity-check only, **not** the final fusion model):
the 32-dim embedding is concatenated with the covariates `[age,
session_index]` and passed through one more linear layer to predict
`ef_composite`. This stands in for where an image embedding would also be
concatenated once the image side exists.

Both are exercised end-to-end by `repo/src/train_phenotype_sanity.py`,
which is currently the only thing validating that the phenotype embedder
produces useful embeddings — see `doc/results.md` for the numbers.

## 5. What's still missing

- Structural (or functional) image embedder — a separate track, not
  started as part of this pipeline.
- Late fusion of image + phenotype embeddings into one joint model.
- The NVFLARE federated training loop across simulated institutes.
- Re-introducing structural-MRI QC (Euler number) as a covariate once the
  image side lands (it was explicitly dropped from this pipeline's scope).

## Reference: pipeline scripts

| Script | Purpose |
|---|---|
| `src/config.py` | Single source of truth for EF task list, QC codes, feature-column selections, thresholds |
| `src/download_openneuro.py` | Downloads `participants.tsv` + `sessions.tsv` from the public OpenNeuro S3 bucket |
| `src/build_ef_composite.py` | Computes `ef_composite` per session |
| `src/build_phenotype_input.py` | Builds the 225×161 phenotype feature matrix + manifest |
| `src/phenotype_model.py` | `PhenotypeEmbedder` + `PhenotypeRegressor` (PyTorch) |
| `src/train_phenotype_sanity.py` | GroupKFold sanity check: MLP vs. Ridge vs. mean-baseline |
