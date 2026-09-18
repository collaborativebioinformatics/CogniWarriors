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

One row per session (225 rows). Built in two passes: first a broad
candidate matrix (161 embedder-input columns), then a feature-selection
pass that pruned it to the final **97** columns actually used. Both are
described below; `doc/results.md` has the full before/after numbers.

### 3.1 Candidate construction (initial 161-column matrix)

- `participants.tsv`: `study_group`, `sex`, `race`, `ethnicity` (one-hot
  encoded — see rationale below), 10 `dx_*` diagnostic flags (kept as
  plain 0/1 inputs; diagnosis is a phenotype *input*, not the target).
- `sessions.tsv`: per-session `age` (used instead of `participants.tsv`'s
  baseline-only age, which is invalid for later sessions) and a derived
  `session_index` (1/2/3) — both kept as explicit covariates concatenated
  in at the regression head, **not** fed through the embedder itself.
- 16 non-EF CNB cognition tasks (including Trail Making A and Digit
  Symbol, both EF-adjacent but excluded from the composite itself — see
  `EF_ADJACENT_EXCLUDED_FROM_COMPOSITE`): one accuracy + one RT column
  each, same QC gating as the EF composite.
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

### 3.2 Feature selection pass (161 → 97 columns)

161 columns for ≈217 usable sessions (132 unique subjects; GroupKFold
folds are only ≈40 samples) is wide relative to N — `phenotype_model.py`
already compensates on the model side (small embedder, dropout, weight
decay), but nothing had trimmed the input side to match, and the initial
matrix was built by a "not part of the EF composite → include it" rule
rather than a chosen feature set. `src/analyze_phenotype_features.py`
computes three things against the 161-column matrix to fix that:

1. **Univariate association** — Pearson + Spearman r of every column vs.
   `ef_composite` (217 sessions with a valid composite).
2. **Redundancy** — pairwise Pearson correlation across all columns
   (225 sessions), flagged at \|r\| ≥ 0.85, split into substantive pairs
   vs. `_was_missing`-indicator pairs (the latter mostly reflect whole
   CNB/self-report batteries being skipped together in a session, not a
   feature-value duplication).
3. **VIF** — variance inflation factor per substantive continuous/binary
   column (one-hot dummies and `_was_missing` indicators excluded, since
   both are collinear by construction and would just re-report #2 as a
   wall of `inf`), via closed-form OLS regression of each column on the
   rest (no `statsmodels` dependency).

**Selection rule applied**: keep a column unconditionally if it's
demographic/diagnostic/structural (`study_group`, `sex`, `race`,
`ethnicity`, `dx_*`, the two covariates) — these are cheap, theoretically
load-bearing, and not the source of the width problem. Otherwise keep it
only if it shows p<0.05 (uncorrected) univariate association with
`ef_composite`. Two exceptions applied regardless of p-value:
`difference_of_im_em_averages` was dropped because it's an exact linear
combination of `im_average`/`em_average` already in the set (VIF 25–41,
the worst in the table), and `cnb_trails_cr` was dropped because it's
constant (25.0) across all 225 sessions post-imputation — zero variance,
zero information.

**Result**: 8 self-report instruments dropped entirely (PANAS, STAI
pre/post, ALS-18, ALES, BDI, PPA, RSAS, E-SWAN DMDD — none had a
significant column, and this removed most of the worst missingness
offenders too), several more trimmed to their surviving subscale(s)
(BIS/BAS → `bas_rr` only; MAP-SR → social+recvoc; Wolf IM/EM →
im/em averages only; E-SWAN ADHD → inattention only; PRIME → total score
only; Tanner → mean stage only), and a handful of CNB RT/total columns
dropped where only the paired accuracy/RT column was significant (VOLT,
Matrix Reasoning, Line Orientation, effort/risk discounting). CNB task
columns survived almost intact (26/32) — the bloat was concentrated in
self-report, not cognition. Full before/after tables in
`doc/results.md`.

**One deliberate non-cut, flagged rather than silently trusted**: Digit
Symbol (both columns) and Trail Making A's RT survive the rule with the
*strongest* univariate signal in the whole table (\|r\| up to 0.67) — but
they're the same EF-adjacent tasks kept out of the composite for
resembling it too closely. Kept as phenotype input, but the embedder may
be partly re-deriving "processing speed ≈ EF" rather than adding
independent signal; worth stating explicitly if these numbers go in a
write-up.

**Not yet done**: the `_was_missing` indicators were not collapsed from
one-per-column to one-per-administered-block, so 106 near-duplicate
indicator pairs still remain post-pruning (down from 163, only because
there are fewer columns overall) — see Section 5.

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
- Collapsing the `_was_missing` indicators to one flag per administered
  battery/form instead of one per column (see Section 3.2).
- Deciding where `age`/`session_index` concatenate in the eventual fusion
  model — currently bolted onto the phenotype-only sanity-check head,
  but they're session-level, not phenotype-specific, so they may belong
  at Model 3's head instead once the image side exists.

## Reference: pipeline scripts

| Script | Purpose |
|---|---|
| `src/config.py` | Single source of truth for EF task list, QC codes, feature-column selections, thresholds |
| `src/download_openneuro.py` | Downloads `participants.tsv` + `sessions.tsv` from the public OpenNeuro S3 bucket |
| `src/build_ef_composite.py` | Computes `ef_composite` per session |
| `src/build_phenotype_input.py` | Builds the 225×97 phenotype feature matrix + manifest |
| `src/analyze_phenotype_features.py` | Per-feature target association, redundancy, and VIF — drives the Section 3.2 selection rule |
| `src/phenotype_model.py` | `PhenotypeEmbedder` + `PhenotypeRegressor` (PyTorch) |
| `src/train_phenotype_sanity.py` | GroupKFold sanity check: MLP vs. Ridge vs. mean-baseline |
