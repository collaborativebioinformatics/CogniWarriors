# Results: Phenotype Pipeline Sanity Check

See `doc/method.md` for the full methodology behind these numbers.

## EF composite coverage

- 225 sessions total; **217 (96.4%)** received a valid `ef_composite`
  (≥2 of 5 EF tasks passing QC).
- 8 sessions (3.6%) have fewer than 2 valid EF tasks and are excluded from
  training/evaluation entirely.
- Task-level coverage (`n_tasks_valid` out of 5, across all 225 sessions):

| n_tasks_valid | 0 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| sessions | 8 | 12 | 6 | 12 | 187 |

### The 8 sessions with no `ef_composite` (0/5 tasks valid)

All 8 have zero usable EF tasks — not partial data, complete absence.
Checked against the raw CNB files directly: 6 have **no row at all** in
the task files (the battery wasn't administered that session); 2 have a
row present but every task's `valid_code` is `n/a` (explicitly marked not
completed).

| participant_id | session_id | Reason |
|---|---|---|
| `sub-20333` | `ses-1` | row present, all `valid_code` = `n/a` |
| `sub-20812` | `ses-2` | no row in any CNB task file |
| `sub-21161` | `ses-1` | no row in any CNB task file |
| `sub-21713` | `ses-1` | row present, all `valid_code` = `n/a` |
| `sub-22510` | `ses-2` | no row in any CNB task file |
| `sub-22617` | `ses-2` | no row in any CNB task file |
| `sub-22618` | `ses-2` | no row in any CNB task file |
| `sub-23582` | `ses-1` | no row in any CNB task file |

These 8 sessions still have full rows in `phenotype_features.tsv` (session
still exists, phenotype/demographic data is generally available) — they
are excluded only from EF-composite training/evaluation, via the
`GroupKFold` join in `train_phenotype_sanity.py` which drops rows with a
`NaN` `ef_composite` before fitting.

## Phenotype feature missingness (limitation)

- 84 candidate input columns (before one-hot expansion) × 225 sessions =
  18,900 cells; **1,949 were missing and imputed (≈10.3% overall)**.
- 70 of the 84 candidate columns had at least one missing value; the other
  14 (mostly CNB task accuracy/RT columns, well-populated once QC-gated)
  had none.
- Missingness is concentrated in the **self-report battery**, not the
  cognitive tasks — self-report forms appear to have been skipped or left
  incomplete more often than computerized tasks were invalidated.
  Worst-affected columns:

| Column | % missing (of 225) |
|---|---|
| `panas_sum_pos` | 44.9% |
| `panas_sum_neg` | 44.4% |
| `rsas_total_score` | 36.4% |
| `ales_severity_mean` | 32.9% |
| `als_total_score` | 31.1% |
| `tanner_mean_stage` | 28.9% |
| `rpas_total_score` | 28.9% |
| `eswan_dmdd_home_total` | 28.9% |
| `eswan_dmdd_friends_total` | 28.4% |
| `eswan_dmdd_school_total` | 27.1% |
| `ppa_total_score` | 26.2% |
| `ales_frequency` | 24.4% |
| `asrm_total_score` | 21.8% |
| `clinically_significant_psychosis_spectrum_symptoms` (PRIME) | 21.8% |
| `post_scan_STAI_state_total_score` | 19.1% |

All missing values are imputed (median for continuous, mode for
categorical) with a paired `<col>_was_missing` indicator retained, rather
than dropping rows — this keeps all 225 sessions in the dataset, but means
**≈10% of the values the phenotype embedder sees are imputed rather than
observed**, concentrated in the self-report scales rather than the
EF/cognitive measures that drive the target. This is a genuine limitation
on phenotype input quality worth stating explicitly in the write-up.

## Phenotype embedder sanity check

GroupKFold, 5 folds, grouped by `participant_id` (a subject's sessions
never split across train/test):

| Model | MAE | R² |
|---|---|---|
| Mean predictor (baseline) | 0.488 ± 0.048 | −0.007 ± 0.006 |
| Ridge regression | 0.418 ± 0.036 | 0.263 ± 0.042 |
| **MLP embedder + head** | **0.350 ± 0.040** | **0.431 ± 0.046** |

The MLP embedder clearly outperforms both the naive mean-predictor and a
linear Ridge baseline on held-out subjects — evidence it is learning real,
generalizable signal from the phenotype features rather than memorizing,
which mattered to check given N≈200 is small for a neural network.

## Diagnosis correlation check (out-of-fold predictions)

| | r(actual `ef_composite`) | r(predicted `ef_composite`) |
|---|---|---|
| `dx_adhd` | −0.122 | −0.223 |
| `dx_psychosis` | −0.017 | 0.003 |
| `study_group == PRO/CHR` | −0.032 | −0.064 |
| `study_group == ADHD` | −0.044 | −0.127 |
| `study_group == TD/NC` | 0.076 | 0.187 |

All correlations are low-to-modest (\|r\| ≤ 0.22) — well short of the
>0.8 that would indicate the model is simply re-deriving diagnosis rather
than learning EF-specific signal (which would be a real concern, since
diagnosis flags are themselves model inputs). They came out lower than the
"moderate" 0.2–0.5 band expected going in; this is plausible (EF deficits
are transdiagnostic by design, and CNB-based composites are known to
correlate only loosely with categorical diagnosis in adolescent samples)
but is flagged as an **open item**, not a confirmed finding, pending more
data or the fused image+phenotype model.

## Status

The phenotype half of the pipeline (EF composite target + phenotype
embedder) is implemented and validated in isolation. **Missing**:
structural/functional image embedder, late fusion of image + phenotype
embeddings, and the NVFLARE federated training loop across simulated
institutes.
