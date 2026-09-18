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

## Phenotype feature missingness (limitation, pre-pruning matrix)

This section describes the **initial 161-column matrix**, before the
feature-selection pass below cut it to 97. Kept for the decision trail —
most of the worst-missingness columns listed here were among the ones
subsequently dropped.

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

## Phenotype feature selection (161 → 97 columns)

Run via `src/analyze_phenotype_features.py`; selection rule and rationale
in `doc/method.md` Section 3.2. Outputs: `data/processed/
feature_target_association.tsv`, `feature_redundancy_pairs.tsv`,
`missingness_redundancy_pairs.tsv`, `feature_vif.tsv`.

### Univariate association with `ef_composite` (217 sessions)

Top 15 by \|Pearson r\| — unchanged by pruning, since only non-significant
columns were cut:

| Column | Pearson r | p |
|---|---|---|
| `cnb_digsym_dscorrt` | −0.669 | 1.6e-29 |
| `cnb_trails_rtcr` | −0.663 | 8.4e-29 |
| `cnb_digsym_dscor` | 0.640 | 2.1e-26 |
| `cnb_pvrt_cr` | 0.556 | 5.3e-19 |
| `cnb_edisc_rt` | −0.546 | 3.2e-18 |
| `cnb_pvrt_rtcr` | −0.487 | 2.4e-14 |
| `cnb_pmat_cr` | 0.487 | 2.6e-14 |
| `cnb_plot_cr` | 0.448 | 4.2e-12 |
| `cnb_er40_cr` | 0.426 | 5.9e-11 |
| `cnb_mpract_mp2rtcr` | −0.421 | 9.5e-11 |
| `cnb_mpract_mp2` | 0.398 | 1.2e-9 |
| `cnb_adt_cr` | 0.389 | 3.0e-9 |
| `cnb_cpf_cr` | 0.382 | 6.1e-9 |
| `cnb_medf_cr` | 0.375 | 1.2e-8 |
| `cnb_cpw_rtcr` | −0.375 | 1.2e-8 |

No demographic, diagnostic, or self-report column reaches this list — the
strongest signal is entirely CNB cognitive-task columns, consistent with
EF deficits being transdiagnostic rather than group- or self-report-coded.

### What was cut

Self-report instruments dropped entirely (no column reached p<0.05):
PANAS, pre/post-scan STAI, ALS-18, ALES, BDI, PPA, RSAS, E-SWAN DMDD.
Of the 15 worst-missingness columns listed above, only 3 survive pruning
(`tanner_mean_stage` 28.9%, `rpas_total_score` 28.9%, `asrm_total_score`
21.8%) — the other 12, including the two worst offenders (PANAS at
~45%), are gone.

Trimmed to their significant subscale(s): BIS/BAS → `bas_rr` only;
MAP-SR → `mapssr_social_total` + `mapssr_recvoc_total`; Wolf IM/EM →
`im_average` + `em_average` (also dropping `difference_of_im_em_averages`
as an exact linear combination, and `validity_never_sleepy` as
non-significant); E-SWAN ADHD → inattention subscale only; PRIME →
total score only; Tanner → mean stage only (dropping `tanner_complete`).

CNB non-EF cognition columns dropped: `cnb_volt_rtcr`, `cnb_pmat_rtcr`,
`cnb_plot_rtcr` (RT non-significant where accuracy was),
`cnb_edisc_tot`, `cnb_rdisc_tot` (total non-significant where RT was),
and `cnb_trails_cr` (constant at 25.0 across all 225 sessions
post-imputation — zero variance). 26 of the original 32 CNB columns
survive — this half of the matrix was mostly earning its place already.

Demographics, diagnosis flags, and covariates are unchanged (kept
unconditionally, per the rule).

### Redundancy, before vs. after

| | Before (161 cols) | After (97 cols) |
|---|---|---|
| Substantive pairs \|r\|≥0.85 | 5 | 4 |
| `_was_missing`/`_was_missing` pairs \|r\|≥0.85 | 163 | 106 |
| Max VIF (substantive columns) | 40.8 (`im_average`) | 10.7 (`cnb_digsym_dscorrt`) |

The `im_average`/`em_average`/`difference_of_im_em_averages` construction
bug (VIF 41/30/25) is gone entirely — that was the single biggest driver
of the pre-pruning VIF numbers. What remains at the top of the VIF table
post-pruning is genuine shared-variance collinearity among CNB tasks
(digit symbol, effort/risk discounting, finger tapping — mostly in the
3–7 range), not a constructed duplicate.

Substantive redundant pairs remaining, all expected: `sex_F`/`sex_M` and
`ethnicity_Hispanic`/`ethnicity_Non-Hispanic` (one-hot complements, r=−1
by construction), `study_group_TD/NC`↔`dx_none` (r=0.90) and
`study_group_PRO/CHR`↔`dx_prodromal` (r=0.87) — both near-tautological
since group assignment and the diagnosis flag encode almost the same
clinical judgment. Left in deliberately (Tier A structural columns are
kept unconditionally), but worth knowing about if either pair ever shows
up as a top "important feature" downstream.

### Open items

- **Digit Symbol and Trail Making A's RT column are the strongest
  predictors in the table (\|r\| up to 0.67) but are the two tasks
  deliberately excluded from the EF composite for being too EF-adjacent**
  (`EF_ADJACENT_EXCLUDED_FROM_COMPOSITE`). They pass the statistical
  selection rule easily; kept as phenotype input, but flagged here rather
  than silently trusted — the embedder may be partly re-deriving
  "processing speed ≈ EF" through these two columns rather than adding
  independent phenotype signal.
- **`_was_missing` indicators are still one-per-column, not
  one-per-administered-block** — 106 pairs remain at \|r\|≥0.85 post-
  pruning (down from 163 only because there are fewer columns overall,
  not because the underlying co-administration pattern changed). Not yet
  addressed.
- **Imputation is computed once on the full 225-row table**, before
  `train_phenotype_sanity.py`'s `GroupKFold` split — a mild leakage of
  held-out-fold values into the medians/modes used to impute training
  rows. Likely small in effect at this N (medians are robust) but not
  yet quantified.

## Phenotype embedder sanity check

Re-run against the pruned 97-column matrix (previously 161 — see
"Phenotype feature selection" above). GroupKFold, 5 folds, grouped by
`participant_id` (a subject's sessions never split across train/test),
217 sessions / 130 unique subjects:

| Model | MAE | R² |
|---|---|---|
| Mean predictor (baseline) | 0.488 ± 0.048 | −0.007 ± 0.006 |
| Ridge regression | 0.378 ± 0.029 | 0.390 ± 0.052 |
| **MLP embedder + head** | **0.356 ± 0.019** | **0.439 ± 0.026** |

The MLP embedder still outperforms both baselines, but only barely beats
Ridge now (R² 0.439 vs. 0.390, was 0.431 vs. 0.263) — **Ridge is the
number that moved**. Pruning 64 mostly-noisy/redundant columns helped the
plain linear model far more than it helped the MLP, which could already
partially shrug off noise via dropout/weight decay. Read together with
the VIF drop (40.8→10.7) above, this is a second, independent signal that
the pre-pruning matrix's extra width was adding noise more than signal —
a linear model is more sensitive to exactly that kind of nuisance
dimensionality than a regularized network is.

## Diagnosis correlation check (out-of-fold predictions)

Re-run alongside the table above (same pruned matrix, same run):

| | r(actual `ef_composite`) | r(predicted `ef_composite`) |
|---|---|---|
| `dx_adhd` | −0.122 | −0.245 |
| `dx_psychosis` | −0.017 | 0.022 |
| `study_group == PRO/CHR` | −0.032 | −0.013 |
| `study_group == ADHD` | −0.044 | −0.170 |
| `study_group == TD/NC` | 0.076 | 0.170 |

All correlations are low-to-modest (\|r\| ≤ 0.245) — well short of the
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
