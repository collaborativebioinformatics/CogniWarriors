"""Shared paths and constants for the phenotype pipeline.

Single source of truth for which CNB task files feed the EF composite, so
build_ef_composite.py and build_phenotype_input.py can never desync: the
phenotype-input script excludes exactly the files listed here.
"""

import os
from pathlib import Path

# repo/src/config.py -> parents[0]=src, [1]=repo, [2]=Hackathon (sibling data/ dir)
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DATA_ROOT = Path(os.environ.get("NBBH_DATA_ROOT", DEFAULT_DATA_ROOT))

PHENOTYPE_DIR = DATA_ROOT / "phenotype"
PARTICIPANTS_TSV = DATA_ROOT / "participants.tsv"
SESSIONS_DIR = DATA_ROOT / "sessions"
PROCESSED_DIR = DATA_ROOT / "processed"

EF_COMPOSITE_TSV = PROCESSED_DIR / "ef_composite.tsv"
PHENOTYPE_FEATURES_TSV = PROCESSED_DIR / "phenotype_features.tsv"
PHENOTYPE_FEATURE_MANIFEST = PROCESSED_DIR / "phenotype_feature_manifest.json"

ID_COLS = ["participant_id", "session_id"]

# Files that are fMRI acquisition sidecars, not phenotype tables (no paired .tsv).
NON_PHENOTYPE_JSON = {"task-nback_bold.json", "task-rest_bold.json"}

# EF-battery task files: (file_stem, primary_column, valid_code_column, higher_is_better)
EF_TASKS = {
    "short_letter_2back": {
        "column": "cnb_lnb_mcr",
        "valid_code_column": "cnb_lnb_valid_code",
        "higher_is_better": True,
    },
    "penn_conditional_exclusion_task": {
        "column": "cnb_pcet_cr",
        "valid_code_column": "cnb_pcet_valid_code",
        "higher_is_better": True,
    },
    "penn_abstraction_inhibition_working_memory_task": {
        "column": "cnb_aim_aimtot",
        "valid_code_column": "cnb_aim_valid_code",
        "higher_is_better": True,
    },
    "short_penn_continuous_performance_test": {
        "column": "cnb_cptnl_total_sen",
        "valid_code_column": "cnb_cptnl_valid_code",
        "higher_is_better": True,
    },
    "trailmaking_test_b": {
        "column": "cnb_trails_rtcr",
        "valid_code_column": "cnb_trails_valid_code",
        "higher_is_better": False,  # lower RT = better
    },
}

MIN_TASKS_REQUIRED = 2

# CNB valid_code levels (documented identically in every task's JSON sidecar):
#   V  = Valid                                            -> usable
#   VC = Valid, with an assessor comment attached          -> usable
#   F  = Likely valid, mild/moderate effect on performance -> usable
#   0  = No autovalidation rule available (NOT a failure -- just means no
#        automated check ran; this is the dominant code for aim/trails/
#        digit_symbol, ~98% of their rows -- treating '0' as invalid would
#        discard nearly all data for those three tasks)                -> usable
#   S  = Skipped, N = Not Valid, C = bare assessor comment (no validity
#        determination), X = Excluded, n/a = missing entirely          -> unusable
VALID_CODES_OK = {"V", "VC", "F", "0"}

# Column-level missingness threshold: drop a candidate embedder-input column
# outright if more than this fraction of the 225 sessions are missing it,
# before imputing what remains.
MISSING_DROP_THRESHOLD = 0.5

# sessions.tsv columns to keep as covariates (allowlist, not blocklist --
# the file also has acq_time and 28 in-scanner n-back columns we don't want).
SESSIONS_ALLOWLIST_COLUMNS = ["age"]

# participants.tsv columns dropped outright (baseline-only age is invalid
# for later sessions -- sessions.tsv's per-session age is used instead).
PARTICIPANTS_DROP_COLUMNS = ["age", "age_months"]

CATEGORICAL_COLUMNS = ["study_group", "sex", "race", "ethnicity"]

# Non-EF cognition tasks: one accuracy + one RT column each (mirrors the
# "single primary column per construct" convention used for EF_TASKS),
# gated by the task's own valid_code column using the same VALID_CODES_OK
# rule as the EF composite.
NON_EF_COGNITION_TASKS = {
    "penn_face_memory_test": {"columns": ["cnb_cpf_cr", "cnb_cpf_rtcr"], "valid_code_column": "cnb_cpf_valid_code"},
    "penn_word_memory_test": {"columns": ["cnb_cpw_cr", "cnb_cpw_rtcr"], "valid_code_column": "cnb_cpw_valid_code"},
    "visual_object_learning_test": {"columns": ["cnb_volt_cr"], "valid_code_column": "cnb_volt_valid_code"},  # rtcr: p=0.198 vs ef_composite, cut
    "penn_emotion_recognition_task": {"columns": ["cnb_er40_cr", "cnb_er40_rtcr"], "valid_code_column": "cnb_er40_valid_code"},
    "measured_emotion_differentiation_test": {"columns": ["cnb_medf_cr", "cnb_medf_rtcr"], "valid_code_column": "cnb_medf_valid_code"},
    "age_differentiation_test": {"columns": ["cnb_adt_cr", "cnb_adt_rtcr"], "valid_code_column": "cnb_adt_valid_code"},
    "penn_matrix_reasoning_test": {"columns": ["cnb_pmat_cr"], "valid_code_column": "cnb_pmat_valid_code"},  # rtcr: p=0.054, cut
    "penn_verbal_reasoning_test": {"columns": ["cnb_pvrt_cr", "cnb_pvrt_rtcr"], "valid_code_column": "cnb_pvrt_valid_code"},
    "variable_short_penn_line_orientation_test": {"columns": ["cnb_plot_cr"], "valid_code_column": "cnb_plot_valid_code"},  # rtcr: p=0.051, cut
    "motor_praxis": {"columns": ["cnb_mpract_mp2", "cnb_mpract_mp2rtcr"], "valid_code_column": "cnb_mpract_valid_code"},
    "short_computerized_finger_tapping_task": {"columns": ["cnb_ctap_dom", "cnb_ctap_non"], "valid_code_column": "cnb_ctap_valid_code"},
    "delay_discounting_task": {"columns": ["cnb_ddisc_tot", "cnb_ddisc_rt"], "valid_code_column": "cnb_ddisc_valid_code"},
    "effort_discounting_task": {"columns": ["cnb_edisc_rt"], "valid_code_column": "cnb_edisc_valid_code"},  # tot: p=0.214, cut
    "risk_discounting_task": {"columns": ["cnb_rdisc_rt"], "valid_code_column": "cnb_rdisc_valid_code"},  # tot: p=0.575, cut
}

# Files excluded from phenotype input entirely (not just the EF composite):
# Trail Making A and Digit Symbol are the same processing-speed/set-shifting
# construct family as Trail Making B, which IS one of the 5 EF_TASKS. Their
# columns were the top univariate predictors of ef_composite (|r| up to 0.67,
# and |r|=0.63-0.70 against z_trailmaking_test_b specifically) -- close
# enough to the target's own construct to be tautological rather than
# independent phenotype signal, even though the raw values are not literally
# duplicated from the EF_TASKS files (see doc/results.md).
EF_ADJACENT_EXCLUDED_FROM_PHENOTYPE_INPUT = ["trailmaking_test_a", "digit_symbol"]

# Self-report scales: summary/subscale columns only (item-level responses
# excluded to keep dimensionality sane at N~225); redundant total-vs-average
# pairs collapsed to one representative column (e.g. als_total_score without
# als_avg_score, which is a constant multiple of it). No valid_code column
# exists for these (REDCap-style forms use a "_complete" flag instead, which
# the underlying summary-score columns already reflect via their own NaNs).
#
# Pruned to columns with p<0.05 (uncorrected) univariate association with
# ef_composite, via src/analyze_phenotype_features.py -- see
# doc/results.md for the full ranked table and rationale. Instruments cut
# entirely (no surviving column): PANAS, pre/post-scan STAI, ALS-18, ALES,
# BDI, PPA, RSAS, E-SWAN DMDD -- this also removed most of the worst
# missingness offenders (PANAS ~45%, RSAS 36%, ALES 33%, ALS 31%).
# difference_of_im_em_averages dropped separately: it's an exact linear
# combination of im_average/em_average (VIF 25-41), not new information.
SELF_REPORT_SCALES = {
    "bisbas_child": ["bas_rr"],
    "ari": ["ari_total_score"],
    "asrm": ["asrm_total_score"],
    "rpas": ["rpas_total_score"],
    "mapssr": ["mapssr_social_total", "mapssr_recvoc_total"],
    "wolf_im_em": ["im_average", "em_average"],
    "eswan_adhd": ["eswan_adhd_inattention_total"],
    "prime": ["prime_total_score"],
}

# substance.tsv has no summary score (sparse multi-select checkbox items,
# per its own JSON sidecar) -- deliberately excluded rather than exploded
# into dozens of near-empty binary features at N~225.
EXCLUDED_PHENOTYPE_FILES = ["substance"]

# Tanner staging: boy (6 items) and girl (8 items) forms are not item-
# parallel, so each subject's file (chosen by their `sex`) is summarized as
# a mean stage + completion flag rather than aligning items across forms.
TANNER_FILES_BY_SEX = {"M": "tanner_boy", "F": "tanner_girl"}
TANNER_ITEM_PREFIXES = {"tanner_boy": "tanner_boy_", "tanner_girl": "tanner_girl_"}
TANNER_COMPLETE_COLUMN = {
    "tanner_boy": "tanner_developmental_boys_complete",
    "tanner_girl": "tanner_developmental_girls_complete",
}

# Human-readable descriptions for every surviving raw/candidate column, for
# interpretability only -- not used by any pipeline logic. Sourced from each
# phenotype file's own BIDS-style JSON sidecar (`LongName`/`Description`
# fields in data/phenotype/*.json) or participants.json, not invented here.
# build_phenotype_input.py expands these into a full per-embedder-input-
# column description (one-hot levels, `_was_missing` indicators) written to
# the manifest as `column_descriptions`.
RAW_COLUMN_DESCRIPTIONS = {
    "age": "Age in fractional years at this session (from sessions.tsv, not participants.tsv's baseline-only age)",
    "session_index": "Session order for this participant (1st/2nd/3rd scan), derived by ranking session numbers in sessions.tsv",
    # CNB non-EF cognition tasks (Penn Computerized Neurobehavioral Battery)
    "cnb_cpf_cr": "Penn Face Memory Test (CPF) -- total correct responses",
    "cnb_cpf_rtcr": "Penn Face Memory Test (CPF) -- median response time for correct responses (ms)",
    "cnb_cpw_cr": "Penn Word Memory Test (CPW) -- total correct responses",
    "cnb_cpw_rtcr": "Penn Word Memory Test (CPW) -- median response time for correct responses (ms)",
    "cnb_volt_cr": "Visual Object Learning Test (VOLT) -- total correct responses",
    "cnb_er40_cr": "Penn Emotion Recognition Task, 40 faces (ER40) -- total correct responses",
    "cnb_er40_rtcr": "Penn Emotion Recognition Task (ER40) -- median response time for correct responses (ms)",
    "cnb_medf_cr": "Measured Emotion Differentiation Test (MEDF) -- total correct responses",
    "cnb_medf_rtcr": "Measured Emotion Differentiation Test (MEDF) -- median response time for correct responses (ms)",
    "cnb_adt_cr": "Age Differentiation Test (ADT) -- total correct responses",
    "cnb_adt_rtcr": "Age Differentiation Test (ADT) -- median response time for correct responses (ms)",
    "cnb_pmat_cr": "Penn Matrix Reasoning Test (PMAT) -- total correct responses",
    "cnb_pvrt_cr": "Penn Verbal Reasoning Test (PVRT) -- total correct responses",
    "cnb_pvrt_rtcr": "Penn Verbal Reasoning Test (PVRT) -- median response time for correct responses (ms)",
    "cnb_plot_cr": "Variable Short Penn Line Orientation Test (PLOT) -- total correct responses",
    "cnb_mpract_mp2": "Motor Praxis Test (MPRACT) -- total correct responses",
    "cnb_mpract_mp2rtcr": "Motor Praxis Test (MPRACT) -- median response time for correct responses (ms)",
    "cnb_ctap_dom": "Short Computerized Finger-Tapping Task (CTAP) -- mean taps, dominant hand",
    "cnb_ctap_non": "Short Computerized Finger-Tapping Task (CTAP) -- mean taps, non-dominant hand",
    "cnb_ddisc_tot": "Delay Discounting Task (DDISC) -- count of total delay chosen (impulsivity / preference-for-immediacy measure)",
    "cnb_ddisc_rt": "Delay Discounting Task (DDISC) -- median reaction time (ms)",
    "cnb_edisc_rt": "Effort Discounting Task (EDISC) -- median reaction time (ms)",
    "cnb_rdisc_rt": "Risk Discounting Task, child version (RDISC) -- median reaction time (ms)",
    # Self-report scales (summary/subscale totals; see doc/method.md 3.2 for
    # which scales were pruned out entirely)
    "bas_rr": "BIS/BAS Child -- Behavioral Activation System, Reward Responsiveness subscale (sum of items 8-12)",
    "ari_total_score": "Affective Reactivity Index (ARI) -- total irritability score (sum of items 1-6)",
    "asrm_total_score": "Altman Self-Rating Mania Scale (ASRM) -- total score (sum of 5 items)",
    "rpas_total_score": "Revised Physical Anhedonia Scale (RPAS) -- total score (sum of 15 true/false items, several reverse-scored)",
    "mapssr_social_total": "Motivation and Pleasure Scale, Self-Report (MAP-SR) -- Social pleasure subscale (sum of items 1-3)",
    "mapssr_recvoc_total": "Motivation and Pleasure Scale, Self-Report (MAP-SR) -- Recreational/Vocational pleasure subscale (sum of items 4-6)",
    "im_average": "Wolf Intrinsic/Extrinsic Motivation Scale -- Intrinsic Motivation average (mean of IM items)",
    "em_average": "Wolf Intrinsic/Extrinsic Motivation Scale -- Extrinsic Motivation average (mean of EM items)",
    "eswan_adhd_inattention_total": "E-SWAN ADHD Scale -- Inattention subscale (sum of items 1-9)",
    "prime_total_score": "PRIME psychosis-risk screen -- total score (sum of items 1-12, not adjusted for symptom duration)",
    # Derived / engineered
    "tanner_mean_stage": "Tanner pubertal staging -- mean stage across the sex-appropriate item set (boy: 6 items, girl: 8 items), sex-coalesced into one column",
    # Diagnosis flags (participants.tsv, clinical interview at timepoint 1)
    "dx_none": "1 if the participant had no clinical diagnosis at the timepoint-1 clinical interview; 0 if they had at least one",
    "dx_prodromal": "1 if the participant had prodromal psychosis / subthreshold psychosis spectrum disorder at the timepoint-1 clinical interview",
    "dx_prodromal_remit": "1 if the participant had prodromal psychosis / subthreshold psychosis spectrum disorder now in remission",
    "dx_psychosis": "1 if the participant had a threshold psychosis spectrum disorder diagnosis at the timepoint-1 clinical interview",
    "dx_moodnos": "1 if the participant had a non-specific (NOS) mood disorder diagnosis at the timepoint-1 clinical interview",
    "dx_mdd": "1 if the participant had a major depressive disorder diagnosis at the timepoint-1 clinical interview",
    "dx_bp": "1 if the participant had a bipolar disorder diagnosis at the timepoint-1 clinical interview",
    "dx_adhd": "1 if the participant had an ADHD diagnosis at the timepoint-1 clinical interview",
    "dx_ptsd": "1 if the participant had a PTSD diagnosis at the timepoint-1 clinical interview",
    "dx_ptsd_remit": "1 if the participant had a PTSD diagnosis now in remission",
}

# Categorical columns (one-hot encoded): base description + per-level text,
# both from participants.json. Used to expand e.g. `study_group_ADHD` into
# a full description rather than just echoing the column name.
CATEGORICAL_COLUMN_DESCRIPTIONS = {
    "study_group": "Experimental group the participant belonged to",
    "sex": "Sex of the participant as reported by the participant",
    "race": "Race of the participant as reported by the participant",
    "ethnicity": "Ethnicity of the participant as reported by the participant",
}
CATEGORICAL_LEVEL_DESCRIPTIONS = {
    "study_group": {
        "ADHD": "participants with ADHD",
        "TD/NC": "typically developing comparator participants",
        "PRO/CHR": "participants with psychosis or at clinical high risk for psychosis",
    },
    "sex": {"M": "male", "F": "female"},
    "race": {
        "White": "White",
        "Black or African American": "Black or African American",
        "Asian": "Asian",
        "Unknown or not reported": "Unknown or not reported",
        "More than one race": "More than one race",
    },
    "ethnicity": {"Hispanic": "Hispanic", "Non - Hispanic": "Non-Hispanic"},
}

# Structural-MRI QC (Euler number) is out of scope here -- that's the
# image-side teammate's territory. Only session-level covariates we own.
COVARIATE_COLUMNS = ["age", "session_index"]

# --- Fusion model (image + phenotype) + LMMNN random-effects loss ---

# Regression target(s), pulled from ef_composite.tsv. Single source of
# truth: swap to the 5 per-task z-score columns (all already present in
# that file) to go from 1 to 5 targets -- nothing else needs to change.
TARGET_COLUMNS = ["ef_composite"]
# To predict the 5 EF sub-scores instead, swap in (all already present in
# ef_composite.tsv, no upstream changes needed):
# TARGET_COLUMNS = [
#     "z_short_letter_2back",
#     "z_penn_conditional_exclusion_task",
#     "z_penn_abstraction_inhibition_working_memory_task",
#     "z_short_penn_continuous_performance_test",
#     "z_trailmaking_test_b",
# ]

IMAGE_DIR = DATA_ROOT / "images"
IMAGE_EMBEDDING_DIR = IMAGE_DIR / "embeddings"
# Both concatenated -- this "train/test" split is by session-count
# (single- vs multi-session subjects), not an ML split. Do not reuse it
# as a train/test split; grouped splitting is done fresh in the training
# script instead.
IMAGE_EMBEDDING_FILES = ["train_embedding.csv", "test_embedding.csv"]
IMAGE_EMBEDDING_DIM = 80

IMAGE_EMBEDDING_OUTPUT_DIM = 32  # mirrors PhenotypeEmbedder's embedding_dim
FUSION_HIDDEN_DIM = 64
FUSION_BATCH_SIZE = 32
FUSION_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"
FUSION_CHECKPOINT_PATH = FUSION_OUTPUT_DIR / "fusion_lmmnn.pt"
LR = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 300
PATIENCE = 20

# Chosen fusion architecture -- "current" from architecture_search.py, not
# "wide". "wide" won on val MSE but has ~2.8x the parameters (56k) on only
# 143 training rows, and its margin over "current" was smaller than the
# run-to-run noise from MPS non-determinism across seeds -- not a robust
# enough win to justify the extra capacity/overfitting risk. Single source
# of truth: both training and checkpoint metadata read this dict, so the
# architecture is guaranteed to match what a saved checkpoint expects.
FUSION_ARCHITECTURE = dict(
    image_embedding_dim=32,
    phenotype_embedding_dim=32,
    embedder_hidden_dim=64,
    head_hidden_dims=[64],
    head_dropout=0.3,
)
