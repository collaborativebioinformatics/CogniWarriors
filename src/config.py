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

# Files excluded from the EF composite deliberately, but still phenotype-embedder input.
EF_ADJACENT_EXCLUDED_FROM_COMPOSITE = ["trailmaking_test_a", "digit_symbol"]

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
    # EF-adjacent tasks deliberately excluded from the EF composite (see
    # EF_ADJACENT_EXCLUDED_FROM_COMPOSITE) but still valid phenotype input.
    # cr dropped: constant (25.0) across all 225 sessions post-imputation --
    # zero variance, zero information. rtcr kept despite being the single
    # strongest univariate predictor in the whole table (|r|=0.66) -- flagged
    # in doc/results.md as a near-EF-tautology risk, not silently trusted.
    "trailmaking_test_a": {"columns": ["cnb_trails_rtcr"], "valid_code_column": "cnb_trails_valid_code"},
    "digit_symbol": {"columns": ["cnb_digsym_dscor", "cnb_digsym_dscorrt"], "valid_code_column": "cnb_digsym_valid_code"},
}

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

# Structural-MRI QC (Euler number) is out of scope here -- that's the
# image-side teammate's territory. Only session-level covariates we own.
COVARIATE_COLUMNS = ["age", "session_index"]
