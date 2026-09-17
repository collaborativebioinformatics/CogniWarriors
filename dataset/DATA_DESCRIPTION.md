# PennLEAD structural MRI — federated learning dataset

Adolescent T1-weighted brain MRI with per-scan phenotype, partitioned into four
centers for federated learning.

Every figure below was measured from the files in this folder.

---

## 1. Contents

```
223 T1w volumes · 132 subjects · 4 centers · 1.5 GB
```

| | |
| --- | --- |
| Modality | T1-weighted structural MRI only |
| Format | NIfTI, gzipped (`.nii.gz`) |
| Phenotype | one row per scan, tab-separated |
| Source | [OpenNeuro `ds007116` v1.0.6](https://openneuro.org/datasets/ds007116) |
| Licence | **CC0** (public domain dedication) |

---

## 2. Layout

```
.
├── DATA_DESCRIPTION.md
├── phenotype.tsv              all 223 scans, all centers
├── participants.tsv           132 subjects, as published upstream
├── sessions_master.tsv        full upstream session table (provenance)
└── centers/
    ├── center-01/
    │   ├── phenotype.tsv      this center's scans only
    │   └── anat/
    │       ├── sub-20302_ses-1_T1w.nii.gz
    │       └── ...
    ├── center-02/
    ├── center-03/
    └── center-04/
```

Each center folder is **self-contained**: its own images, its own phenotype
table, and image paths relative to that folder (`anat/...`). A worker process
given only `centers/center-02/` has everything it needs and can reach nothing
else. That is the privacy model expressed as a directory structure.

`phenotype.tsv` at the top level is the same rows with a `center_id` column,
for pooled baselines.

---

## 3. Phenotype

One row per scan.

| Column | Type | Description |
| --- | --- | --- |
| `participant_id` | `sub-XXXXX` | subject identifier |
| `session` | `ses-1` … `ses-3` | visit; **not always starting at 1** — see §6 |
| `age` | float, years | age at *this* scan, decimal |
| `sex` | `M` / `F` | |
| `diagnosis` | `PRO/CHR`, `TD/NC`, `ADHD` | study group |
| `center_id` | `center-01` … `center-04` | **assigned, not observed** — see §5 |
| `followup_time` | int, days | days since this subject's first scan; `0` at baseline |
| `n_sessions` | 1–3 | total scans for this subject |
| `dprime` | float | n-back task sensitivity; blank for 11 scans |
| `t1w` | path | relative to the center folder |

`PRO/CHR` = prodromal / clinical high risk for psychosis.
`TD/NC` = typically developing control.

### Cohort

```
132 subjects · 223 scans

diagnosis    PRO/CHR 69    TD/NC 36    ADHD 27
sex          M 67          F 65
age          8.7 – 20.1 years (mean 13.4), recorded per scan
sessions     1 scan: 59    2 scans: 53    3 scans: 20
             -> 73 of 132 subjects (55%) are longitudinal
follow-up    median 1.9 years, maximum 4.5 years
```

`dprime` is n-back working-memory sensitivity, recorded per session for 212 of
223 scans. It is the only continuous repeated-measures outcome here, and a
stronger modelling target than a static diagnosis label.

---

## 4. The images

```python
import nibabel as nib
volume = nib.load("centers/center-01/anat/sub-20302_ses-1_T1w.nii.gz").get_fdata()
```

| Property | Value |
| --- | --- |
| Shape | `(176, 256, 256)` — **identical for all 223 scans** |
| Voxel size | 1.0 × 1.0 × 1.0 mm isotropic |
| Field of view | 176 × 256 × 256 mm |
| Stored dtype | `int16`, intensity 0–4095 |
| Space | native acquisition space |
| File size | 4–12 MB each |

Uniform shape and isotropic 1 mm resolution means **no resampling is needed for
shape consistency**.

### These are raw acquisitions

Nothing has been skull-stripped, bias-corrected, registered or intensity-
normalised. Before training:

1. **Skull-strip** — non-brain tissue dominates the intensity histogram and
   varies with head position and defacing.
2. **Bias-field correct** — N4 or equivalent.
3. **Normalise intensity** — MRI has no absolute units, so 0–4095 is not
   comparable between scans. **In a federated setting this is not a local
   decision:** if each center standardises using its own statistics, the same
   brain is encoded differently depending on which center holds it, and the
   global model learns center identity instead of biology. Compute global
   moments with a federated statistics round first, then apply them everywhere.
4. **Crop or downsample** — `(176, 256, 256)` as float32 is 46 MB per volume in
   memory; a batch of 8 is 370 MB before activations.

The volumes are **defaced**: facial features have been removed or replaced for
deidentification. Harmless for brain measures, but it alters voxels near the
face, so avoid whole-head intensity statistics.

---

## 5. `center_id` is simulated

PennLEAD is a **single-site study**. There is no hospital, scanner or center
field in the source data. The four centers here are assigned — subjects shuffled
with a fixed seed and dealt round-robin.

Assignment is **by subject**, so a subject's repeat scans are always in the same
center. Splitting by scan would place the same brain in two centers and inflate
every federated result.

| Center | Scans | Subjects | Mean age | M/F | Longitudinal | Diagnoses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| center-01 | 62 | 33 | 13.3 | 18/15 | 24 | 19 PRO/CHR · 8 TD/NC · 6 ADHD |
| center-02 | 53 | 33 | 13.1 | 18/15 | 16 | 17 PRO/CHR · 6 TD/NC · 10 ADHD |
| center-03 | 60 | 33 | 14.2 | 18/15 | 20 | 18 PRO/CHR · 11 TD/NC · 4 ADHD |
| center-04 | 48 | 33 | 12.9 | 13/20 | 13 | 15 PRO/CHR · 11 TD/NC · 7 ADHD |

Subject counts are equal, but **scan counts differ by 30 %** (48 to 62) because
longitudinal subjects are not evenly distributed. FedAvg weights by sample count,
so center-01 carries more influence than center-04 — worth comparing against
equal weighting.

This partition correctly measures **what federation costs relative to pooling**,
because nothing but the partition differs between arms. It says nothing about
generalising across real scanners or sites. State that in any write-up.

---

## 6. Things that will mislead you silently

1. **Session numbering does not always start at 1.** Two subjects
   (`sub-20303`, `sub-20812`) begin at `ses-2`, because their first session has
   no T1w. Sort by session; never assume `ses-1` exists.
2. **223 scans, not 225.** The two sessions above are excluded — they exist
   upstream with T2w only. Both subjects still appear, via their later sessions.
3. **`dprime` is missing for 11 scans.** Handle nulls; do not impute silently.
4. **`diagnosis` is a study group, not an exclusive label.** Upstream,
   `dx_adhd` is positive for 68 subjects while `diagnosis == ADHD` for 27.
   Comorbidity is normal in a transdiagnostic cohort. Pick one definition and
   state it.
5. **Age is a confound, not a covariate.** Brain structure changes rapidly
   between 8 and 20, and the diagnostic groups differ in age. A model can score
   well by estimating age alone. Report an age+sex-only baseline beside any
   imaging result; if imaging does not beat it, that is the finding.
6. **`followup_time` comes from a shifted clock.** Upstream acquisition
   timestamps are date-shifted for deidentification, so absolute dates are
   meaningless — but intervals are preserved. Verified across all 93 follow-up
   scans against the per-scan age deltas: median disagreement **2 days**, maximum
   6, all within 30. `followup_time` is genuine elapsed time.
7. **Classes are imbalanced** (69 / 36 / 27 subjects). Report AUC or C-index,
   never accuracy.
8. **n = 132.** Small — roughly 33 subjects and 13–24 longitudinal subjects per
   center. Everything needs a confidence interval; nothing here supports a
   clinical claim.

---

## 7. What this dataset supports

**Supported**

- Federated training across 4 self-contained centers
- 3D structural MRI input, uniform shape, no resampling required
- Phenotype with age, sex, diagnosis, center and follow-up time
- Longitudinal modelling — 73 subjects with 2+ scans and real elapsed time
- A continuous repeated-measures outcome (`dprime`) beyond diagnosis

**Not supported**

- **Real multi-center variation** — centers are simulated; there is one scanner
- **Training on raw voxels as-is** — preprocessing (§4) comes first
- **2D ImageNet-pretrained backbones without adaptation** — this data is 3D
  single-channel; ImageNet weights are 2D three-channel
- **Strong statistical claims** — see §6.8

---

## 8. Provenance and citation

Derived from **PennLEAD** (Penn Longitudinal Executive functioning in Adolescent
Development), published as OpenNeuro `ds007116` v1.0.6 under **CC0**, funded by
NIH **R01MH113550**.

This folder is a filtered and repartitioned copy: T1w only, with T2w, fMRI,
diffusion, ASL, fieldmaps and multi-echo GRE removed, and `center_id` added.
Voxel data is unmodified from upstream.

Cite the original dataset in any publication — see its
[OpenNeuro record](https://openneuro.org/datasets/ds007116) for the full author
list and DOI.
