# Presentation Outline

## Introduction
- Goal: predict cognitive function 
- Use multimodal data (imaging + genomics + phenotyping) on large longitudinal cohorts, like biobanks.
- Repeated observations per patient → dependency + other random effects must be modeled, not ignored.
- No biobank access yet → proof of concept on Penn LEAD (OpenNeuro `ds007116`): 132 adolescents, 225 imaging sessions, images + CNB cognitive tests.
- Question: can tests + images predict executive-function scores, accounting for longitudinality?

## Data
- **Target — 5 executive-function indicators** (Penn CNB tasks, z-scored and averaged into `ef_composite`):

  | Task | Construct |
  |---|---|
  | N-back | working memory |
  | PCET | abstraction / set-shifting |
  | AIM | abstraction / inhibition / working memory |
  | CPT | sustained attention / inhibition |
  | Trail Making B | cognitive flexibility (processing speed) |

  Valid for 217/225 sessions (≥2 of 5 tasks passing QC).

- **Phenotype inputs**: 91 features (demographics, dx flags, 14 non-EF CNB tasks, 8 self-report scales, Tanner staging) — the 5 EF task files above are excluded wholesale, plus 2 EF-adjacent tasks (Trail Making A, Digit Symbol) dropped for being near-duplicates of the target construct.
- **No-leakage check**: max |Pearson r| between any phenotype feature and `ef_composite` *or any of its 5 components* = **0.58** — well under the 0.85 redundancy threshold used elsewhere in the pipeline. → inputs are informative but not tautological with the target.
- **Images**: {TBD} "structural MRI → ROI cortical thickness, 68 Desikan-Killiany regions → 80-dim feature vector."

## Model
- Two separate encoders (image, phenotype) — simple, modular, easy to add other modalities (genomics, ...) later.
- Fusion kept simple for now (late fusion / concatenation), with room to extend (e.g. cooperative learning) without breaking modularity.
- Architecture diagram:

  ![Architecture](architecture.png)

  Covariates chosen are the ones carrying the longitudinal information (age, session index).
- Longitudinality handled via the loss $0.5 \cdot [(y - f(x))^\top V^{-1} (y - f(x)) + \log|V|]$, modeling the same-patient random effect (for now with smaller data this is still not a problem in terms of time, otherwise we could maybe move to approches like FEMA).
- Grouped splitting: a given patient's sessions never cross train/val/test.

## Federation
- {TBD}


## Aplication