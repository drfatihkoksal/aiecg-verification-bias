# Partial verification bias in AI-ECG accuracy for left ventricular systolic dysfunction

Analysis code, frozen analysis plans and aggregate results (MIMIC-IV).

**Question.** AI-ECG models for left ventricular systolic dysfunction (LVSD; here LVEF ≤45%) are
usually validated in patients who had an echocardiogram near the ECG. In the population such a model
would screen, the first ECG of every adult without known LVSD, only some patients are referred for
echocardiography, and referral rises with the model score. How far do sensitivity and specificity
measured in verified patients differ from those in the whole intended-use population? Two frozen public
models (EchoNext-Mini, DeepECG-SL) were applied to the first ECG of each adult in MIMIC-IV-ECG. Accuracy
was estimated in verified patients only and, correcting for partial verification, by inverse
probability weighting, with a missing-not-at-random sensitivity analysis.

## Contents

| Path | What it holds |
|---|---|
| `src/` | Cohort, scoring, verification model and estimators, exploratory analyses, aggregate tables, figures |
| `upstream/afef/src/` | EchoNext-Mini input pipeline from the author's earlier MIMIC-IV project (unchanged copies; see its README) |
| `concept_frozen_2026-11-01.md`, `concept_frozen_v2_2026-11-01.md`, `concept_frozen_deepecg_2026-11-01.md` | Protocol snapshots frozen with `cohort_spec_v1.yaml`, `cohort_spec.yaml` and `cohort_spec_addendum_deepecg.yaml` (their SHA-256 hashes are in those files) |
| `cohort_spec_v1.yaml`, `cohort_spec.yaml` | Design frozen before any index ECG was scored, with SHA-256 hashes of inputs, scripts and the protocol snapshot. Version 1 was superseded by version 2 before scoring. |
| `cohort_spec_addendum_deepecg.yaml` | Second-model analysis, frozen before any DeepECG-SL score was computed; deviations and later changes are logged in the file |
| `outputs/*.json`, `outputs/wp1_flow.csv`, `outputs/wp3_results.md` | Aggregate results written by the scripts |
| `outputs/tables/` | Aggregate tables (CSV): cohort characteristics, verification by characteristic, cross-tabulation, severity (from `src/19_aggregate_tables.py`); accuracy by design, setting, sex and sensitivity analysis (formatted from the result JSON files) |
| `outputs/figures/` | Figure 1 and the study flow diagram |

No patient-level data are included. Patient-level intermediates (`*.parquet`, scoring chunks, logs and
codebooks that list free-text chief complaints) are written to `outputs/` but are not part of the
repository. Section numbers (§) in docstrings of `src/` refer to the study protocol (the frozen snapshots above).

## Data access

The data are available to credentialed PhysioNet users under the PhysioNet data use agreement; they
cannot be shared here.

- MIMIC-IV 3.1 (https://doi.org/10.13026/kpb9-mt58), MIMIC-IV-ECG 1.0 (https://doi.org/10.13026/4nqg-sb35),
  MIMIC-IV-ED 2.2 (https://doi.org/10.13026/5ntk-km72) and MIMIC-IV-ECHO 1.0 (https://doi.org/10.13026/nrjh-5r77).
- EchoNext 1.1.1 (https://doi.org/10.13026/7cfw-d091), test split only, for the threshold and implementation check.

Data locations are set as constants at the top of each script (`R`, `HOSP`, `ECG`, `ED`, `ECHO`); edit
them to your local copies.

## Models and upstream code

Both models are used without training or fine-tuning; their weights are not redistributed here.

| Component | Source | Pinned version |
|---|---|---|
| EchoNext-Mini model code and weights | https://github.com/PierreElias/IntroECG | SHA-256 of the weight files in `cohort_spec.yaml` |
| EchoNext-Mini input pipeline: waveform conversion, tabular features, rhythm screen and the three patient-level input tables they write | `upstream/afef/src/` (path constant `AFEF`) | SHA-256 of the scripts and tables in `cohort_spec.yaml` |
| DeepECG-SL fine-tuned for LVEF (regression) | https://huggingface.co/heartwise/deepecg-sl_finetuned_LVEF_MHI | revision 1ee6698, `LVEF_MSE_SL.pt` |
| DeepECG-SL loader and ported preprocessing | https://github.com/drfatihkoksal/rise-spec (`analysis/14_deepecg_check.py`, `analysis/15_deepecg_score.py`), ported from https://github.com/HeartWise-AI/DeepECG_Docker | DeepECG_Docker commit c81d677; SHA-256 in `cohort_spec_addendum_deepecg.yaml` |

## Environment

Python 3.13 with the packages in `requirements.txt`. Scoring uses a CUDA GPU if one is available;
DeepECG-SL inference runs in full FP32 precision (TF32 off), because reduced-precision arithmetic made
predictions depend on batch size.

## Reproducing the results

Run `upstream/afef/src/` first (see its README), then the scripts below from the repository root, in this order.

| Step | Script | Writes (aggregate files in the repository) |
|---|---|---|
| Design inputs, score-blind | `src/00_decision_inputs.py` | `outputs/decision_inputs.json` |
| Operating threshold on the EchoNext test split (developers' rule: first ROC point with sensitivity ≥70%) | `src/01_source_threshold.py` | `outputs/source_threshold.json` |
| Intended-use cohort and verification within 30 days | `src/02_wp1_cohort.py` | `outputs/wp1_flow.csv`, `outputs/wp1_summary.json` |
| Verification-model covariates | `src/03_wp1b_covariates.py` | `outputs/wp1b_summary.json` |
| Freeze the design | `src/04_freeze_spec.py` | `cohort_spec.yaml` |
| Score every index ECG with EchoNext-Mini | `src/05_wp2_score.py` | patient-level only |
| Technical exclusions and transport gate | `src/06_wp2_gate.py` | `outputs/wp2_summary.json` |
| Primary analysis: verification model, naive and weighted estimators, MNAR range, bootstrap | `src/07_wp3_analysis.py` | `outputs/wp3_results.json` |
| Results table from the JSON | `src/08_wp3_tables.py` | `outputs/wp3_results.md` |
| Exploratory: conventional design, tipping points, natriuretic peptide and troponin, truncation | `src/09_wp3_exploratory.py` | `outputs/wp3_exploratory.json`, `outputs/fig1_data.json` |
| Exploratory: replication of a published MIMIC-IV validation design; atrial rate set to 0 | `src/11_e1b_e5.py` | `outputs/wp3_e1b_e5.json`, `outputs/fig1_e1b_roc.json` |
| Freeze the second-model analysis | `src/12a_freeze_deepecg.py` | `cohort_spec_addendum_deepecg.yaml` |
| Exploratory: second model, DeepECG-SL | `src/12_deepecg.py` | `outputs/wp3_deepecg.json` |
| Exploratory: accuracy by sex | `src/15_sex.py` | `outputs/wp3_sex.json` |
| Aggregate tables from patient-level data | `src/19_aggregate_tables.py` | `outputs/tables/table1.csv`, `s6_verification.csv`, `s7_crosstab.csv`, `s8_severity.csv`, `outputs/cohort_characteristics.json` |
| Figure 1 | `src/10_fig1.py` | `outputs/figures/fig1.png`, `.pdf` |
| Study flow diagram | `src/17_flow_diagram.py` | `outputs/figures/figS1_flow.png`, `.pdf` |

`src/wp3_design.py` holds the cohort, design matrix and estimators shared by the exploratory scripts;
`src/09_wp3_exploratory.py` checks that it reproduces the primary point estimates before running. The
bootstrap (1,000 patient resamples, models refitted in each) takes several hours on a multi-core machine.

## Prespecification

The analysis plan, the code and the checksums of all inputs were frozen (`cohort_spec.yaml`) before any
MIMIC-IV ECG was scored. Scripts and outputs marked EXPLORATORY were added after the primary results were
known; the primary analysis is unchanged.

## Licence and citation

Code: MIT licence (`LICENSE`). Please cite the software as described in `CITATION.cff`.
