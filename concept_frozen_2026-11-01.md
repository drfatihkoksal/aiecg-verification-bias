# VERIFY-GAP — How much does partial verification bias distort AI-ECG accuracy in the intended-use population?

Status: pre-registration · v0.2 · frozen before WP2 scoring on 2026-11-01 (`cohort_spec.yaml`; snapshot `concept_frozen_2026-11-01.md`)
Scope rule: minimum design that answers one question. Anything not listed here goes to §9 and needs an explicit decision before it is added.

## 1. Thesis

AI-ECG models for left ventricular systolic dysfunction (LVSD) are developed and validated on ECG–echo pairs, that is, on patients whose echo was clinically indicated. They are then proposed for opportunistic screening of every ECG.

The verified population can answer the gatekeeper question: can echo be deferred in a patient for whom it is being considered? It cannot answer the screening question: does the model find disease in patients nobody would refer [1]?

Referral depends on ECG findings the model itself uses and on clinical suspicion. This is the textbook setting for partial verification bias, which inflates sensitivity and deflates specificity [2,3]. Its magnitude has never been quantified for AI-ECG.

MIMIC-IV-ECHO v1.0 provides structured measurements for 206,488 BIDMC echo studies (2008–2022) of MIMIC-IV patients [4]. It can therefore be determined whether, and when, each MIMIC-IV-ECG tracing was followed by an echo, and the bias becomes estimable.

## 2. Research question

The intended-use population is the first ECG of every adult without known LVSD. In this population, how far do verified-only ("naive") estimates of an AI-ECG model's sensitivity, specificity and PPV for LVEF ≤45% deviate from verification-bias-corrected estimates?

## 3. Gap

- Screening-population evidence for AI-ECG LVSD comes from a few cohorts in which everyone had an echo [5,6]. Most published and regulatory validations use verified-only clinical cohorts, which are sometimes described as unselected [7].
- Verification-bias correction is standard diagnostic-accuracy methodology [2,3,8]. No AI-ECG application was found (Consensus literature check, 31 Oct 2026).

## 4. Pre-specified hypotheses and readout

- **H1 (direction):** naive sensitivity > corrected sensitivity, and naive specificity < corrected specificity, because the probability of verification rises with the AI score.
- **H2 (implementation):** corrected PPV in the intended-use population is lower than naive PPV.
- **AUROC:** reported with no directional hypothesis, because spectrum effects can move it either way [5].

| Result (MAR estimate, 95% CI) | Reading |
|---|---|
| Δsens and Δspec both < 5 percentage points in absolute value | No material distortion detected |
| Δsens or Δspec ≥ 5 points, CI excluding 0 | Verified-only validation misstates intended-use accuracy |
| Conclusion flips within the MNAR range (§6.5) | Assumption-dependent; report as bounds only |

Δ = naive − corrected. 5 percentage points is the pre-specified materiality threshold.

## 5. Data

- **MIMIC-IV-ECG:** waveforms and `machine_measurements` (intervals, report text).
- **MIMIC-IV-ECHO v1.0:** `structured-measurement.csv.gz` (study time, `test_type`, LVEF) [4]. TTE only (`test_type = 'tte'`).
  - Local copy is v1.0 (published 10 Mar 2026). The SHA256 of `structured-measurement.csv.gz` matches the release's `SHA256SUMS.txt`. `echo-study-list.csv` has 7,243 DICOM studies, which is v1.0.1's 7,228 plus the 15 that v1.0.1 removed.
  - v1.0.1 (25 Aug 2026) only removed 15 DICOM studies and 9 DICOM files, with no schema changes. This study uses no DICOM. v1.0 files are no longer downloadable, so cite v1.0 and state that the structured measurements are unchanged in v1.0.1.
- **MIMIC-IV `hosp` v3.1 and MIMIC-IV-ED v2.2:** `patients`, `admissions`, `diagnoses_icd`, `edstays` and `triage`, for covariates and ECG setting.
- **MIMIC-IV-ECG v1.0:** 800,035 records. About 63% fall outside any hospital admission (ED or outpatient); measured in the afef project.
- Local root: `/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c/` (`mimiciv/3.1`, `mimic-iv-ecg/1.0`, `echo/echo`, `mimic-iv-ed-2.2`, `echonext`).
- All processing is local (PhysioNet DUA). Only aggregate outputs leave the machine.

## 6. Design

### 6.1 Index test

- **EchoNext Mini-Model** [9,10,12] (`echonext_multilabel_minimodel`, `PierreElias/IntroECG`), frozen. Output `lvef_lte_45` is index 0 of 12 sigmoid outputs. The model was developed on NewYork-Presbyterian/Columbia data, so there is no overlap with MIMIC.
  - Already downloaded, loaded (`strict=True`) and verified in the afef project. On the EchoNext test split it reproduced the published composite SHD AUROC of 0.820, with LVEF ≤45% AUROC 0.852. Weights SHA256: `aac57aad…21cd6`.
  - Input: 10-s 12-lead waveform `[N, 1, 2500, 12]` at 250 Hz, plus 7 tabular features (sex, age, ventricular rate, atrial rate, PR, QRS, QTc).
  - The MIMIC → EchoNext pipeline is reused unchanged from afef (`src/mimic_to_echonext.py`, `src/08_mimic_tabular.py`, frozen in `cohort_spec_v1`). It uses raw ADC `d_signal`, takes every second sample (500 → 250 Hz), re-derives the limb leads from I and II, and reproduces the source's tabular transform (max difference 3.9e-07).
  - Tabular arm: `atrial_rate` is not in MIMIC. It is reconstructed (= ventricular rate if P waves are detected, else 0). In AF, atrial rate and PR are 0 by the source's own rule, so rhythm feeds the score directly through the linear head. Relevant here because all non-paced rhythms are included.
  - Licence: non-commercial use; patent application 63/555,968 must be declared; weights are not redistributed.
  - Model scope [12]:
    - Training excluded ECGs with ventricular pacing, ECGs failing MUSE quality control, and patients with prosthetic valves.
    - Positive label: ECG within 1 year before a TTE showing the diagnosis. Negative label: ECG before a negative TTE, at any interval, provided the patient never had the diagnosis.
    - Training ECG settings: 48.2% inpatient, 31.5% ED, 17.2% outpatient, 3.2% procedural.
- Published LVEF ≤45% performance: AUROC 85.2 (83.9–86.5), AUPRC 59.9 (56.6–63.3) on the EchoNext-Mini test split ([12], Fig. 4). afef reproduced these locally (0.852 / 0.599).
- Operating threshold: EchoNext-Mini publishes no LVEF ≤45% threshold.
  - The paper's Statistical Analysis computes binary metrics "at the cutoff leading to 70% sensitivity", but reports them only for composite SHD: sensitivity 70.1, specificity 77.9, PPV 70.2, NPV 77.8 [12]. The cutoff value itself is not reported.
  - The repo notebook `EchoNext_Mini_example.ipynb` implements this as the first ROC cutoff with sensitivity ≥0.70 on the test split.
  - Checked: full text (PDF in this folder), PubMed abstract, GitHub README and notebook, PhysioNet page. The Supplementary Appendix was not checked; the main text refers to it only for missingness and label summaries.
- **Decision (31 Oct 2026): source-anchored threshold.** The authors' rule is applied to `lvef_lte_45` on the EchoNext test split. It is external to MIMIC, so it is untouched by MIMIC verification.
  - Script: `src/01_source_threshold.py`; output: `outputs/source_threshold.json`. No MIMIC data are touched.
  - Rule check: the same code on composite SHD reproduces the published sens/spec/PPV/NPV and their 95% CIs exactly: 70.1 (68.3–71.9) / 77.9 (76.5–79.4) / 70.2 (68.3–72.0) / 77.8 (76.4–79.3).
  - **Frozen threshold: positive if sigmoid output ≥ 0.5675751566886902** (logit 0.2720; 56.76 on the 0–100 scale).
  - At this threshold on the EchoNext test split (n = 5,442; 962 positives, 17.7%): sensitivity 70.1 (67.0–73.0), specificity 83.6 (82.5–84.7), PPV 47.8 (45.3–50.5), NPV 92.9 (92.1–93.7), with 25.9% of ECGs flagged. The same cutoff results with `drop_intermediate=False`.
  - The threshold is fixed before any MIMIC score is computed and held constant across all estimators and bootstrap resamples.

### 6.2 Intended-use cohort

- Unit: first ECG per adult patient (≥18 y) in MIMIC-IV-ECG. All non-paced rhythms are included, because the model would run on every ECG within its scope.
- Exclude known LVSD or heart failure before the index ECG:
  - any resting TTE before the index ECG with D = 1 under the §6.3 rule; or
  - any I50.x/428.x or I42.x/425.x code on an admission with `dischtime` < index ECG time.
  - Codes from the admission that contains the index ECG are **not** used. They are assigned at discharge and include diagnoses made after the ECG, such as new HF found on the verifying echo, so using them would remove incident cases.
- Exclude technically unusable tracings (flat or missing leads).
- **Exclude paced ECGs (decision 31 Oct 2026).** EchoNext-Mini was not trained on ventricular-paced ECGs [12].
  - Definition: any pacing statement in the machine report: ventricular, atrial or unspecified. The regex is afef `src/02_rhythm_screen.py` (`rhythm_rules_v2`: `f_vpaced | f_apaced | f_paced_generic`).
  - Applied to the patient's first ECG: if it is paced, the patient is excluded, not moved to a later ECG. A paced first ECG means a known cardiac device, which is not the screening population. Counts are reported by paced subtype.
- **Prosthetic valves are not excluded (decision 1 Nov 2026).**
  - EchoNext excluded prosthetic-valve TTEs from training [12] without stating why. The exclusion fits valve-lesion labels, but the LVEF label is valid with a prosthesis, and unlike pacing a prosthesis does not alter the ECG input.
  - Pre-index ascertainment is incomplete: 468 patients have pre-index evidence, while 552 show a prosthesis only on a TTE after the index ECG (`outputs/decision_inputs.json`).
  - Exclusion based on the verifying TTE is not allowed, because that information exists only when V = 1.
  - Pre-index prosthesis evidence enters the verification model instead (§6.4).

### 6.3 Verification and reference standard

- V = 1 if a resting TTE with LVEF information from any source in the D rule below occurs 0–30 days after the index ECG. Use the first such TTE. The window was confirmed on 1 Nov 2026.
  - Rationale: a 14-day window would lose 9.1% of 30-day verifications overall and 14.9% in outpatients, the setting closest to screening. A TTE without any LVEF information changes V30 by only 0.17 points (16.70% vs 16.53%).
- D = LVEF ≤45% on that TTE. D is defined only when V = 1.
- LVEF fields (verified in afef): `lvef` is the lower bound and `lvef_upper` the upper bound of a reported range. `biplane_lvef` and `lvef_3d` are separate measurements that are mostly present when the reported range is absent.
- Coding relative to 45 among 179,928 resting TTEs, from `src/00_decision_inputs.py` (`outputs/decision_inputs.json`):

  | Source | Studies |
  |---|---|
  | Reported range | 133,399 |
  | Biplane | 27,975 |
  | 3D | 11,541 |
  | Qualitative text only | 1,743 |
  | Nothing | 5,270 |

  Classified as definitely ≤45: 29,853; definitely >45: 141,581; straddling 45: 3,224; no information: 5,270. Of the 3,224 straddling studies, 2,704 are "45–50".
  - Correction: an earlier version of this section listed counts from afef's feasibility report (51,360 single values … 58,694 with no EF). Those counts cover all 206,488 studies, including stress echo and TEE, and only the `lvef` field. They do not describe the TTEs.
- **D rule (decision 1 Nov 2026).** Sources are taken in priority order; numeric values are used only if plausible (5–95).
  1. `biplane_lvef`: D = 1 if ≤45.
  2. Reported range [`lvef`, `lvef_upper`]:
     - Whole range ≤45 → D = 1. Lower bound >45 → D = 0.
     - Closed range straddling 45 → **midpoint rule**: D = 1 if (low + high)/2 ≤45. So 45–50 → 0; 40–50 → 1; 45–55 → 0.
     - Open-ended range (`lvef_upper` = 100 or missing) with lower bound ≤45 → D = 1 (lower-bound rule).
  3. `lvef_3d`: D = 1 if ≤45.
  4. Qualitative `lv_function_text`:
     - "Nl", "Hyper", "Borderline (EF50–55%)" → 0
     - "Sev", "Mod (EF 30–39%)", "Mod-sev", "Mild-mod (EF 35–45%)" → 1
     - "Mild (EF 40–55%)" → midpoint 47.5 → 0
     - "Depressed" → 1
     - "Can't assess" and blank → no information.
- Straddling studies are **never** treated as undetermined or unverified. That would make V depend on the true EF (MNAR by construction) and remove borderline cases, which inflates accuracy.
- Sensitivity analysis: every straddling study counted as D = 1 (lower-bound rule). Together with the primary analysis, this bounds misclassification in the reference standard.

### 6.4 Verification model

Logistic regression for P(V = 1 | X), with X:

- AI score (restricted cubic spline)
- Age (spline), sex, approximate era
  - MIMIC shifts dates per patient; the true year is known only through `anchor_year_group` (3-year bins).
  - Approximate year = midpoint of `anchor_year_group` + (year(`ecg_time`) − `anchor_year`), accurate to about ±1.5 years. It is entered as an era category: 2008–2010, 2011–2013, 2014–2016, 2017–2019, 2020–2022.
- Setting at ECG: ED / inpatient / outpatient (from `edstays` and `admissions` times)
- Machine ECG findings: AF/flutter, LBBB, RBBB, infarct/Q-wave statement, LVH, low voltage; QRS duration and heart rate (paced ECGs are excluded, §6.2)
- Prior codes: CAD/MI, AF, valve disease, hypertension, diabetes, CKD, COPD. These come only from admissions with `dischtime` < index ECG time, as in §6.2. The one exception is status codes describing a pre-existing state, listed below.
  - "Valve disease" includes pre-index prosthesis evidence: prior codes Z95.2/Z95.3/Z95.4/V43.3/V42.2; prior valve-replacement procedures (ICD-9 35.05–35.08, 35.2x; ICD-10-PCS 02RF/02RG/02RH/02RJ); the same status codes on the index admission; and prior TTE prosthesis descriptors.
- Prior resting TTE with LVEF information within 365 days before the index ECG (yes/no). Prior LVEF ≤45% is excluded (§6.2), so these are prior non-reduced studies. They cover 3.3% of the skeleton cohort (1.7% within 30 days), and such patients are less likely to be re-referred for reasons not otherwise in X (added 1 Nov 2026).
- ED chief complaint category (keyword-mapped) if the ECG falls within an ED stay: dyspnoea, chest pain, syncope, oedema, other

Report the verification gradient (P(V) by AI-score decile and by setting) and the C-statistic of the verification model.

**Implementation (WP1b, 1 Nov 2026).** Built by `src/03_wp1b_covariates.py` → `outputs/wp1b_covariates.parquet`. Every code, statement and complaint that fires is listed in `outputs/codebook_icd.csv`, `codebook_ecg_statements.csv` and `codebook_ed_cc.csv`.
- **Machine ECG findings:** per-statement regex on `report_0..17`.
  - LBBB and RBBB are complete blocks only ("incomplete" and "atypical" excluded).
  - The infarct/Q flag is any statement with "infarct" or "Q wave", including "possible" and "consider" forms.
  - LVH includes "possible", "probable" and "consider" forms but excludes "normal variant".
  - AF/flutter comes from afef `rhythm_rules_v2`.
  - HR = 60000/RR and QRS = `qrs_end` − `qrs_onset`, with afef plausibility gates; 0.1% missing.
- **ED chief complaint:** four non-exclusive keyword flags (dyspnoea, chest pain incl. STEMI/NSTEMI, syncope incl. presyncope, oedema excl. angioedema), plus "other" and "missing". These are set only when the index ECG falls within an ED stay.
- **Added:** `prior_admission`, an indicator for any admission discharged before the index ECG. Prior codes can be observed only in these patients, and only 10.5% have one, so the indicator separates "no history recorded" from "no disease".
  - As a result, prior codes are sparse: HTN 3.4%, DM 1.4%, CAD/MI 0.5%. The verification model will rest mainly on ECG findings, setting and ED complaint.
- **V30 by flag (descriptive, score-blind):** AF/flutter 38.1% vs 15.1% without; LBBB 33.9% vs 16.3%; infarct/Q 28.9% vs 14.4%; ED dyspnoea 30.8% vs 16.1%; pre-index prosthesis 30.8% vs 16.5% (`outputs/wp1b_summary.json`).

### 6.5 Estimators (same threshold throughout)

1. **Naive:** verified ECGs only. Sensitivity, specificity, PPV, NPV, AUROC.
2. **MAR-corrected (primary):** inverse-probability weighting of verified ECGs by 1/P̂(V = 1 | X) [3], with weights truncated at the 99th percentile. Report the same metrics plus corrected prevalence, ECGs screened per case and echos per case.
3. **MNAR range:** outcome-model imputation for unverified ECGs, with
   P(D = 1 | X, V = 0) = expit(logit P̂(D = 1 | X, V = 1) + log λ), λ ∈ {0.25, 0.5, 0.75, 1, 1.5, 2}.
   - λ < 1 encodes unrecorded clinical suspicion: unverified patients are less often diseased.
   - λ > 1 encodes the opposite: unverified patients are more often diseased, e.g. through death before echo (2,524 patients, §8). Added 1 Nov 2026.
   - λ = 1 must closely reproduce estimator 2; this is a built-in consistency check.

CIs come from a nonparametric bootstrap over patients (1,000 resamples), refitting the verification and outcome models in each resample.

**Primary estimand:** Δsens and Δspec (naive − MAR-corrected).

**Secondary:**
- Corrected PPV and echos per true positive
- AUROC, naive vs corrected
- Corrected estimates by setting (outpatient is closest to screening)

### 6.6 Critical controls

| Threat | Control |
|---|---|
| Referral driven by ECG findings the model also reads | Score and machine ECG findings in the verification model (this is the mechanism IPW corrects) |
| Unrecorded clinical suspicion (MNAR) | λ range (§6.5); conclusions reported as bounds if they flip |
| "Screening" cohort containing known disease | Prior LVEF ≤45% and HF/cardiomyopathy codes excluded |
| Arbitrary verification window | Sensitivity analysis with a 0–365-day window, matching EchoNext's 1-year label horizon [11,12] |
| Extreme weights (positivity) | Truncation; report the % truncated and the weight distribution |
| LVEF reported as a range straddling 45 (3,224 TTEs; 10.8% of definite positives) | Midpoint rule (primary); all straddlers counted as D = 1 (sensitivity); never treated as unverified (§6.3) |
| Recent normal echo suppresses re-referral | Prior TTE within 365 days in the verification model (§6.4) |
| Exclusions that depend on verification | Exclusions use pre-index information only; the verifying TTE is never used to exclude (§6.2) |

## 7. Work packages and gates

- **WP1 — Cohort & verification linkage** (DuckDB). Report the verification rate.
  **G1:** at least 100 verified LVEF ≤45% cases. If not met, stop.
  **Result (1 Nov 2026, `src/02_wp1_cohort.py` → `outputs/wp1_summary.json`, `wp1_flow.csv`, `wp1_cohort.parquet`): G1 passed.**
  - Flow: 161,352 first ECGs → 161,332 linked with sex recorded → 161,287 adults → 158,903 non-paced (2,384 paced: 942 ventricular, 859 unspecified, 583 atrial) → 158,262 without prior TTE LVEF ≤45 → **157,841** without prior HF/cardiomyopathy code.
  - Setting (ED stay takes precedence over admission): outpatient 82,652; ED 51,619; inpatient 23,570.
  - Verification: **V30 = 26,084 (16.5%)**; ED 17.4%, inpatient 27.2%, outpatient 13.0%. V365 = 22.4%. Median time to the verifying TTE is 1.5 days (IQR 0.7–4.0). Sources of the verifying TTE: reported range 25,351; biplane 551; 3D 144; text 38.
  - **Verified cases with LVEF ≤45%: 4,423** (prevalence 17.0% among verified); 4,908 with straddlers counted as D = 1; 5,323 within 365 days.
  - Unverified patients who died within 30 days: 2,524 (see §8).
  - Technically unusable tracings are excluded in WP2, where waveforms are read.
- **WP2 — Scoring.** Verify the output head, input spec and threshold first.
  **G2 (transport):** naive AUROC ≥0.80. If not met, stop, because the paper would become a transport-failure study, which is a different question.
  Prior evidence, not blind: in afef the same model and pipeline gave AUROC 0.849 (95% CI 0.842–0.856; n = 16,245; prevalence 17.5%) on non-audit MIMIC ECG–echo pairs (±24 h, nearest ECG per echo). That is a different cohort from §6.2/§6.3, so G2 is still evaluated, but it is expected to pass.
- **WP3 — Analysis:** §6.4 → §6.5 (estimators 1 → 2 → 3) → secondary → sensitivity analyses (0–365-day window; straddling LVEF ranges as D = 1).

## 8. Limitations accepted up front

- MAR is untestable. The λ range bounds the dependence on it but does not remove it.
- Echos done outside BIDMC are invisible and appear as unverified.
- Death precludes verification. 2,524 unverified patients (1.6% of the cohort) died within 30 days of the index ECG. Their LVSD prevalence is plausibly higher, which is an MNAR mechanism in the direction opposite to λ < 1. Death is post-index, so it cannot enter X. It is reported, not modelled.
- The intended-use population is a hospital ECG population (ED/inpatient heavy), not a community screening population.
- The study uses one model, one target and one threshold.
- History is left-truncated. The first MIMIC-IV-ECG tracing is often the first contact, so known LVSD or HF diagnosed elsewhere or before 2008 is invisible. The pre-index exclusion removed only 1,062 of 158,903 adult non-paced patients (0.67%; WP1). Index ECGs also pile up in the early years (2008–2010: 54,936 of 157,841), because "first in the database" is not "first ever". Index-admission codes cannot fill the gap, because they include diagnoses made after the index ECG. Some patients in the "screening" cohort will therefore have known LVSD.

## 9. Out of scope (do not add without a decision)

- Training or fine-tuning any model; other EchoNext targets.
- Doubly robust or semiparametric-efficient estimators.
- Within-patient referral-moment analysis (whether echo follows a patient's highest-score ECG).
- Community or protocol-echo cohorts; cost-effectiveness.

## 10. Open decisions

- [x] EchoNext weights: Mini-Model, local and verified, with an `lvef_lte_45` output (§6.1).
- [x] EchoNext-Mini [12] publishes binary metrics only for composite SHD, at the cutoff giving 70% sensitivity on its test split. It gives no LVEF ≤45% threshold and no cutoff value (§6.1). Checked in the full text.
- [x] Paced ECGs: excluded (31 Oct 2026). A patient whose first ECG is paced is excluded (§6.2).
- [x] Prosthetic valves: not excluded (1 Nov 2026). Pre-index prosthesis evidence enters the verification model, and the verifying TTE is never used to exclude (§6.2, §6.4).
- [x] Threshold rule: source-anchored (31 Oct 2026). Positive if score ≥ 0.5675751566886902, from the 70%-sensitivity rule on `lvef_lte_45` in the EchoNext test split (§6.1).
- [x] LVEF ranges straddling 45: midpoint rule as primary, all counted as D = 1 as sensitivity, never unverified (1 Nov 2026, §6.3).
- [x] Primary verification window: 30 days (1 Nov 2026, §6.3).
- [x] Rhythm: all non-paced rhythms (1 Nov 2026). AF and flutter are verified 2.6 times as often as sinus (V30 38.3% / 36.8% vs 14.8%), so they are central to the mechanism. No sinus-only analysis.
- [x] Verification model: prior TTE within 365 days added as a covariate (1 Nov 2026, §6.4).

## Key references (peer-reviewed / dataset)

1. Tang Q, et al. AI-ECG for echocardiography triage in structural heart disease: evidence, implementation, and future directions. *Int J Gen Med.* 2026. doi:10.2147/IJGM.S610298
2. Begg CB, Greenes RA. Assessment of diagnostic tests when disease verification is subject to selection bias. *Biometrics.* 1983. doi:10.2307/2530820
3. Alonzo TA, Pepe MS. Assessing accuracy of a continuous screening test in the presence of verification bias. *J R Stat Soc Ser C Appl Stat.* 2005. doi:10.1111/j.1467-9876.2005.00477.x
4. Gow B, Pollard T, Greenbaum N, et al. MIMIC-IV-ECHO: Echocardiogram Matched Subset (version 1.0). *PhysioNet.* 2026. doi:10.13026/nrjh-5r77
5. Kashou AH, et al. Artificial intelligence-augmented electrocardiogram detection of left ventricular systolic dysfunction in the general population. *Mayo Clin Proc.* 2021. doi:10.1016/j.mayocp.2021.02.029
6. Rhee TM, et al. Artificial intelligence–driven electrocardiogram screening for asymptomatic left ventricular systolic dysfunction in the general population. *JACC Adv.* 2026. doi:10.1016/j.jacadv.2026.102660
7. Carter RE, et al. Multisite, external validation of an AI-enabled ECG algorithm for detection of low ejection fraction. *JACC Adv.* 2026. doi:10.1016/j.jacadv.2025.102537
8. Arifin WN, et al. Correcting for partial verification bias in diagnostic accuracy studies: a tutorial using R. *Stat Med.* 2022. doi:10.1002/sim.9311
9. Poterucha TJ, et al. Detecting structural heart disease from electrocardiograms using AI. *Nature.* 2025. doi:10.1038/s41586-025-09227-0
10. Elias P, Finer J. EchoNext: a dataset for detecting echocardiogram-confirmed structural heart disease from ECGs (v1.1.1). *PhysioNet.* 2026. doi:10.13026/7cfw-d091
11. EchoNext: an ECG-based deep learning model to detect structural heart disease (Abstract 14647). *Circulation.* 2023;148(Suppl 1). doi:10.1161/circ.148.suppl_1.14647
12. Hughes J, Jing L, Finer J, et al. EchoNext-Mini: a dataset and baseline AI model for detecting structural heart disease from electrocardiograms. *NEJM AI.* 2026;3(5). doi:10.1056/AIdbp2500516
