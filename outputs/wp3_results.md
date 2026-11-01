# WP3 results — VERIFY-GAP

Generated 2026-11-01T06:18:11 by `src/07_wp3_analysis.py` (B = 1000, seed 20261101). Analytic cohort 157,623; verified (V30) 26,041. Threshold: score ≥ 0.5675751566886902. Values are % (95% bootstrap CI) unless stated.

## Verification model

- C-statistic 0.7713; 38 design columns.
- V30 by AI-score decile (low → high): 5.0, 6.5, 8.0, 9.4, 11.6, 14.1, 17.7, 22.6, 28.8, 41.5%.
- V30 by test result: T = 1 39.7%, T = 0 13.2%.
- V30 by setting: ED 17.4%, inpatient 27.2%, outpatient 13.0%.
- IPW weights (untruncated): median 3.9143, p90 11.4411, p99 34.0875, max 280.753; 1.0% truncated at 34.1.

## Main estimators

| Metric | Naive (verified only) | IPW (MAR, primary) | Δ (naive − IPW) |
|---|---|---|---|
| Sensitivity | 67.9 (66.5–69.2) | 51.8 (49.9–53.7) | 16.1 (14.9–17.3) |
| Specificity | 77.3 (76.7–77.9) | 90.1 (89.8–90.4) | -12.8 (-13.2–-12.3) |
| PPV | 37.9 (36.9–39.0) | 34.3 (33.1–35.5) | 3.6 (3.0–4.3) |
| NPV | 92.2 (91.8–92.6) | 94.9 (94.6–95.2) | -2.7 (-3.0–-2.5) |
| Prevalence | 17.0 (16.5–17.4) | 9.1 (8.7–9.4) |  |
| AUROC | 80.1 (79.4–80.8) | 81.7 (80.6–82.9) | -1.6 (-2.5–-0.8) |
| ECGs per detected case | — | 21.2 (20.4–22.1) |  |
| Echos per detected case | 2.6 (2.6–2.7) | 2.9 (2.8–3.0) |  |

Naive 'ECGs per detected case' is undefined (only verified patients are observed).

## MNAR range (outcome-model imputation)

| λ | Sensitivity | Specificity | PPV | NPV | Prevalence | AUROC | Echos per case |
|---|---|---|---|---|---|---|---|
| 0.25 | 61.4 (59.9–62.9) | 89.7 (89.6–89.9) | 22.5 (21.9–23.3) | 98.0 (97.8–98.1) | 4.6 (4.5–4.8) | 85.7 (84.8–86.4) | 4.4 (4.3–4.6) |
| 0.5 | 57.0 (55.4–58.5) | 90.2 (90.1–90.4) | 27.5 (26.7–28.4) | 97.0 (96.8–97.2) | 6.1 (5.9–6.3) | 84.0 (82.9–84.9) | 3.6 (3.5–3.7) |
| 0.75 | 53.7 (52.1–55.2) | 90.6 (90.5–90.8) | 31.4 (30.5–32.3) | 96.1 (95.8–96.3) | 7.4 (7.1–7.7) | 82.8 (81.6–83.8) | 3.2 (3.1–3.3) |
| 1.0 | 51.1 (49.4–52.6) | 90.9 (90.8–91.1) | 34.4 (33.5–35.4) | 95.2 (94.9–95.5) | 8.5 (8.2–8.9) | 81.9 (80.6–83.0) | 2.9 (2.8–3.0) |
| 1.5 | 47.0 (45.4–48.5) | 91.4 (91.2–91.6) | 39.2 (38.2–40.3) | 93.6 (93.1–94.0) | 10.5 (10.1–11.0) | 80.5 (79.0–81.7) | 2.5 (2.5–2.6) |
| 2.0 | 44.0 (42.3–45.5) | 91.8 (91.6–91.9) | 42.8 (41.8–43.9) | 92.1 (91.5–92.6) | 12.3 (11.8–12.9) | 79.4 (77.8–80.7) | 2.3 (2.3–2.4) |

λ = 1 minus IPW (points): sens -0.73, spec +0.86, ppv +0.14, npv +0.30, prev -0.56, auroc +0.18.

## By setting

| Setting | Estimator | Sensitivity | Specificity | PPV | Prevalence | AUROC |
|---|---|---|---|---|---|---|
| outpatient | naive | 64.2 (61.8–66.4) | 80.7 (79.8–81.5) | 36.5 (34.7–38.2) | 14.7 (14.0–15.4) | 80.4 (79.3–81.5) |
| outpatient | ipw | 46.7 (43.8–49.3) | 91.9 (91.5–92.3) | 33.3 (31.5–35.0) | 8.0 (7.5–8.4) | 81.2 (79.4–82.9) |
| ED | naive | 69.2 (66.9–71.4) | 75.8 (74.8–76.7) | 37.3 (35.6–39.1) | 17.3 (16.5–18.0) | 79.9 (78.7–81.1) |
| ED | ipw | 55.2 (51.6–58.9) | 88.7 (87.8–89.5) | 33.0 (30.6–35.4) | 9.2 (8.5–9.9) | 82.0 (79.9–84.0) |
| inpatient | naive | 71.0 (68.6–73.4) | 73.6 (72.3–74.7) | 40.5 (38.5–42.6) | 20.2 (19.3–21.3) | 79.3 (78.0–80.7) |
| inpatient | ipw | 58.6 (55.8–61.6) | 86.0 (85.2–86.8) | 38.1 (36.1–40.2) | 12.8 (12.1–13.6) | 81.3 (79.6–82.9) |

## Sensitivity analyses

| Analysis | Δsens | Δspec | ΔPPV | IPW sens | IPW spec |
|---|---|---|---|---|---|
| Primary (0–30 d, midpoint rule) | 16.1 (14.9–17.3) | -12.8 (-13.2–-12.3) | 3.6 (3.0–4.3) | 51.8 (49.9–53.7) | 90.1 (89.8–90.4) |
| 0–365-day window | 14.1 (13.2–14.9) | -10.8 (-11.1–-10.4) | 2.5 (2.1–2.9) | 51.9 (50.5–53.4) | 90.4 (90.1–90.6) |
| Straddlers as D = 1 | 16.5 (15.4–17.5) | -12.5 (-13.0–-12.1) | 3.6 (3.0–4.3) | 49.3 (47.4–51.0) | 90.4 (90.1–90.7) |

## Pre-specified readout (§4)

- Δsens 16.1 (14.9–17.3), Δspec -12.8 (-13.2–-12.3).
- **Category: Verified-only validation misstates intended-use accuracy.** Material (≥5 points) at every λ: True.
- H1: naive sens > corrected True; naive spec < corrected True.
- H2 (MAR): corrected PPV < naive True.
