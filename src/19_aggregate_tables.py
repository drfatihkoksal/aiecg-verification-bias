"""Aggregate tables that need patient-level data, written as CSV (aggregates only).

  outputs/tables/table1.csv            cohort characteristics by verification status
  outputs/tables/s6_verification.csv   verification within 30 days by recorded characteristic
  outputs/tables/s7_crosstab.csv       index test against reference standard among verified patients (STARD 2015 item 23)
  outputs/tables/s8_severity.csv       severity of LVSD among verified cases, naive sensitivity by severity (item 21a)
  outputs/cohort_characteristics.json  cohort shares used in the text (setting, age, sex, positive EchoNext-Mini)

Inputs: the intended-use cohort (src/02), covariates (src/03), EchoNext-Mini scores (src/05), DeepECG-SL scores
(src/12) and the DeepECG-SL matched-positivity threshold (outputs/wp3_deepecg.json).
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

P = Path(__file__).resolve().parents[1]
OUT = P / "outputs"; TAB = OUT / "tables"; TAB.mkdir(exist_ok=True)
AFEF = Path("/home/fatih/projects/afef/outputs")
dg = json.loads((OUT / "wp3_deepecg.json").read_text())
thr_d = dg["cohort"]["threshold_matched_pred_lvef"]

k = pd.read_parquet(OUT / "wp1_cohort.parquet"); x = pd.read_parquet(OUT / "wp1b_covariates.parquet"); s = pd.read_parquet(OUT / "wp2_scores.parquet")
dgs = pd.read_parquet(OUT / "wp3_deepecg_scores.parquet")
d = k.merge(x, on=["subject_id", "study_id"]).merge(s[["subject_id", "positive", "technical_exclusion"]], on="subject_id")
d = d[~d.technical_exclusion].merge(dgs, on=["subject_id", "study_id"], how="left").reset_index(drop=True)
d["dg_pos"] = np.where(d.pred_lvef.isna(), np.nan, (d.pred_lvef <= thr_d).astype(float))

# ---------------- Table 1: characteristics by verification status
grp = {"All": d, "Echo within 30 days": d[d.V30 == 1], "No echo within 30 days": d[d.V30 == 0]}


def med(col, nd=0):
    return {g: f"{v[col].median():.{nd}f} ({v[col].quantile(.25):.{nd}f}–{v[col].quantile(.75):.{nd}f})" for g, v in grp.items()}


def pct(mask_fn):
    out = {}
    for g, v in grp.items():
        mm = mask_fn(v)
        out[g] = f"{int(mm.sum()):,} ({100 * mm.mean():.1f})"
    return out


rows = [{"Characteristic": "Patients, n", **{g: f"{len(v):,}" for g, v in grp.items()}},
        {"Characteristic": "Age, y", **med("age")},
        {"Characteristic": "Female", **pct(lambda v: v.gender == "F")},
        {"Characteristic": "Setting of the index ECG", **{g: "" for g in grp}}]
for lab, val in [("  Outpatient", "outpatient"), ("  Emergency department", "ED"), ("  Inpatient", "inpatient")]:
    rows.append({"Characteristic": lab, **pct(lambda v, val=val: v.setting == val)})
rows.append({"Characteristic": "Index ECG findings", **{g: "" for g in grp}})
for lab, col in [("  Atrial fibrillation or flutter", "ecg_af_flutter"), ("  Left bundle branch block", "ecg_lbbb"),
                 ("  Right bundle branch block", "ecg_rbbb"), ("  Left ventricular hypertrophy", "ecg_lvh"),
                 ("  Infarct or Q-wave statement", "ecg_infarct_q"), ("  Low QRS voltage", "ecg_low_voltage")]:
    rows.append({"Characteristic": lab, **pct(lambda v, col=col: v[col])})
rows.append({"Characteristic": "  QRS duration, ms", **med("qrs_ms")})
rows.append({"Characteristic": "  Heart rate, beats/min", **med("hr")})
rows.append({"Characteristic": "Recorded history", **{g: "" for g in grp}})
for lab, col in [("  Hypertension", "hx_htn"), ("  Diabetes", "hx_dm"), ("  Chronic kidney disease", "hx_ckd"),
                 ("  COPD", "hx_copd"), ("  Coronary artery disease or MI", "prior_cad_mi"),
                 ("  Valve disease or prosthetic valve", "prior_valve"), ("  TTE in the previous year", "prior_tte_365")]:
    rows.append({"Characteristic": lab, **pct(lambda v, col=col: v[col])})
rows.append({"Characteristic": "Positive AI-ECG", **{g: "" for g in grp}})
rows.append({"Characteristic": "  EchoNext-Mini", **pct(lambda v: v.positive == 1)})
rows.append({"Characteristic": "  DeepECG-SL (matched positivity)", **pct(lambda v: v.dg_pos == 1)})
rows.append({"Characteristic": "LVEF ≤45% on the verifying TTE", "All": "—",
             "Echo within 30 days": f"{int((grp['Echo within 30 days'].d_v30 == 1).sum()):,} ({100 * (grp['Echo within 30 days'].d_v30 == 1).mean():.1f})",
             "No echo within 30 days": "—"})
pd.DataFrame(rows).to_csv(TAB / "table1.csv", index=False)

(OUT / "cohort_characteristics.json").write_text(json.dumps({
    "n": int(len(d)),
    "pct_outpatient": 100 * float((d.setting == "outpatient").mean()), "pct_ed": 100 * float((d.setting == "ED").mean()),
    "pct_inpatient": 100 * float((d.setting == "inpatient").mean()), "age_median": float(d.age.median()),
    "pct_female": 100 * float((d.gender == "F").mean()), "pct_positive_echonext": 100 * float((d.positive == 1).mean())}, indent=2))

# ---------------- S6: verification by recorded characteristic
FL = [("Atrial fibrillation or flutter", "ecg_af_flutter"), ("Left bundle branch block", "ecg_lbbb"), ("Right bundle branch block", "ecg_rbbb"),
      ("Left ventricular hypertrophy", "ecg_lvh"), ("Infarct or Q-wave statement", "ecg_infarct_q"), ("Low QRS voltage", "ecg_low_voltage"),
      ("Hypertension", "hx_htn"), ("Diabetes", "hx_dm"), ("Chronic kidney disease", "hx_ckd"), ("COPD", "hx_copd"),
      ("Coronary artery disease or MI", "prior_cad_mi"), ("Valve disease or prosthetic valve", "prior_valve"), ("TTE in the previous year", "prior_tte_365"),
      ("ED complaint: dyspnea", "ed_cc_dyspnoea"), ("ED complaint: chest pain", "ed_cc_chest_pain"), ("ED complaint: syncope", "ed_cc_syncope"),
      ("ED complaint: edema", "ed_cc_oedema"), ("EchoNext-Mini positive", None)]
rows = []
for lab_, col in FL:
    mask = (d.positive == 1) if col is None else d[col].astype(bool)
    rows.append({"Characteristic": lab_, "Present, n": f"{int(mask.sum()):,}", "Verified if present, %": f"{100 * d.V30[mask].mean():.1f}",
                 "Verified if absent, %": f"{100 * d.V30[~mask].mean():.1f}"})
pd.DataFrame(rows).to_csv(TAB / "s6_verification.csv", index=False)

# ---------------- S7: index test by reference standard among verified patients
ver = d[d.V30 == 1]
rows = []
for mod, pos in [("EchoNext-Mini (frozen threshold)", ver.positive == 1), ("DeepECG-SL (matched positivity)", ver.pred_lvef <= thr_d)]:
    vv = ver if mod.startswith("Echo") else ver[ver.pred_lvef.notna()]
    pos = pos if mod.startswith("Echo") else (vv.pred_lvef <= thr_d)
    y = vv.d_v30 == 1
    rows += [{"Model": mod, "Index test": "Positive", "LVEF ≤45%": f"{int((pos & y).sum()):,}", "LVEF >45%": f"{int((pos & ~y).sum()):,}", "Total": f"{int(pos.sum()):,}"},
             {"Model": "", "Index test": "Negative", "LVEF ≤45%": f"{int((~pos & y).sum()):,}", "LVEF >45%": f"{int((~pos & ~y).sum()):,}", "Total": f"{int((~pos).sum()):,}"},
             {"Model": "", "Index test": "Total", "LVEF ≤45%": f"{int(y.sum()):,}", "LVEF >45%": f"{int((~y).sum()):,}", "Total": f"{len(vv):,}"}]
pd.DataFrame(rows).to_csv(TAB / "s7_crosstab.csv", index=False)

# ---------------- S8: severity, best numeric LVEF of the verifying TTE (same source order as the reference standard)
ef = pd.read_parquet(AFEF / "echo_ef_studies.parquet",
                     columns=["measurement_id", "lvef_low", "lvef_high", "lvef_biplane", "lvef_3d", "lv_function_text"])
okv = lambda v: v.where(v.between(5, 95))
bp, lo, hi, d3 = okv(ef.lvef_biplane), okv(ef.lvef_low), okv(ef.lvef_high), okv(ef.lvef_3d)
mid = np.where(lo.notna() & hi.notna(), (np.fmin(lo, hi) + np.fmax(lo, hi)) / 2, lo)
ef["ef_value"] = np.where(bp.notna(), bp, np.where(lo.notna(), mid, d3))
cases = ver[ver.d_v30 == 1].merge(ef[["measurement_id", "ef_value"]], left_on="tte_id_v30", right_on="measurement_id", how="left")
cat = pd.cut(cases.ef_value, [0, 30, 40, 45.0001], labels=["≤30%", "31%-40%", "41%-45%"], right=True)
cases["severity"] = cat.astype(str).where(cases.ef_value.notna(), "Qualitative only")
rows = []
for sev in ["≤30%", "31%-40%", "41%-45%", "Qualitative only"]:
    c = cases[cases.severity == sev]
    if len(c) == 0:
        continue
    rows.append({"LVEF on verifying TTE": sev, "Cases, n (%)": f"{len(c):,} ({100 * len(c) / len(cases):.1f})",
                 "EchoNext-Mini positive, %": f"{100 * (c.positive == 1).mean():.1f}",
                 "DeepECG-SL positive, %": f"{100 * (c.pred_lvef <= thr_d)[c.pred_lvef.notna()].mean():.1f}"})
pd.DataFrame(rows).to_csv(TAB / "s8_severity.csv", index=False)
print("written:", ", ".join(["table1.csv", "s6_verification.csv", "s7_crosstab.csv", "s8_severity.csv", "cohort_characteristics.json"]))
