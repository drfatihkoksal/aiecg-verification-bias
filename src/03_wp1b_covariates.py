"""WP1b - verification-model covariates X (concept.md §6.4), AI score excluded (WP2).

Score-blind. One row per cohort patient. Codebooks are written out so that every flag is auditable.
Outputs:
  outputs/wp1b_covariates.parquet      subject_id + X
  outputs/codebook_icd.csv             every ICD code that fired, by group, with title and n patients
  outputs/codebook_ecg_statements.csv  every machine statement that fired, by flag
  outputs/codebook_ed_cc.csv           every ED chief complaint that fired, by flag
  outputs/wp1b_summary.json            prevalence, missingness, V30 by flag
"""
import json, re, datetime
from pathlib import Path
import numpy as np, pandas as pd, duckdb

R = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c")
HOSP = R / "mimiciv/3.1/hosp"
ECG = R / "mimic-iv-ecg/1.0"
ED = R / "mimic-iv-ed-2.2/ed"
SM = R / "echo/echo/structured-measurement.csv.gz"
OUT = Path(__file__).resolve().parents[1] / "outputs"

c = duckdb.connect()
c.execute("SET threads=16")
c.execute("SET enable_progress_bar=false")
q = lambda s: c.execute(s).df()
c.execute(f"CREATE TABLE k AS SELECT subject_id, study_id, ecg_time, index_hadm_id, ed_stay_id, V30 FROM read_parquet('{OUT / 'wp1_cohort.parquet'}')")
c.execute(f"CREATE TABLE adm AS SELECT subject_id, hadm_id, CAST(admittime AS TIMESTAMP) admittime, CAST(dischtime AS TIMESTAMP) dischtime FROM read_csv_auto('{HOSP / 'admissions.csv.gz'}')")
c.execute(f"CREATE TABLE dx AS SELECT CAST(subject_id AS BIGINT) subject_id, CAST(hadm_id AS BIGINT) hadm_id, icd_code, icd_version FROM read_csv_auto('{HOSP / 'diagnoses_icd.csv.gz'}', all_varchar=true)")
c.execute(f"CREATE TABLE px AS SELECT CAST(subject_id AS BIGINT) subject_id, CAST(chartdate AS DATE) chartdate, icd_code, icd_version FROM read_csv_auto('{HOSP / 'procedures_icd.csv.gz'}', all_varchar=true)")
c.execute(f"CREATE TABLE d_dx AS SELECT icd_code, icd_version, long_title FROM read_csv_auto('{HOSP / 'd_icd_diagnoses.csv.gz'}', all_varchar=true)")
c.execute(f"CREATE TABLE d_px AS SELECT icd_code, icd_version, long_title FROM read_csv_auto('{HOSP / 'd_icd_procedures.csv.gz'}', all_varchar=true)")

# ------------------------------------------------------------------ 1. prior diagnosis codes (dischtime < index ECG)
ICD = {  # group: (ICD-10 prefixes, ICD-9 prefixes)
    "cad_mi":   (["I20", "I21", "I22", "I23", "I24", "I25"], ["410", "411", "412", "413", "414"]),
    "af":       (["I48"], ["4273"]),
    "valve_dx": (["I05", "I06", "I07", "I08", "I34", "I35", "I36", "I37"], ["394", "395", "396", "3970", "3971", "4240", "4241", "4242", "4243"]),
    "htn":      (["I10", "I11", "I12", "I13", "I15", "I16"], ["401", "402", "403", "404", "405"]),
    "dm":       (["E08", "E09", "E10", "E11", "E13"], ["250"]),
    "ckd":      (["N18"], ["585"]),
    "copd":     (["J43", "J44"], ["491", "492", "496"]),
}
PROSTH_DX = (["Z952", "Z953", "Z954"], ["V433", "V422"])                     # status codes (pre-existing state)
PROSTH_PX = (["02RF", "02RG", "02RH", "02RJ"], ["352", "3505", "3506", "3507", "3508"])


def cond(prefixes):
    v10, v9 = prefixes
    a = " OR ".join(f"icd_code LIKE '{p}%'" for p in v10)
    b = " OR ".join(f"icd_code LIKE '{p}%'" for p in v9)
    return f"((icd_version='10' AND ({a})) OR (icd_version='9' AND ({b})))"


c.execute("""CREATE TABLE prior_dx AS
SELECT DISTINCT k.subject_id, dx.icd_code, dx.icd_version
FROM k JOIN adm a ON a.subject_id=k.subject_id AND a.dischtime < k.ecg_time
JOIN dx ON dx.hadm_id=a.hadm_id""")
X = q("SELECT subject_id, study_id FROM k ORDER BY subject_id")
X["prior_admission"] = X.subject_id.isin(q("SELECT DISTINCT k.subject_id FROM k JOIN adm a ON a.subject_id=k.subject_id AND a.dischtime < k.ecg_time").subject_id)
cb = []
for g, pre in ICD.items():
    hits = q(f"SELECT * FROM prior_dx WHERE {cond(pre)}")
    X[f"prior_{g}"] = X.subject_id.isin(hits.subject_id)
    cb.append(q(f"""SELECT '{g}' grp, p.icd_version, p.icd_code, any_value(d.long_title) title, count(DISTINCT p.subject_id) n_patients
                    FROM prior_dx p LEFT JOIN d_dx d USING (icd_code, icd_version) WHERE {cond(pre).replace('icd_', 'p.icd_')}
                    GROUP BY ALL"""))

# ------------------------------------------------------------------ 1b. chronic conditions also from the index admission (spec v2, 1 Nov 2026)
# Index admission = the admission containing the index ECG, or the admission the index ED stay led to.
# Only chronic conditions that a verifying echo cannot newly create (HTN, DM, CKD, COPD); never CAD/AF/valve.
CHRONIC = ["htn", "dm", "ckd", "copd"]
c.execute(f"""CREATE TABLE idx_adm AS
SELECT subject_id, index_hadm_id AS hadm_id FROM k WHERE index_hadm_id IS NOT NULL
UNION
SELECT k.subject_id, CAST(e.hadm_id AS BIGINT) FROM k JOIN read_csv_auto('{ED / "edstays.csv.gz"}') e
  ON e.stay_id = k.ed_stay_id WHERE e.hadm_id IS NOT NULL""")
c.execute("CREATE TABLE idx_dx AS SELECT DISTINCT i.subject_id, dx.icd_code, dx.icd_version FROM idx_adm i JOIN dx ON dx.hadm_id=i.hadm_id")
X["index_admitted"] = X.subject_id.isin(q("SELECT DISTINCT subject_id FROM idx_adm").subject_id)
for g in CHRONIC:
    hits = q(f"SELECT DISTINCT subject_id FROM idx_dx WHERE {cond(ICD[g])}").subject_id
    X[f"hx_{g}"] = X[f"prior_{g}"] | X.subject_id.isin(hits)
    cb.append(q(f"""SELECT '{g}_index_admission' grp, p.icd_version, p.icd_code, any_value(d.long_title) title,
                    count(DISTINCT p.subject_id) n_patients
                    FROM idx_dx p LEFT JOIN d_dx d USING (icd_code, icd_version) WHERE {cond(ICD[g]).replace('icd_', 'p.icd_')}
                    GROUP BY ALL"""))

# ------------------------------------------------------------------ 2. pre-index prosthetic-valve evidence
p_dx = q(f"SELECT DISTINCT subject_id FROM prior_dx WHERE {cond(PROSTH_DX)}").subject_id
p_px = q(f"SELECT DISTINCT k.subject_id FROM k JOIN px ON px.subject_id=k.subject_id AND px.chartdate < CAST(k.ecg_time AS DATE) WHERE {cond(PROSTH_PX).replace('icd_', 'px.icd_')}").subject_id
p_idx = q(f"SELECT DISTINCT k.subject_id FROM k JOIN dx ON dx.hadm_id=k.index_hadm_id WHERE {cond(PROSTH_DX).replace('icd_', 'dx.icd_')}").subject_id
c.execute(f"""
CREATE TABLE echo_prosth AS
SELECT DISTINCT CAST(subject_id AS BIGINT) subject_id, CAST(measurement_datetime AS TIMESTAMP) echo_time
FROM read_csv_auto('{SM}', all_varchar=true)
WHERE test_type='tte' AND result IS NOT NULL AND (
  (measurement='av_leaflets' AND result IN ('Bioprosthesis','Bileaflet mechanical','Mechanical','AVR homograft',
                                            'Other prosthesis','Single tilting disk','Homograft','SIngle tilting disc'))
  OR (measurement='mv_leaflets' AND result IN ('Bioprosthesis','Mechanical','SIngle tilting disc'))
  OR (measurement='tv_leaflets' AND result IN ('Bioprosthetic','Other prosthesis','Mechanical'))
  OR measurement IN ('avr_structure1','pvr_structure1')
  OR (measurement IN ('mvr_structure1','tvr_structure1') AND result ILIKE 'Prosthesis%'))""")
p_echo = q("SELECT DISTINCT k.subject_id FROM k JOIN echo_prosth e ON e.subject_id=k.subject_id AND e.echo_time < k.ecg_time").subject_id
X["prosthesis_pre_index"] = X.subject_id.isin(set(p_dx) | set(p_px) | set(p_idx) | set(p_echo))
X["prior_valve"] = X.prior_valve_dx | X.prosthesis_pre_index
for name, pre, tbl in [("prosthesis_status", PROSTH_DX, "d_dx"), ("prosthesis_proc", PROSTH_PX, "d_px")]:
    cb.append(q(f"""SELECT '{name}' grp, icd_version, icd_code, long_title title, NULL::BIGINT n_patients
                    FROM {tbl} WHERE {cond(pre)}"""))
pd.concat(cb).sort_values(["grp", "n_patients"], ascending=[True, False]).to_csv(OUT / "codebook_icd.csv", index=False)
prosth_sources = {"prior_status_code": int(len(p_dx)), "prior_procedure": int(len(p_px)),
                  "index_admission_status_code": int(len(p_idx)), "prior_tte_descriptor": int(len(p_echo))}

# ------------------------------------------------------------------ 3. machine ECG findings and numerics
cols = ", ".join(f"report_{i}" for i in range(18))
mm = q(f"""SELECT CAST(m.study_id AS BIGINT) study_id, [{cols}] AS stmts,
                  TRY_CAST(rr_interval AS DOUBLE) rr, TRY_CAST(qrs_onset AS DOUBLE) qon, TRY_CAST(qrs_end AS DOUBLE) qend
           FROM read_csv_auto('{ECG / "machine_measurements.csv"}', all_varchar=true) m
           JOIN k ON CAST(m.study_id AS BIGINT)=k.study_id""")
ECG_RX = {  # flag: (include regex, exclude regex) -- applied per statement, case-insensitive
    "lbbb": (r"left bundle branch block|\blbbb\b", r"incomplete|atypical"),
    "rbbb": (r"right bundle branch block|\brbbb\b", r"incomplete|atypical"),
    "infarct_q": (r"infarct|q wave", None),
    "lvh": (r"left ventricular hypertrophy|\blvh\b", r"normal variant"),
    "low_voltage": (r"low (?:qrs )?voltage", None),
}
stm = mm[["study_id", "stmts"]].explode("stmts").dropna()
stm["stmts"] = stm.stmts.str.strip()
stm = stm[stm.stmts != ""]
cbe = []
for f, (inc, exc) in ECG_RX.items():
    hit = stm.stmts.str.contains(inc, case=False, regex=True)
    if exc:
        hit &= ~stm.stmts.str.contains(exc, case=False, regex=True)
    X[f"ecg_{f}"] = X.study_id.isin(stm.study_id[hit])
    cbe.append(stm[hit].groupby("stmts").study_id.nunique().rename("n").reset_index().assign(flag=f))
pd.concat(cbe)[["flag", "stmts", "n"]].sort_values(["flag", "n"], ascending=[True, False]).to_csv(OUT / "codebook_ecg_statements.csv", index=False)
rhy = q(f"SELECT subject_id, rhythm_candidate FROM read_parquet('{OUT / 'wp1_cohort.parquet'}')")
X = X.merge(rhy, on="subject_id")
X["ecg_af_flutter"] = X.rhythm_candidate.isin(["AF", "atrial_flutter"])
mm["hr"] = (60000 / mm.rr).where(lambda s: s.between(20, 300))       # afef tabular_rules_v2 gates
mm["qrs_ms"] = (mm.qend - mm.qon).where(lambda s: s.between(40, 300))
X = X.merge(mm[["study_id", "hr", "qrs_ms"]], on="study_id", how="left")

# ------------------------------------------------------------------ 4. ED chief complaint (only if the index ECG is within an ED stay)
tri = q(f"""SELECT k.subject_id, lower(t.chiefcomplaint) cc FROM k
            LEFT JOIN read_csv_auto('{ED / "triage.csv.gz"}') t ON t.stay_id=k.ed_stay_id WHERE k.ed_stay_id IS NOT NULL""")
CC_RX = {
    "dyspnoea": r"dyspn|\bsob\b|short(?:ness)? of breath|\bdoe\b|difficulty breathing|trouble breathing|resp(?:iratory)? distress|hypox",
    "chest_pain": r"chest pain|\bcp\b|chest pressure|chest tight|chest discomfort|angina|\bn?stemi\b",
    "syncope": r"syncop|pre-?syncop|passed out|faint|loss of consciousness|\bloc\b",
    "oedema": r"(?<!angio)edema|(?<!angi)oedema|(?:leg|\ble|lower ext\w*|ankle|feet|foot|bilat\w*|extremit\w*) swelling|swelling (?:of )?(?:the )?(?:legs?|ankles?|feet|foot)",
}
cbc = []
for f, rx in CC_RX.items():
    hit = tri.cc.fillna("").str.contains(rx, regex=True)
    X[f"ed_cc_{f}"] = X.subject_id.isin(tri.subject_id[hit])
    cbc.append(tri[hit].groupby("cc").size().rename("n").reset_index().assign(flag=f))
pd.concat(cbc)[["flag", "cc", "n"]].sort_values(["flag", "n"], ascending=[True, False]).to_csv(OUT / "codebook_ed_cc.csv", index=False)
in_ed = X.subject_id.isin(tri.subject_id)
X["ed_cc_missing"] = X.subject_id.isin(tri.subject_id[tri.cc.isna()])
X["ed_cc_other"] = in_ed & ~X[[f"ed_cc_{f}" for f in CC_RX]].any(axis=1) & ~X.ed_cc_missing

# ------------------------------------------------------------------ write
X = X.drop(columns=["rhythm_candidate"])
assert X.subject_id.is_unique and len(X) == int(q("SELECT count(*) n FROM k").n[0])
X.to_parquet(OUT / "wp1b_covariates.parquet", index=False)
v = q("SELECT subject_id, V30 FROM k")
XV = X.merge(v, on="subject_id")
flags = [col for col in X.columns if X[col].dtype == bool]
summary = {
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "n": int(len(X)),
    "flags": {f: {"n": int(XV[f].sum()), "pct": round(100 * XV[f].mean(), 2),
                  "V30_if_present_pct": round(100 * XV.V30[XV[f]].mean(), 1) if XV[f].any() else None,
                  "V30_if_absent_pct": round(100 * XV.V30[~XV[f]].mean(), 1)} for f in flags},
    "numeric": {f: {"missing_pct": round(100 * X[f].isna().mean(), 2),
                    "median_iqr": [round(float(x), 1) for x in X[f].quantile([.5, .25, .75])]} for f in ["hr", "qrs_ms"]},
    "ed_stays_with_index_ecg": int(in_ed.sum()),
    "prosthesis_pre_index_sources": prosth_sources,
}
(OUT / "wp1b_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
