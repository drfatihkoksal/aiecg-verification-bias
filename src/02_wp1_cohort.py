"""WP1 - intended-use cohort and verification linkage (concept.md §6.2, §6.3, §7).

Score-blind: no AI score is computed or read here.
Outputs (local only, PhysioNet DUA):
  outputs/wp1_tte_d45.parquet   resting TTEs with the §6.3 D rule applied
  outputs/wp1_cohort.parquet    one row per patient (index ECG), V/D, setting, exclusion flags
  outputs/wp1_flow.csv          exclusion flow
  outputs/wp1_summary.json      verification rates, G1 gate
Technically unusable tracings (§6.2) are excluded in WP2, where the waveforms are read.
"""
import json, datetime
from pathlib import Path
import numpy as np, pandas as pd, duckdb

R = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c")
HOSP = R / "mimiciv/3.1/hosp"
ECG = R / "mimic-iv-ecg/1.0"
ED = R / "mimic-iv-ed-2.2/ed"
AFEF = Path("/home/fatih/projects/afef/outputs")
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)
WIN, WIN_LONG, THR = 30, 365, 45

c = duckdb.connect()
c.execute("SET threads=16")
c.execute("SET enable_progress_bar=false")
q = lambda s: c.execute(s).df()

# ------------------------------------------------------------------ §6.3 D rule on resting TTEs
e = q(f"""SELECT subject_id, measurement_id, echo_time, lvef_low, lvef_high, lvef_biplane, lvef_3d, lv_function_text
          FROM read_parquet('{AFEF / "echo_ef_studies.parquet"}') WHERE test_type='tte'""")
ok = lambda s: s.where(s.between(5, 95))
bp, lo, hi, d3 = ok(e.lvef_biplane), ok(e.lvef_low), ok(e.lvef_high), ok(e.lvef_3d)
lo_c, hi_c = np.fmin(lo, hi), np.fmax(lo, hi)          # closed range, order-safe
closed = lo.notna() & hi.notna()
TEXT_NEG = ["Nl (M>52; F>54%)", "Hyper (EF>75%)", "Borderline (EF50-55%)"]
TEXT_POS = ["Sev (<30%)", "Mod (EF 30-39%)", "Mod-sev (EF 25-35%)", "Mild-mod (EF 35-45%)", "Depressed"]
TEXT_MID = ["Mild (EF 40-55%)"]                         # midpoint 47.5 -> 0; straddles 45

src = np.select([bp.notna(), lo.notna(), d3.notna(),
                 e.lv_function_text.isin(TEXT_NEG + TEXT_POS + TEXT_MID)],
                ["biplane", "reported", "3d", "text"], "none")
d = np.full(len(e), np.nan)
strad = np.zeros(len(e), bool)
m = src == "biplane"; d[m] = (bp[m] <= THR)
m = src == "3d"; d[m] = (d3[m] <= THR)
r = src == "reported"
m = r & closed & (hi_c <= THR); d[m] = 1
m = r & closed & (lo_c > THR); d[m] = 0
m = r & closed & (lo_c <= THR) & (hi_c > THR); d[m] = ((lo_c[m] + hi_c[m]) / 2 <= THR); strad[m] = True
m = r & ~closed & (lo > THR); d[m] = 0
m = r & ~closed & (lo <= THR); d[m] = 1; strad[m] = True          # open-ended, lower-bound rule
t = src == "text"
d[t & e.lv_function_text.isin(TEXT_NEG)] = 0
d[t & e.lv_function_text.isin(TEXT_POS)] = 1
m = t & e.lv_function_text.isin(TEXT_MID); d[m] = 0; strad[m] = True
assert not np.isnan(d[src != "none"]).any()
e["src"], e["d"], e["straddle"] = src, d, strad
e["d_sens"] = np.where(strad, 1, d)
tte = e[["subject_id", "measurement_id", "echo_time", "src", "d", "d_sens", "straddle"]]
tte.to_parquet(OUT / "wp1_tte_d45.parquet", index=False)
c.register("tte_df", tte)
c.execute("CREATE TABLE tte AS SELECT * FROM tte_df")
tte_summary = {
    "n_resting_tte": int(len(tte)),
    "by_source": tte.src.value_counts().to_dict(),
    "d1_primary": int((tte.d == 1).sum()), "d0_primary": int((tte.d == 0).sum()),
    "straddle": int(tte.straddle.sum()),
    "straddle_closed_midpoint_d1": int((tte.straddle & (tte.d == 1) & (tte.src == "reported") & closed).sum()),
    "reported_upper_only_unused": int((lo.isna() & hi.notna() & (src != "biplane")).sum()),
}

# ------------------------------------------------------------------ index ECG and exclusion flow
c.execute(f"""
CREATE TABLE rec AS
SELECT subject_id, study_id, ecg_time,
       row_number() OVER (PARTITION BY subject_id ORDER BY ecg_time, study_id) rn
FROM read_csv_auto('{ECG / "record_list.csv"}')""")
c.execute(f"CREATE TABLE pat AS SELECT * FROM read_csv_auto('{HOSP / 'patients.csv.gz'}')")
c.execute(f"CREATE TABLE adm AS SELECT subject_id, hadm_id, CAST(admittime AS TIMESTAMP) admittime, CAST(dischtime AS TIMESTAMP) dischtime FROM read_csv_auto('{HOSP / 'admissions.csv.gz'}')")
c.execute(f"CREATE TABLE dx AS SELECT subject_id, hadm_id, icd_code, icd_version FROM read_csv_auto('{HOSP / 'diagnoses_icd.csv.gz'}', all_varchar=true)")
c.execute(f"CREATE TABLE eds AS SELECT subject_id, stay_id, CAST(intime AS TIMESTAMP) intime, CAST(outtime AS TIMESTAMP) outtime FROM read_csv_auto('{ED / 'edstays.csv.gz'}')")
c.execute(f"""CREATE TABLE rhy AS SELECT subject_id, study_id, rhythm_candidate, paced_flag, paced_subtype, report_empty
              FROM read_parquet('{AFEF / "ecg_rhythm_candidates.parquet"}')""")

c.execute("""
CREATE TABLE idx AS
SELECT r.subject_id, r.study_id, r.ecg_time,
  p.subject_id IS NOT NULL AS in_patients, p.gender,
  p.anchor_age + (year(r.ecg_time) - p.anchor_year) AS age,
  p.anchor_year_group, p.anchor_year, CAST(p.dod AS DATE) AS dod,
  h.rhythm_candidate, coalesce(h.paced_flag, false) AS paced, h.paced_subtype, h.report_empty,
  EXISTS (SELECT 1 FROM tte t WHERE t.subject_id=r.subject_id AND t.echo_time < r.ecg_time AND t.d=1) AS prior_lvsd_tte,
  EXISTS (SELECT 1 FROM dx JOIN adm USING (subject_id, hadm_id)
          WHERE dx.subject_id=r.subject_id AND adm.dischtime < r.ecg_time
            AND (icd_code LIKE 'I50%' OR icd_code LIKE '428%' OR icd_code LIKE 'I42%' OR icd_code LIKE '425%')) AS prior_hf_code
FROM rec r LEFT JOIN pat p USING (subject_id) LEFT JOIN rhy h USING (subject_id, study_id)
WHERE r.rn = 1""")

steps = [
    ("first ECG per patient (MIMIC-IV-ECG)", "TRUE"),
    ("linked to MIMIC-IV patients, sex recorded", "in_patients AND gender IS NOT NULL"),
    ("age >= 18 at index ECG", "age >= 18"),
    ("index ECG not paced", "NOT paced"),
    ("no prior TTE with LVEF <= 45", "NOT prior_lvsd_tte"),
    ("no prior HF/cardiomyopathy code (dischtime < ECG)", "NOT prior_hf_code"),
]
flow, cond = [], []
for label, s in steps:
    cond.append(f"({s})")
    n = int(q(f"SELECT count(*) n FROM idx WHERE {' AND '.join(cond)}").n[0])
    flow.append({"step": label, "n_remaining": n})
for i in range(1, len(flow)):
    flow[i]["n_excluded"] = flow[i - 1]["n_remaining"] - flow[i]["n_remaining"]
pd.DataFrame(flow).to_csv(OUT / "wp1_flow.csv", index=False)
paced_sub = q("SELECT paced_subtype, count(*) n FROM idx WHERE in_patients AND gender IS NOT NULL AND age>=18 AND paced GROUP BY 1 ORDER BY n DESC")
c.execute(f"CREATE TABLE coh AS SELECT * FROM idx WHERE {' AND '.join(cond)}")

# ------------------------------------------------------------------ setting, verification, covariate stubs
c.execute(f"""
CREATE TABLE cohort AS
SELECT k.*,
  (SELECT min(stay_id) FROM eds WHERE eds.subject_id=k.subject_id AND k.ecg_time BETWEEN intime AND outtime) AS ed_stay_id,
  (SELECT min(hadm_id) FROM adm WHERE adm.subject_id=k.subject_id AND k.ecg_time BETWEEN admittime AND dischtime) AS index_hadm_id,
  EXISTS (SELECT 1 FROM tte t WHERE t.subject_id=k.subject_id AND t.echo_time < k.ecg_time
          AND t.echo_time >= k.ecg_time - INTERVAL {WIN_LONG} DAY) AS prior_tte_365
FROM coh k""")
c.execute("""ALTER TABLE cohort ADD COLUMN setting VARCHAR;
UPDATE cohort SET setting = CASE WHEN ed_stay_id IS NOT NULL THEN 'ED'
                                 WHEN index_hadm_id IS NOT NULL THEN 'inpatient' ELSE 'outpatient' END""")


def link(win, tag):
    c.execute(f"""
    CREATE TABLE v_{tag} AS
    SELECT subject_id, measurement_id AS tte_id_{tag}, days AS days_{tag}, d AS d_{tag}, d_sens AS d_sens_{tag}, src AS src_{tag}
    FROM (SELECT k.subject_id, t.measurement_id, t.d, t.d_sens, t.src,
                 date_diff('minute', k.ecg_time, t.echo_time) / 1440.0 AS days,
                 row_number() OVER (PARTITION BY k.subject_id ORDER BY t.echo_time, t.measurement_id) rn
          FROM cohort k JOIN tte t ON t.subject_id=k.subject_id AND t.src <> 'none'
           AND t.echo_time >= k.ecg_time AND t.echo_time <= k.ecg_time + INTERVAL {win} DAY)
    WHERE rn = 1""")


link(WIN, "v30"); link(WIN_LONG, "v365")
df = q("""SELECT k.*, a.tte_id_v30, a.days_v30, a.d_v30, a.d_sens_v30, a.src_v30,
                 b.tte_id_v365, b.days_v365, b.d_v365, b.d_sens_v365, b.src_v365
          FROM cohort k LEFT JOIN v_v30 a USING (subject_id) LEFT JOIN v_v365 b USING (subject_id)
          ORDER BY subject_id""")
df["V30"] = df.tte_id_v30.notna().astype(int)
df["V365"] = df.tte_id_v365.notna().astype(int)
df["died_30d"] = (pd.to_datetime(df.dod) <= (df.ecg_time + pd.Timedelta(days=WIN)).dt.normalize()).fillna(False)
lo_y = df.anchor_year_group.str[:4].astype(int)
df["approx_year"] = lo_y + 1 + (df.ecg_time.dt.year - df.anchor_year)
df["era"] = pd.cut(df.approx_year, [0, 2010, 2013, 2016, 2019, 9999],
                   labels=["2008-2010", "2011-2013", "2014-2016", "2017-2019", "2020-2022"])
assert df.subject_id.is_unique
df.to_parquet(OUT / "wp1_cohort.parquet", index=False)

# ------------------------------------------------------------------ summary and G1
ver = df[df.V30 == 1]
by_setting = (df.groupby("setting").agg(n=("subject_id", "size"), V30=("V30", "mean"), V365=("V365", "mean"))
              .assign(V30=lambda x: (100 * x.V30).round(1), V365=lambda x: (100 * x.V365).round(1)).reset_index())
g1_cases = int((ver.d_v30 == 1).sum())
summary = {
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "tte_d_rule": tte_summary,
    "flow": flow,
    "paced_subtypes_excluded": paced_sub.to_dict("records"),
    "cohort_n": int(len(df)),
    "setting": df.setting.value_counts().to_dict(),
    "era": df.era.value_counts().sort_index().astype(int).to_dict(),
    "rhythm": df.rhythm_candidate.fillna("missing").value_counts().to_dict(),
    "verification": {
        "V30_n": int(df.V30.sum()), "V30_pct": round(100 * df.V30.mean(), 2),
        "V365_n": int(df.V365.sum()), "V365_pct": round(100 * df.V365.mean(), 2),
        "by_setting": by_setting.to_dict("records"),
        "V30_days_median_iqr": [round(float(x), 2) for x in ver.days_v30.quantile([.5, .25, .75])],
        "V30_source": ver.src_v30.value_counts().to_dict(),
        "unverified_died_within_30d": int((df.died_30d & (df.V30 == 0)).sum()),
        "prior_tte_365_pct": round(100 * df.prior_tte_365.mean(), 2),
    },
    "G1": {
        "rule": ">= 100 verified (V30) cases with LVEF <= 45 (primary D rule)",
        "verified_cases_primary": g1_cases,
        "verified_cases_sensitivity_straddle_as_1": int((ver.d_sens_v30 == 1).sum()),
        "prevalence_among_verified_pct": round(100 * (ver.d_v30 == 1).mean(), 2),
        "V365_cases_primary": int((df.d_v365 == 1).sum()),
        "passed": g1_cases >= 100,
    },
}
(OUT / "wp1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
