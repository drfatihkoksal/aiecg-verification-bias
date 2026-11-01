"""Score-blind inputs for the open design decisions (concept.md §10).

No AI score is computed or read. No reference-standard result (D) is cross-tabulated against
any ECG feature, window or subgroup. Only these are counted:
  A. how resting-TTE LVEF is coded relative to 45 (echo table only, not linked to ECGs);
  B. timing of the first TTE after each patient's first ECG (V by window and setting);
  C. prosthetic-valve evidence available before the index ECG;
  D. rhythm mix of index ECGs and V by rhythm.

The cohort here is an APPROXIMATE skeleton (first ECG per adult, paced excluded, prior LVSD/HF
excluded). It is NOT the frozen WP1 cohort.
"""
import json, datetime
from pathlib import Path
import duckdb

R = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c")
HOSP = R / "mimiciv/3.1/hosp"
ECG = R / "mimic-iv-ecg/1.0"
ED = R / "mimic-iv-ed-2.2/ed"
SM = R / "echo/echo/structured-measurement.csv.gz"
AFEF = Path("/home/fatih/projects/afef/outputs")
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

c = duckdb.connect()
c.execute("SET threads=16")
c.execute("SET enable_progress_bar=false")
q = lambda s: c.execute(s).df()
res = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
       "note": "score-blind; approximate cohort skeleton, not the frozen WP1 cohort"}

# ---------------------------------------------------------------- A. LVEF coding at 45
PL = "BETWEEN 5 AND 95"  # plausibility for numeric EF
c.execute(f"""
CREATE TABLE tte AS
SELECT subject_id, measurement_id, echo_time, lvef_low, lvef_high, lvef_biplane, lvef_3d, lv_function_text,
  CASE
    WHEN lvef_biplane {PL} THEN 'biplane'
    WHEN lvef_low {PL} THEN 'reported'
    WHEN lvef_3d {PL} THEN '3d'
    WHEN lv_function_text IS NOT NULL AND lv_function_text NOT IN ('.', 'Can''t assess') THEN 'text'
    ELSE 'none' END AS src,
  CASE
    WHEN lvef_biplane {PL} THEN (CASE WHEN lvef_biplane <= 45 THEN 'pos' ELSE 'neg' END)
    WHEN lvef_low {PL} THEN (CASE
        WHEN lvef_high IS NOT NULL AND lvef_high < 100 AND lvef_high <= 45 THEN 'pos'
        WHEN lvef_low > 45 THEN 'neg'
        ELSE 'straddle' END)
    WHEN lvef_3d {PL} THEN (CASE WHEN lvef_3d <= 45 THEN 'pos' ELSE 'neg' END)
    WHEN lv_function_text LIKE 'Nl%' OR lv_function_text LIKE 'Hyper%' OR lv_function_text LIKE 'Borderline%' THEN 'neg'
    WHEN lv_function_text LIKE 'Sev%' OR lv_function_text LIKE 'Mod (%' OR lv_function_text LIKE 'Mod-sev%'
         OR lv_function_text LIKE 'Mild-mod%' THEN 'pos'
    WHEN lv_function_text LIKE 'Mild (%' OR lv_function_text = 'Depressed' THEN 'straddle'
    ELSE 'none' END AS d45
FROM read_parquet('{AFEF / "echo_ef_studies.parquet"}')
WHERE test_type = 'tte'
""")
a = q("SELECT src, d45, count(*) n FROM tte GROUP BY ALL ORDER BY src, d45")
strad = q("""SELECT coalesce(CAST(lvef_low AS VARCHAR),'') || '-' || coalesce(CAST(lvef_high AS VARCHAR),'open') AS range,
             count(*) n FROM tte WHERE src='reported' AND d45='straddle' GROUP BY 1 ORDER BY n DESC LIMIT 8""")
tot = q("SELECT d45, count(*) n FROM tte GROUP BY 1").set_index("d45").n.to_dict()
res["A_lvef_coding_resting_tte"] = {
    "n_tte": int(sum(tot.values())),
    "by_d45": {k: int(v) for k, v in tot.items()},
    "by_source_and_d45": a.to_dict("records"),
    "top_straddling_reported_ranges": strad.to_dict("records"),
    "straddle_vs_definite_pos": round(tot.get("straddle", 0) / tot.get("pos", 1), 3),
}

# ---------------------------------------------------------------- cohort skeleton
c.execute(f"""
CREATE TABLE first_ecg AS
SELECT r.subject_id, r.study_id, r.ecg_time,
       p.anchor_age + (year(r.ecg_time) - p.anchor_year) AS age
FROM (SELECT subject_id, study_id, ecg_time,
             row_number() OVER (PARTITION BY subject_id ORDER BY ecg_time, study_id) rn
      FROM read_csv_auto('{ECG / "record_list.csv"}')) r
JOIN read_csv_auto('{HOSP / "patients.csv.gz"}') p USING (subject_id)
WHERE r.rn = 1
""")
c.execute(f"CREATE TABLE adm AS SELECT subject_id, hadm_id, admittime, dischtime FROM read_csv_auto('{HOSP / 'admissions.csv.gz'}')")
c.execute(f"CREATE TABLE dx AS SELECT subject_id, hadm_id, icd_code, icd_version FROM read_csv_auto('{HOSP / 'diagnoses_icd.csv.gz'}', all_varchar=true)")
c.execute(f"CREATE TABLE px AS SELECT subject_id, hadm_id, chartdate, icd_code, icd_version FROM read_csv_auto('{HOSP / 'procedures_icd.csv.gz'}', all_varchar=true)")
c.execute(f"""CREATE TABLE rhy AS SELECT subject_id, study_id, rhythm_candidate, paced_flag, paced_subtype
              FROM read_parquet('{AFEF / "ecg_rhythm_candidates.parquet"}')""")

c.execute("""
CREATE TABLE coh0 AS
SELECT f.*, r.rhythm_candidate, r.paced_flag, r.paced_subtype,
  EXISTS (SELECT 1 FROM tte t WHERE t.subject_id=f.subject_id AND t.echo_time < f.ecg_time AND t.d45='pos') AS prior_lvsd_tte,
  EXISTS (SELECT 1 FROM dx JOIN adm USING (subject_id, hadm_id)
          WHERE dx.subject_id=f.subject_id AND CAST(adm.dischtime AS TIMESTAMP) < f.ecg_time
            AND (icd_code LIKE 'I50%' OR icd_code LIKE '428%' OR icd_code LIKE 'I42%' OR icd_code LIKE '425%')) AS prior_hf_code
FROM first_ecg f LEFT JOIN rhy r USING (subject_id, study_id)
""")
flow = q("""SELECT count(*) all_first_ecg,
  sum(CASE WHEN age >= 18 THEN 1 ELSE 0 END) adult,
  sum(CASE WHEN age >= 18 AND paced_flag THEN 1 ELSE 0 END) adult_paced,
  sum(CASE WHEN age >= 18 AND NOT coalesce(paced_flag,false) AND (prior_lvsd_tte OR prior_hf_code) THEN 1 ELSE 0 END) adult_nonpaced_prior_lvsd_hf
  FROM coh0""").iloc[0].to_dict()
paced_sub = q("SELECT paced_subtype, count(*) n FROM coh0 WHERE age>=18 AND paced_flag GROUP BY 1 ORDER BY n DESC")
c.execute("CREATE TABLE coh AS SELECT * FROM coh0 WHERE age >= 18 AND NOT coalesce(paced_flag,false) AND NOT prior_lvsd_tte AND NOT prior_hf_code")

# setting at ECG
c.execute(f"""
CREATE TABLE coh_s AS
SELECT k.*,
  CASE WHEN EXISTS (SELECT 1 FROM adm a WHERE a.subject_id=k.subject_id
                    AND k.ecg_time BETWEEN CAST(a.admittime AS TIMESTAMP) AND CAST(a.dischtime AS TIMESTAMP)) THEN 'inpatient'
       WHEN EXISTS (SELECT 1 FROM read_csv_auto('{ED / "edstays.csv.gz"}') e WHERE e.subject_id=k.subject_id
                    AND k.ecg_time BETWEEN CAST(e.intime AS TIMESTAMP) AND CAST(e.outtime AS TIMESTAMP)) THEN 'ED'
       ELSE 'outpatient' END AS setting
FROM coh k
""")
n_coh = int(q("SELECT count(*) n FROM coh_s").n[0])
flow["cohort_skeleton"] = n_coh
res["cohort_flow"] = {k: int(v) for k, v in flow.items()}
res["paced_subtypes_adult_first_ecg"] = paced_sub.to_dict("records")

# ---------------------------------------------------------------- B. timing of first TTE after index ECG
c.execute("""
CREATE TABLE nxt AS
SELECT k.subject_id, k.setting, k.rhythm_candidate,
  min(date_diff('minute', k.ecg_time, t.echo_time)) / 1440.0 AS days_any,
  min(CASE WHEN t.d45 <> 'none' THEN date_diff('minute', k.ecg_time, t.echo_time) END) / 1440.0 AS days_ef
FROM coh_s k LEFT JOIN tte t ON t.subject_id=k.subject_id AND t.echo_time >= k.ecg_time
GROUP BY ALL
""")
V = lambda w, col="days_ef": f"avg(CASE WHEN {col} <= {w} THEN 1.0 ELSE 0 END)"
timing = q(f"""SELECT setting, count(*) n,
  round(100*{V(1)},1) v1, round(100*{V(7)},1) v7, round(100*{V(14)},1) v14, round(100*{V(30)},1) v30,
  round(100*{V(90)},1) v90, round(100*{V(365)},1) v365,
  round(100*({V(30)}-{V(14)})/nullif({V(30)},0),1) pct_of_v30_after_day14
  FROM nxt GROUP BY ROLLUP(setting) ORDER BY setting NULLS LAST""")
noef = q(f"""SELECT round(100*{V(30,'days_any')},2) any_tte_30, round(100*{V(30)},2) tte_with_ef_info_30 FROM nxt""").iloc[0].to_dict()
res["B_verification_timing_pct"] = timing.to_dict("records")
res["B_any_tte_vs_tte_with_ef_info_30d_pct"] = noef

# prior TTE shortly before the index ECG (not an exclusion; information only)
prior = q("""SELECT
  round(100*avg(CASE WHEN EXISTS (SELECT 1 FROM tte t WHERE t.subject_id=k.subject_id AND t.echo_time < k.ecg_time
        AND t.echo_time >= k.ecg_time - INTERVAL 30 DAY AND t.d45 <> 'none') THEN 1.0 ELSE 0 END),2) prior_tte_30d_pct,
  round(100*avg(CASE WHEN EXISTS (SELECT 1 FROM tte t WHERE t.subject_id=k.subject_id AND t.echo_time < k.ecg_time
        AND t.echo_time >= k.ecg_time - INTERVAL 365 DAY AND t.d45 <> 'none') THEN 1.0 ELSE 0 END),2) prior_tte_365d_pct
  FROM coh_s k""").iloc[0].to_dict()
res["B_prior_tte_before_index_pct"] = prior

# ---------------------------------------------------------------- C. prosthetic valves (pre-index evidence only)
DX = "(icd_version='10' AND icd_code IN ('Z952','Z953','Z954')) OR (icd_version='9' AND icd_code IN ('V433','V422'))"
PX = ("(icd_version='9' AND (icd_code LIKE '352%' OR icd_code IN ('3505','3506','3507','3508'))) "
      "OR (icd_version='10' AND (icd_code LIKE '02RF%' OR icd_code LIKE '02RG%' OR icd_code LIKE '02RH%' OR icd_code LIKE '02RJ%'))")
dxd = q(f"""SELECT d.icd_version, d.icd_code, any_value(d.long_title) title, count(*) n
            FROM dx JOIN read_csv_auto('{HOSP / "d_icd_diagnoses.csv.gz"}', all_varchar=true) d USING (icd_code, icd_version)
            WHERE {DX.replace('icd_version', 'dx.icd_version').replace('icd_code', 'dx.icd_code')} GROUP BY ALL ORDER BY n DESC""")
pxd = q(f"""SELECT left(px.icd_code,4) stem, px.icd_version, any_value(d.long_title) example_title, count(*) n
            FROM px JOIN read_csv_auto('{HOSP / "d_icd_procedures.csv.gz"}', all_varchar=true) d USING (icd_code, icd_version)
            WHERE {PX.replace('icd_version', 'px.icd_version').replace('icd_code', 'px.icd_code')} GROUP BY ALL ORDER BY n DESC""")
c.execute(f"""
CREATE TABLE echo_prosth AS
SELECT DISTINCT CAST(subject_id AS BIGINT) subject_id, CAST(measurement_datetime AS TIMESTAMP) echo_time,
  measurement || ': ' || result AS evidence
FROM read_csv_auto('{SM}', all_varchar=true)
WHERE test_type='tte' AND result IS NOT NULL AND (
  (measurement='av_leaflets' AND result IN ('Bioprosthesis','Bileaflet mechanical','Mechanical','AVR homograft',
                                            'Other prosthesis','Single tilting disk','Homograft','SIngle tilting disc'))
  OR (measurement='mv_leaflets' AND result IN ('Bioprosthesis','Mechanical','SIngle tilting disc'))
  OR (measurement='tv_leaflets' AND result IN ('Bioprosthetic','Other prosthesis','Mechanical'))
  OR measurement IN ('avr_structure1','pvr_structure1')
  OR (measurement IN ('mvr_structure1','tvr_structure1') AND result ILIKE 'Prosthesis%'))
""")
pv = q(f"""SELECT count(*) n,
  sum(CASE WHEN p_dx OR p_px THEN 1 ELSE 0 END) prior_code,
  sum(CASE WHEN idx_status THEN 1 ELSE 0 END) index_admission_status_code,
  sum(CASE WHEN p_echo THEN 1 ELSE 0 END) prior_tte,
  sum(CASE WHEN p_dx OR p_px OR idx_status OR p_echo THEN 1 ELSE 0 END) any_pre_index,
  sum(CASE WHEN post_echo30 AND NOT (p_dx OR p_px OR idx_status OR p_echo) THEN 1 ELSE 0 END) only_on_verifying_tte_30d,
  sum(CASE WHEN post_echo30 AND NOT (p_dx OR p_px OR idx_status OR p_echo) AND idx_px THEN 1 ELSE 0 END) of_which_valve_surgery_in_index_admission,
  sum(CASE WHEN idx_px AND NOT (p_dx OR p_px OR idx_status OR p_echo) THEN 1 ELSE 0 END) valve_surgery_in_index_admission_no_pre_index
FROM (SELECT k.subject_id,
  EXISTS (SELECT 1 FROM px JOIN adm USING (subject_id, hadm_id) WHERE px.subject_id=k.subject_id
          AND k.ecg_time BETWEEN CAST(adm.admittime AS TIMESTAMP) AND CAST(adm.dischtime AS TIMESTAMP)
          AND CAST(px.chartdate AS DATE) >= CAST(k.ecg_time AS DATE) AND ({PX})) idx_px,
  EXISTS (SELECT 1 FROM dx JOIN adm USING (subject_id, hadm_id) WHERE dx.subject_id=k.subject_id
          AND CAST(adm.dischtime AS TIMESTAMP) < k.ecg_time AND ({DX})) p_dx,
  EXISTS (SELECT 1 FROM px WHERE px.subject_id=k.subject_id AND CAST(px.chartdate AS DATE) < CAST(k.ecg_time AS DATE) AND ({PX})) p_px,
  EXISTS (SELECT 1 FROM dx JOIN adm USING (subject_id, hadm_id) WHERE dx.subject_id=k.subject_id
          AND k.ecg_time BETWEEN CAST(adm.admittime AS TIMESTAMP) AND CAST(adm.dischtime AS TIMESTAMP) AND ({DX})) idx_status,
  EXISTS (SELECT 1 FROM echo_prosth e WHERE e.subject_id=k.subject_id AND e.echo_time < k.ecg_time) p_echo,
  EXISTS (SELECT 1 FROM echo_prosth e WHERE e.subject_id=k.subject_id AND e.echo_time >= k.ecg_time
          AND e.echo_time <= k.ecg_time + INTERVAL 30 DAY) post_echo30
  FROM coh_s k)""").iloc[0].to_dict()
res["C_prosthetic_valve_in_cohort_skeleton"] = {k: int(v) for k, v in pv.items()}
res["C_codes_seen_dx"] = dxd.to_dict("records")
res["C_codes_seen_px"] = pxd.to_dict("records")

# ---------------------------------------------------------------- D. rhythm mix and V by rhythm
rh = q(f"""SELECT coalesce(rhythm_candidate,'missing') rhythm, count(*) n,
  round(100.0*count(*)/sum(count(*)) OVER (),1) pct, round(100*{V(30)},1) v30
  FROM nxt GROUP BY 1 ORDER BY n DESC""")
res["D_rhythm_index_ecg"] = rh.to_dict("records")

(OUT / "decision_inputs.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps(res, indent=2, default=str))
