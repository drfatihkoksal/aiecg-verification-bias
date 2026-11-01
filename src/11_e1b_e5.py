"""EXPLORATORY (post hoc, 1 Nov 2026; requested after reading Otabor et al. 2026, doi:10.1093/ehjdh/ztag147).
The pre-specified primary analysis (src/07) is unchanged.

E1b  Replication of the Otabor et al. MIMIC-IV validation design, as described in their Methods and Figure 1:
     adults whose ECG preceded an eligible resting TTE by up to 365 days; eligible TTE = numeric LVEF reported
     (biplane, reported range or 3D) + at least one valve finding, and no repaired/replaced valve; ECG excluded for
     ventricular pacing, poor quality, lead reversal or invalid measurements; one ECG per patient = the most recent
     eligible ECG (ties: higher study_id), labelled by the first eligible TTE within the following 365 days.
     Operationalisation (ours, approximations of their text): valve finding = any non-missing aortic/mitral/
     tricuspid/pulmonic stenosis, regurgitation or leaflet field; repaired/replaced = prosthesis, ring, clip or
     AVR/MVR/TVR/PVR structure fields; ventricular pacing = afef paced_subtype ventricular or unspecified; poor
     quality = machine statements on data quality / unsuitable leads / technically unsatisfactory / external noise,
     or the WP2 flat/missing-lead rule; lead reversal = machine statements on arm/limb/electrode reversal; invalid
     measurements = afef tabular ventricular rate, QRS or QTc outside plausibility gates. LVEF <= 45 by our primary
     D rule (Otabor's own LVEF field rule is in their supplement, not available here).
E5   Atrial-rate rule: every MIMIC ECG scored with atrial_rate = 0 (the released preprocessing rule applied
     literally, as in Otabor et al.) instead of the afef reconstruction (= ventricular rate if P waves detected).
     Applied to (a) the intended-use analytic cohort: naive, IPW and delta re-estimated with the alternative score
     (verification model refitted on it; frozen threshold unchanged) and (b) the E1b cohort.
Both scores come from one waveform pass; only the tabular input differs. The primary-rule logits are asserted to
reproduce WP2. Bootstrap: 1000 patient resamples, seed 20261103.
"""
import sys, json, hashlib, importlib, datetime, warnings, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, duckdb
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=UserWarning)
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import wp3_design as W  # noqa: E402

OUT = W.OUT
R = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c")
AFEF = Path("/home/fatih/projects/afef")
B, SEED, CHUNK = 1000, 20261103, 8192


def ci(v):
    return np.percentile(v, [2.5, 97.5]).round(4).tolist()


def score_two_rules(pairs):
    """Return logits under the primary tabular rule and under atrial_rate = 0, plus technical exclusion."""
    import torch
    wp2 = importlib.import_module("05_wp2_score")
    tabmod = importlib.import_module("08_mimic_tabular")
    from cradlenet.models.resnet1d_tabular import ResNet1dWithTabular
    tab_path = AFEF / "outputs/ecg_tabular_features.parquet"
    assert hashlib.sha256(tab_path.read_bytes()).hexdigest() == wp2.TAB_SHA
    rl = pd.read_csv(wp2.ECG / "record_list.csv", usecols=["subject_id", "study_id", "path"])
    df = (pairs[["subject_id", "study_id"]].merge(rl, on=["subject_id", "study_id"])
          .merge(pd.read_parquet(tab_path), on=["subject_id", "study_id"], suffixes=("", "_tab")).reset_index(drop=True))
    assert len(df) == len(pairs)
    T1 = tabmod.transform(df)
    dz = df.copy(); dz["atrial_rate"] = 0.0
    T0 = tabmod.transform(dz)
    pool = ProcessPoolExecutor(max_workers=28, mp_context=mp.get_context("spawn"))
    list(pool.map(abs, range(28)))
    sd = torch.load(AFEF / "models/echonext_minimodel/weights.pt", map_location="cpu", weights_only=False)["model"]
    model = ResNet1dWithTabular(len_tabular_feature_vector=7, filter_size=16, num_classes=12)
    model.load_state_dict(sd, strict=True); model.eval().cuda()
    l1 = np.full(len(df), np.nan); l0 = np.full(len(df), np.nan); tech = np.zeros(len(df), bool)
    for a in range(0, len(df), CHUNK):
        sl = slice(a, min(a + CHUNK, len(df)))
        res = list(pool.map(wp2.work, df.path.iloc[sl].tolist(), chunksize=16))
        ok = np.array([r[0] is not None for r in res])
        X = np.zeros((len(res), 1, 2500, 12), np.float32)
        for i, (x, _) in enumerate(res):
            if x is not None:
                X[i, 0] = x
        q = pd.DataFrame([r[1] for r in res])
        tech[sl] = ~ok | (q.get("flat_leads", pd.Series(0, index=q.index)).fillna(0).to_numpy() > 0) | \
            (q.get("nan_leads", pd.Series(0, index=q.index)).fillna(0).to_numpy() > 0)
        with torch.no_grad():
            for tt, out in [(T1, l1), (T0, l0)]:
                for b in range(0, len(res), 512):
                    xb = torch.from_numpy(X[b:b + 512]).cuda()
                    tb = torch.from_numpy(tt[sl][b:b + 512]).cuda()
                    out[a + b:a + b + len(xb)] = model((xb, tb))[:, 0].float().cpu().numpy()
        l1[sl][~ok] = np.nan; l0[sl][~ok] = np.nan
        print(f"  scored {sl.stop:,}/{len(df):,}", flush=True)
    pool.shutdown()
    df["logit_primary"], df["logit_ar0"], df["technical_exclusion"] = l1, l0, tech
    return df[["subject_id", "study_id", "logit_primary", "logit_ar0", "technical_exclusion"]]


def simple_metrics(y, s, thr, rng_seed):
    """Metrics in a fully verified cohort (no weighting), with patient bootstrap CIs."""
    T = (1 / (1 + np.exp(-s)) >= thr).astype(int)
    m = W.metrics(T, y, np.ones(len(y)), s, 0)
    rng = np.random.default_rng(rng_seed)
    bs = [W.metrics(T[i], y[i], np.ones(len(i)), s[i], 0) for i in (rng.integers(0, len(y), len(y)) for _ in range(B))]
    return {k: {"est": round(m[k], 4), "ci": ci([b[k] for b in bs])} for k in ["sens", "spec", "ppv", "npv", "prev", "auroc"]} | \
           {"pct_test_positive": round(100 * T.mean(), 4)}


def build_e1b_pairs():
    """E1b (Otabor design) pairs, ordered by subject_id; also used by src/12."""
    c = duckdb.connect(); c.execute("SET threads=24"); c.execute("SET enable_progress_bar=false")
    SM = R / "echo/echo/structured-measurement.csv.gz"
    MM = R / "mimic-iv-ecg/1.0/machine_measurements.csv"
    cols = ", ".join(f"report_{i}" for i in range(18))

    # ---------------- E1b cohort
    c.execute(f"""
    CREATE TABLE echo_flags AS
    SELECT CAST(measurement_id AS BIGINT) measurement_id,
      max(CASE WHEN measurement IN ('aortic_stenosis','aortic_regurg','mitral_regurg','tricuspid_regurg','pulm_regurg',
                                    'mitral_stenosis','pulm_stenosis','av_leaflets','mv_leaflets','tv_leaflets','pv_leaflets')
               AND result IS NOT NULL THEN 1 ELSE 0 END) valve_finding,
      max(CASE WHEN (measurement='av_leaflets' AND result IN ('Bioprosthesis','Bileaflet mechanical','Mechanical','AVR homograft',
                                                             'Other prosthesis','Single tilting disk','Homograft','SIngle tilting disc'))
                 OR (measurement='mv_leaflets' AND result IN ('Bioprosthesis','Mechanical','SIngle tilting disc','Annular ring','Bileaflet'))
                 OR (measurement='tv_leaflets' AND result IN ('Annuloplasty ring','Bioprosthetic','Other prosthesis','Mechanical','Bileaflet'))
                 OR (measurement IN ('avr_structure1','mvr_structure1','tvr_structure1','pvr_structure1') AND result IS NOT NULL)
               THEN 1 ELSE 0 END) repaired_replaced
    FROM read_csv_auto('{SM}', all_varchar=true) WHERE test_type='tte' GROUP BY 1""")
    c.execute(f"""
    CREATE TABLE tte_ok AS
    SELECT t.subject_id, t.measurement_id, t.echo_time, t.d FROM '{OUT / "wp1_tte_d45.parquet"}' t
    JOIN echo_flags f USING (measurement_id)
    WHERE t.src IN ('biplane','reported','3d') AND f.valve_finding=1 AND f.repaired_replaced=0""")
    c.execute(f"""
    CREATE TABLE ecg_ok AS
    WITH st AS (SELECT CAST(study_id AS BIGINT) study_id, lower(concat_ws(' | ', {cols})) s FROM read_csv_auto('{MM}', all_varchar=true))
    SELECT r.subject_id, r.study_id, r.ecg_time
    FROM read_csv_auto('{R / "mimic-iv-ecg/1.0/record_list.csv"}') r
    JOIN read_csv_auto('{R / "mimiciv/3.1/hosp/patients.csv.gz"}') p USING (subject_id)
    JOIN '{AFEF / "outputs/ecg_tabular_features.parquet"}' tb USING (subject_id, study_id)
    LEFT JOIN '{AFEF / "outputs/ecg_rhythm_candidates.parquet"}' h USING (subject_id, study_id)
    JOIN st USING (study_id)
    WHERE p.gender IS NOT NULL AND p.anchor_age + (year(r.ecg_time) - p.anchor_year) >= 18
      AND coalesce(h.paced_subtype, '') NOT IN ('ventricular_paced','paced_unspecified')
      AND NOT regexp_matches(st.s, 'lead reversal|electrode reversal|arm lead revers|limb lead revers')
      AND NOT regexp_matches(st.s, 'data quality|unsuitable for analysis|technically unsatisfactory|technically poor|external noise')
      AND tb.ventricular_rate IS NOT NULL AND tb.qrs_duration IS NOT NULL AND tb.qt_corrected IS NOT NULL""")
    pairs = c.execute("""
    WITH m AS (SELECT e.subject_id, e.study_id, e.ecg_time, t.measurement_id, t.d,
                      date_diff('minute', e.ecg_time, t.echo_time) / 1440.0 AS days,
                      row_number() OVER (PARTITION BY e.study_id ORDER BY t.echo_time, t.measurement_id) rn
               FROM ecg_ok e JOIN tte_ok t ON t.subject_id=e.subject_id
                AND t.echo_time >= e.ecg_time AND t.echo_time <= e.ecg_time + INTERVAL 365 DAY),
    f AS (SELECT *, row_number() OVER (PARTITION BY subject_id ORDER BY ecg_time DESC, study_id DESC) rp FROM m WHERE rn=1)
    SELECT subject_id, study_id, measurement_id, d, days FROM f WHERE rp=1 ORDER BY subject_id""").df()
    flow = {"eligible_tte": int(c.execute("SELECT count(*) FROM tte_ok").fetchone()[0]),
                       "eligible_ecg": int(c.execute("SELECT count(*) FROM ecg_ok").fetchone()[0]),
                       "patients_paired": int(len(pairs)), "median_days_ecg_to_tte": round(float(pairs.days.median()), 4)}
    return pairs, flow


def main():
    thr = float(json.loads((OUT / "source_threshold.json").read_text())["lvef_lte_45"]["cutoff_prob"])
    res = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"), "status": "EXPLORATORY (post hoc)", "threshold": thr}
    pairs, res["E1b_flow"] = build_e1b_pairs()

    # ---------------- scoring: E1b pairs + intended-use index ECGs, one pass
    d = W.load()
    to_score = pd.concat([pairs[["subject_id", "study_id"]], d[["subject_id", "study_id"]]]).drop_duplicates().reset_index(drop=True)
    cache = OUT / "wp3_e1b_e5_scores.parquet"  # patient-level, local only
    if cache.exists() and len(pd.read_parquet(cache).merge(to_score, on=["subject_id", "study_id"])) == len(to_score):
        sc = pd.read_parquet(cache)
    else:
        print(f"scoring {len(to_score):,} ECGs", flush=True)
        sc = score_two_rules(to_score)
        sc.to_parquet(cache, index=False)

    # guard: primary rule reproduces WP2
    chk = d[["subject_id", "study_id", "logit"]].merge(sc, on=["subject_id", "study_id"])
    maxdiff = float(np.nanmax(np.abs(chk.logit - chk.logit_primary)))
    assert maxdiff < 1e-4, maxdiff
    res["primary_rule_reproduces_wp2_max_abs_logit_diff"] = maxdiff

    # ---------------- E1b metrics, both rules
    e = pairs.merge(sc, on=["subject_id", "study_id"])
    e = e[~e.technical_exclusion].reset_index(drop=True)
    y = e.d.to_numpy(float)
    res["E1b_flow"]["after_technical_exclusion"] = int(len(e))
    res["E1b_primary_rule"] = simple_metrics(y, e.logit_primary.to_numpy(float), thr, SEED)
    res["E1b_atrial_rate_0"] = simple_metrics(y, e.logit_ar0.to_numpy(float), thr, SEED)
    res["E1b_otabor_reported"] = {"n": 45878, "prev": 16.0, "auroc": [83.5, 83.0, 83.9], "median_days": 2}
    # aggregate ROC points for Figure 1B (no patient-level data)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y.astype(int), e.logit_primary.to_numpy(float))
    ii = np.unique(np.linspace(0, len(fpr) - 1, 150).astype(int))
    m1 = res["E1b_primary_rule"]
    (OUT / "fig1_e1b_roc.json").write_text(json.dumps({
        "roc": {"fpr": fpr[ii].round(4).tolist(), "tpr": tpr[ii].round(4).tolist()},
        "operating_point": {"fpr": round(1 - m1["spec"]["est"] / 100, 4), "tpr": round(m1["sens"]["est"] / 100, 4)},
        "auroc": round(m1["auroc"]["est"], 1), "n": int(len(e))}, indent=1))

    # ---------------- E5 intended-use cohort with atrial_rate = 0
    d0 = d.merge(sc[["subject_id", "study_id", "logit_ar0"]], on=["subject_id", "study_id"], validate="1:1")
    assert len(d0) == len(d) and (d0.subject_id.to_numpy() == d.subject_id.to_numpy()).all()
    d0["logit"] = d0.logit_ar0
    d0["positive"] = (1 / (1 + np.exp(-d0.logit)) >= thr).astype(int)
    res["E5_score_shift"] = {"mean_logit_change": round(float((d0.logit_ar0 - d.logit).mean()), 4),
                             "pct_positive_primary": round(100 * d.positive.mean(), 4),
                             "pct_positive_ar0": round(100 * d0.positive.mean(), 4),
                             "pct_test_result_changed": round(100 * (d0.positive != d.positive).mean(), 4)}
    X0 = W.design(d0)
    T, V, D, S = d0.positive.to_numpy(int), d0.V30.to_numpy(int), d0.d_v30.to_numpy(float), d0.logit.to_numpy(float)
    N = len(d0)

    def run(idx):
        v, t, dd, s_ = V[idx], T[idx], D[idx], S[idx]
        vv = v == 1
        nv = W.metrics(t[vv], dd[vv], np.ones(vv.sum()), s_[vv], 0)
        w = W.truncate(1 / W.fit_predict(X0[idx], v, X0[idx][vv]), 0.99)
        ip = W.metrics(t[vv], dd[vv], w, s_[vv], 1)
        out = {f"naive_{k}": nv[k] for k in ["sens", "spec", "ppv", "auroc"]}
        out |= {f"ipw_{k}": ip[k] for k in ["sens", "spec", "ppv", "auroc"]}
        out |= {f"delta_{k}": nv[k] - ip[k] for k in ["sens", "spec", "ppv", "auroc"]}
        return out

    pt = run(np.arange(N))
    bs = Parallel(n_jobs=30)(delayed(run)(np.random.default_rng([SEED, b]).integers(0, N, N)) for b in range(B))
    res["E5_intended_use_atrial_rate_0"] = {k: {"est": round(v, 4), "ci": ci([b[k] for b in bs])} for k, v in pt.items()}
    (OUT / "wp3_e1b_e5.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
