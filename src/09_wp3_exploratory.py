"""WP3 EXPLORATORY analyses (post hoc). Decided on 1 Nov 2026 after the WP3 results were seen. The pre-specified primary analysis (src/07) is
unchanged.

E1  Conventional echo-anchored validation, as published AI-ECG validations do it (e.g. ECG and TTE within 30 days
    of each other): every resting TTE with LVEF information paired with the nearest non-paced adult ECG within
    +-30 days; one pair per patient (the patient's first such TTE); prior LVSD/HF NOT excluded; same model,
    pipeline, technical exclusion, frozen threshold and primary D rule. Paired ECGs are scored here with the
    unchanged WP2 pipeline.
E2  Tipping points of the MNAR analysis: lambda at which |naive - MNAR| = 5 points for sensitivity and for
    specificity (full-data point estimates), with the implied LVSD prevalence among unverified patients.
E3  Verification model + NT-proBNP and troponin (labs drawn 24 h before to 6 h after the index ECG): measured
    indicators, log10 of the maximum NT-proBNP (median-filled when not measured), troponin T >= 0.01 ng/mL.
    IPW re-estimated; same outcome, threshold, truncation.
E4  IPW weight-truncation sensitivity: none, 95th, 99th (primary), 99.5th percentile.
Bootstrap: 1000 patient resamples (seed 20261102); models refitted in every resample.
Also writes aggregate figure data (outputs/fig1_data.json). No patient-level data in that file.
"""
import sys, json, hashlib, importlib, datetime, warnings, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, duckdb
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, roc_curve
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=UserWarning)
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import wp3_design as W  # noqa: E402

OUT = W.OUT
R = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c")
B, SEED, MATERIAL = 1000, 20261102, 5.0


def ci(vals):
    return np.percentile(vals, [2.5, 97.5]).round(2).tolist()


def est(v, c):
    return {"est": round(float(v), 2), "ci": c}


def score_ecgs(pairs):
    """Score ECGs with the unchanged WP2 pipeline (src/05 work() + afef tabular transform + frozen weights)."""
    import torch
    wp2 = importlib.import_module("05_wp2_score")
    tabmod = importlib.import_module("08_mimic_tabular")
    from cradlenet.models.resnet1d_tabular import ResNet1dWithTabular
    tab_path = wp2.AFEF / "outputs/ecg_tabular_features.parquet"
    assert hashlib.sha256(tab_path.read_bytes()).hexdigest() == wp2.TAB_SHA
    rl = pd.read_csv(wp2.ECG / "record_list.csv", usecols=["subject_id", "study_id", "path"])
    df = pairs.merge(rl, on=["subject_id", "study_id"]).merge(pd.read_parquet(tab_path), on=["subject_id", "study_id"], suffixes=("", "_tab"))
    assert len(df) == len(pairs)
    Tt = tabmod.transform(df)
    pool = ProcessPoolExecutor(max_workers=28, mp_context=mp.get_context("spawn"))
    list(pool.map(abs, range(28)))
    res = list(pool.map(wp2.work, df.path.tolist(), chunksize=16))
    pool.shutdown()
    sd = torch.load(wp2.AFEF / "models/echonext_minimodel/weights.pt", map_location="cpu", weights_only=False)["model"]
    model = ResNet1dWithTabular(len_tabular_feature_vector=7, filter_size=16, num_classes=12)
    model.load_state_dict(sd, strict=True); model.eval().cuda()
    ok = np.array([r[0] is not None for r in res])
    X = np.zeros((len(df), 1, 2500, 12), np.float32)
    for i, (x, _) in enumerate(res):
        if x is not None:
            X[i, 0] = x
    lg = np.full(len(df), np.nan, np.float32)
    with torch.no_grad():
        for b in range(0, len(df), 512):
            lg[b:b + 512] = model((torch.from_numpy(X[b:b + 512]).cuda(), torch.from_numpy(Tt[b:b + 512]).cuda()))[:, 0].float().cpu().numpy()
    q = pd.DataFrame([r[1] for r in res])
    df["logit"] = np.where(ok, lg, np.nan)
    df["score"] = 1 / (1 + np.exp(-df.logit.astype(float)))
    df["technical_exclusion"] = ~ok | (q.flat_leads.fillna(0).to_numpy() > 0) | (q.nan_leads.fillna(0).to_numpy() > 0)
    return df


def main():
    d = W.load()
    thr = float(d.threshold.iloc[0])
    T = d.positive.astype(int).to_numpy()
    V = d.V30.to_numpy(int)
    D = d.d_v30.to_numpy(float)
    sc = d.logit.to_numpy(float)
    X0 = W.design(d)
    ver = V == 1
    N = len(d)

    # ---------- reproduce 07 point estimates (guard)
    p0 = W.fit_predict(X0, V, X0)
    w0 = 1 / p0[ver]
    naive = W.metrics(T[ver], D[ver], np.ones(ver.sum()), sc[ver], 0)
    ipw = W.metrics(T[ver], D[ver], W.truncate(w0, 0.99), sc[ver], 1)
    r07 = json.loads((OUT / "wp3_results.json").read_text())["main"]
    for m in ["sens", "spec", "ppv", "auroc"]:
        assert abs(naive[m] - r07["naive"][m]["est"]) < 0.01 and abs(ipw[m] - r07["ipw"][m]["est"]) < 0.01, m
    res = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"), "status": "EXPLORATORY (post hoc)",
           "reproduces_07_point_estimates": True}

    # ---------- E1 conventional echo-anchored validation
    c = duckdb.connect(); c.execute("SET threads=16"); c.execute("SET enable_progress_bar=false")
    pairs = c.execute(f"""
    WITH tte AS (SELECT subject_id, measurement_id, echo_time, d FROM '{OUT / "wp1_tte_d45.parquet"}' WHERE src <> 'none'),
    ecg AS (SELECT r.subject_id, r.study_id, r.ecg_time FROM read_csv_auto('{R / "mimic-iv-ecg/1.0/record_list.csv"}') r
            JOIN read_csv_auto('{R / "mimiciv/3.1/hosp/patients.csv.gz"}') p USING (subject_id)
            LEFT JOIN '/home/fatih/projects/afef/outputs/ecg_rhythm_candidates.parquet' h USING (subject_id, study_id)
            WHERE NOT coalesce(h.paced_flag, false) AND p.gender IS NOT NULL
              AND p.anchor_age + (year(r.ecg_time) - p.anchor_year) >= 18),
    m AS (SELECT t.subject_id, t.measurement_id, t.echo_time, t.d, e.study_id,
                 row_number() OVER (PARTITION BY t.measurement_id ORDER BY abs(date_diff('minute', e.ecg_time, t.echo_time)), e.ecg_time, e.study_id) rn
          FROM tte t JOIN ecg e ON e.subject_id=t.subject_id AND abs(date_diff('minute', e.ecg_time, t.echo_time)) <= 30*1440),
    f AS (SELECT *, row_number() OVER (PARTITION BY subject_id ORDER BY echo_time, measurement_id) rp FROM m WHERE rn=1)
    SELECT subject_id, study_id, measurement_id, d FROM f WHERE rp=1 ORDER BY subject_id""").df()
    sc_conv = score_ecgs(pairs[["subject_id", "study_id"]])
    conv = pairs.merge(sc_conv[["subject_id", "study_id", "logit", "score", "technical_exclusion"]], on=["subject_id", "study_id"])
    conv = conv[~conv.technical_exclusion].reset_index(drop=True)
    Tc, Dc, Sc = (conv.score >= thr).astype(int).to_numpy(), conv.d.to_numpy(float), conv.logit.to_numpy(float)
    in_cohort = conv.subject_id.isin(d.subject_id)
    mc = W.metrics(Tc, Dc, np.ones(len(conv)), Sc, 0)
    rng = np.random.default_rng(SEED)
    bc = [W.metrics(Tc[i], Dc[i], np.ones(len(i)), Sc[i], 0) for i in (rng.integers(0, len(conv), len(conv)) for _ in range(B))]
    res["E1_conventional"] = {
        "design": "resting TTE with LVEF info + nearest non-paced adult ECG within +-30 d; first such TTE per patient; prior LVSD/HF not excluded",
        "n_pairs": int(len(pairs)), "technical_exclusion": int(sc_conv.technical_exclusion.sum()), "n": int(len(conv)),
        "pct_patients_also_in_intended_use_cohort": round(100 * in_cohort.mean(), 1),
        **{m: est(mc[m], ci([b[m] for b in bc])) for m in ["sens", "spec", "ppv", "npv", "prev", "auroc"]}}

    # ---------- E2 tipping points (full data)
    pdv = W.fit_predict(X0[ver], D[ver].astype(int), X0[~ver])
    lgu = np.log(pdv / (1 - pdv))

    def mnar(lam):
        pl = 1 / (1 + np.exp(-(lgu + np.log(lam))))
        Tm = np.concatenate([T[ver], T[~ver], T[~ver]])
        Dm = np.concatenate([D[ver], np.ones((~ver).sum()), np.zeros((~ver).sum())])
        wm = np.concatenate([np.ones(ver.sum()), pl, 1 - pl])
        sm = np.concatenate([sc[ver], sc[~ver], sc[~ver]])
        out = W.metrics(Tm, Dm, wm, sm, 1)
        out["prev_unverified"] = 100 * pl.mean()
        return out

    f_sens = lambda ll: (naive["sens"] - mnar(np.exp(ll))["sens"]) - MATERIAL
    f_spec = lambda ll: (mnar(np.exp(ll))["spec"] - naive["spec"]) - MATERIAL
    tip = {}
    for name, f, lo, hi in [("sens", f_sens, np.log(1e-4), np.log(1.0)), ("spec", f_spec, np.log(1.0), np.log(1e4))]:
        if np.sign(f(lo)) != np.sign(f(hi)):
            ll = brentq(f, lo, hi, xtol=1e-6)
            mm = mnar(np.exp(ll))
            tip[name] = {"lambda": round(float(np.exp(ll)), 4), "mnar_sens": round(mm["sens"], 2), "mnar_spec": round(mm["spec"], 2),
                         "implied_prev_unverified_pct": round(mm["prev_unverified"], 2)}
        else:
            tip[name] = {"lambda": None, "note": f"no crossing in [{np.exp(lo):g}, {np.exp(hi):g}]"}
    mar = mnar(1.0)
    res["E2_tipping_points"] = {"naive_sens": round(naive["sens"], 2), "naive_spec": round(naive["spec"], 2),
                                "prev_verified_pct": round(100 * D[ver].mean(), 2),
                                "implied_prev_unverified_at_lambda1_pct": round(mar["prev_unverified"], 2), **tip}

    # ---------- E3 labs
    labs = c.execute(f"""
    SELECT k.subject_id,
      max(CASE WHEN l.itemid=50963 THEN 1 ELSE 0 END) bnp_measured,
      max(CASE WHEN l.itemid=50963 AND l.valuenum > 0 THEN l.valuenum END) bnp_max,
      max(CASE WHEN l.itemid IN (51003, 51002, 52642) THEN 1 ELSE 0 END) trop_measured,
      max(CASE WHEN l.itemid=51003 AND l.valuenum >= 0.01 THEN 1 ELSE 0 END) trop_elevated
    FROM '{OUT / "wp1_cohort.parquet"}' k JOIN '{OUT / "labs_bnp_trop.parquet"}' l
      ON l.subject_id=k.subject_id AND l.charttime BETWEEN k.ecg_time - INTERVAL 24 HOUR AND k.ecg_time + INTERVAL 6 HOUR
    GROUP BY 1""").df()
    dl = d[["subject_id"]].merge(labs, on="subject_id", how="left")
    for col in ["bnp_measured", "trop_measured", "trop_elevated"]:
        dl[col] = dl[col].fillna(0)
    lb = np.log10(dl.bnp_max)
    lb = lb.fillna(lb.median())
    extra = np.column_stack([dl.bnp_measured, lb, dl.trop_measured, dl.trop_elevated])
    X3 = W.design(d, extra)
    res["E3_labs_availability_pct"] = {"bnp_measured": round(100 * dl.bnp_measured.mean(), 2),
                                       "trop_measured": round(100 * dl.trop_measured.mean(), 2),
                                       "trop_elevated": round(100 * dl.trop_elevated.mean(), 2),
                                       "V30_if_bnp_measured": round(100 * V[dl.bnp_measured == 1].mean(), 1),
                                       "V30_if_trop_measured": round(100 * V[dl.trop_measured == 1].mean(), 1)}
    p3 = W.fit_predict(X3, V, X3)
    res["E3_verification_c_statistic"] = {"base": round(roc_auc_score(V, p0), 4), "with_labs": round(roc_auc_score(V, p3), 4)}

    # ---------- E3 + E4 point estimates and bootstrap
    TQ = {"none": None, "p95": 0.95, "p99_primary": 0.99, "p99.5": 0.995}

    def run(idx):
        v, t, dd, s_ = V[idx], T[idx], D[idx], sc[idx]
        vv = v == 1
        nv = W.metrics(t[vv], dd[vv], np.ones(vv.sum()), s_[vv], 0)
        out = {}
        w = 1 / W.fit_predict(X0[idx], v, X0[idx][vv])
        for k_, q in TQ.items():
            m = W.metrics(t[vv], dd[vv], W.truncate(w, q), s_[vv], 1)
            out[f"E4_{k_}"] = {"sens": m["sens"], "spec": m["spec"], "ppv": m["ppv"], "auroc": m["auroc"],
                               "d_sens": nv["sens"] - m["sens"], "d_spec": nv["spec"] - m["spec"]}
        w3 = W.truncate(1 / W.fit_predict(X3[idx], v, X3[idx][vv]), 0.99)
        m = W.metrics(t[vv], dd[vv], w3, s_[vv], 1)
        out["E3_labs"] = {"sens": m["sens"], "spec": m["spec"], "ppv": m["ppv"], "auroc": m["auroc"],
                          "d_sens": nv["sens"] - m["sens"], "d_spec": nv["spec"] - m["spec"]}
        return out

    pt = run(np.arange(N))
    bs = Parallel(n_jobs=30)(delayed(run)(np.random.default_rng([SEED, b]).integers(0, N, N)) for b in range(B))
    for k_, v_ in pt.items():
        res[k_] = {m: est(v_[m], ci([b[k_][m] for b in bs])) for m in v_}

    # ---------- figure data (aggregate only)
    def roc_pts(y, s, w=None, n=150):
        fpr, tpr, _ = roc_curve(y, s, sample_weight=w)
        i = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
        return {"fpr": fpr[i].round(4).tolist(), "tpr": tpr[i].round(4).tolist()}

    wt = W.truncate(w0, 0.99)
    dec = pd.qcut(d.logit, 10, labels=False)
    grad = []
    for g in range(10):
        m_ = (dec == g).to_numpy()
        n_, k_ = int(m_.sum()), int(V[m_].sum())
        z = 1.96; ph = k_ / n_
        den = 1 + z * z / n_; cen = (ph + z * z / (2 * n_)) / den; hw = z * np.sqrt(ph * (1 - ph) / n_ + z * z / (4 * n_ * n_)) / den
        grad.append({"decile": g + 1, "n": n_, "v30_pct": round(100 * ph, 2), "lo": round(100 * (cen - hw), 2), "hi": round(100 * (cen + hw), 2),
                     "pct_test_positive": round(100 * T[m_].mean(), 1)})
    r07all = json.loads((OUT / "wp3_results.json").read_text())
    fig = {"roc": {"conventional": roc_pts(Dc.astype(int), Sc), "naive": roc_pts(D[ver].astype(int), sc[ver]),
                   "ipw": roc_pts(D[ver].astype(int), sc[ver], wt)},
           "operating_points": {"conventional": {"fpr": round(1 - mc["spec"] / 100, 4), "tpr": round(mc["sens"] / 100, 4)},
                                "naive": {"fpr": round(1 - naive["spec"] / 100, 4), "tpr": round(naive["sens"] / 100, 4)},
                                "ipw": {"fpr": round(1 - ipw["spec"] / 100, 4), "tpr": round(ipw["sens"] / 100, 4)},
                                "source_split": {"fpr": round(1 - 83.6 / 100, 4), "tpr": round(70.1 / 100, 4)}},
           "auroc": {"conventional": round(mc["auroc"], 1), "naive": round(naive["auroc"], 1), "ipw": round(ipw["auroc"], 1)},
           "verification_gradient": grad,
           "mnar": [{"lambda": float(l), **{m: r07all["main"][f"mnar_{l}"][m] for m in ["sens", "spec"]}} for l in ["0.25", "0.5", "0.75", "1.0", "1.5", "2.0"]],
           "naive": {m: r07all["main"]["naive"][m] for m in ["sens", "spec"]},
           "ipw": {m: r07all["main"]["ipw"][m] for m in ["sens", "spec"]},
           "n": {"cohort": N, "verified": int(ver.sum()), "conventional": int(len(conv))}}
    (OUT / "fig1_data.json").write_text(json.dumps(fig, indent=1))
    (OUT / "wp3_exploratory.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
