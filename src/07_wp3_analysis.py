"""WP3 - verification model and estimators (concept.md §6.4-§6.6, spec v2).

Implementation details fixed here BEFORE any accuracy estimate was computed:
  * Analytic cohort: WP1 cohort minus WP2 technical exclusions. Test positive T = score >= frozen threshold.
  * Splines: restricted cubic splines (Harrell), 5 knots at the 5/27.5/50/72.5/95th percentiles of the analytic
    cohort; AI score entered on the logit scale. Knots are fixed across bootstrap resamples.
  * Era: approx-year bins; 2020-2022 (n~130; MIMIC-IV-ECG ends 2019, approx year is +-1.5 y) merged into 2017+.
  * HR and QRS: linear; the 0.1% missing values median-imputed.
  * ED complaint flags: dyspnoea, chest pain, syncope, oedema, other, missing (non-ED patients: all 0).
  * Logistic regression on standardised design, L2 with C = 1e4 (numerical stability only; effectively MLE).
  * Same covariates in the verification model P(V|X) and the outcome model P(D|X, V=1).
  * IPW: Hajek-normalised; weights 1/p truncated at the 99th percentile of weights among verified patients.
  * AUROC with weights: sklearn roc_auc_score(sample_weight) = weighted Mann-Whitney.
  * MNAR (outcome imputation): unverified patient contributes D=1 with weight p_lambda and D=0 with 1 - p_lambda.
  * ECGs screened per detected case = N / TP ; echos per detected case = (TP + FP) / TP = 1 / PPV.
  * Bootstrap: 1000 patient resamples (seed 20261101), both models refitted, threshold and knots fixed;
    percentile 95% CIs.
  * Sensitivity analyses: (a) 0-365 day window (V365, D365; verification model refitted on V365);
    (b) straddling LVEF ranges as D = 1 (V30 model unchanged).
"""
import json, datetime, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=UserWarning)
OUT = Path(__file__).resolve().parents[1] / "outputs"
import os
B, SEED, LAMBDAS, TRUNC_Q = int(os.environ.get("WP3_B", 1000)), 20261101, [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 0.99
MATERIAL = 5.0  # percentage points

# ------------------------------------------------------------------ data
k = pd.read_parquet(OUT / "wp1_cohort.parquet")
x = pd.read_parquet(OUT / "wp1b_covariates.parquet")
s = pd.read_parquet(OUT / "wp2_scores.parquet")
d = (k.merge(x, on=["subject_id", "study_id"], validate="1:1")
      .merge(s[["subject_id", "logit", "score", "positive", "technical_exclusion"]], on="subject_id", validate="1:1"))
d = d[~d.technical_exclusion].reset_index(drop=True)
N = len(d)
T = d.positive.astype(int).to_numpy()


def rcs(v, knots):
    t = knots; kk = len(t)
    cols = [v]
    for j in range(kk - 2):
        cols.append((np.maximum(v - t[j], 0) ** 3
                     - np.maximum(v - t[kk - 2], 0) ** 3 * (t[kk - 1] - t[j]) / (t[kk - 1] - t[kk - 2])
                     + np.maximum(v - t[kk - 1], 0) ** 3 * (t[kk - 2] - t[j]) / (t[kk - 1] - t[kk - 2])) / (t[kk - 1] - t[0]) ** 2)
    return np.column_stack(cols)


Q = [.05, .275, .5, .725, .95]
kn_s, kn_a = np.quantile(d.logit, Q), np.quantile(d.age, Q)
era = d.era.astype(str).replace({"2020-2022": "2017-2019"})
FLAGS = ["ecg_af_flutter", "ecg_lbbb", "ecg_rbbb", "ecg_infarct_q", "ecg_lvh", "ecg_low_voltage",
         "prior_admission", "index_admitted", "prior_cad_mi", "prior_af", "prior_valve",
         "hx_htn", "hx_dm", "hx_ckd", "hx_copd", "prior_tte_365",
         "ed_cc_dyspnoea", "ed_cc_chest_pain", "ed_cc_syncope", "ed_cc_oedema", "ed_cc_other", "ed_cc_missing"]
Xd = np.column_stack([
    rcs(d.logit.to_numpy(float), kn_s), rcs(d.age.to_numpy(float), kn_a),
    (d.gender == "M").astype(float),
    pd.get_dummies(era, drop_first=True).astype(float).to_numpy(),
    pd.get_dummies(d.setting).drop(columns="outpatient").astype(float).to_numpy(),
    d[FLAGS].astype(float).to_numpy(),
    d.hr.fillna(d.hr.median()).to_numpy(float), d.qrs_ms.fillna(d.qrs_ms.median()).to_numpy(float),
])
Xd = (Xd - Xd.mean(0)) / np.where(Xd.std(0) > 0, Xd.std(0), 1)
assert np.isfinite(Xd).all()


def fit_predict(X, y, Xpred):
    m = LogisticRegression(C=1e4, max_iter=5000).fit(X, y)
    return m.predict_proba(Xpred)[:, 1]


def metrics(Tv, Dv, w, sv, n_total):
    tp = np.sum(w * Tv * Dv); fn = np.sum(w * (1 - Tv) * Dv)
    fp = np.sum(w * Tv * (1 - Dv)); tn = np.sum(w * (1 - Tv) * (1 - Dv))
    return {"sens": 100 * tp / (tp + fn), "spec": 100 * tn / (tn + fp), "ppv": 100 * tp / (tp + fp),
            "npv": 100 * tn / (tn + fn), "prev": 100 * (tp + fn) / np.sum(w),
            "auroc": 100 * roc_auc_score(Dv, sv, sample_weight=w),
            "ecgs_per_case": np.sum(w) / tp if n_total else np.nan, "echos_per_case": (tp + fp) / tp}


def estimate(idx, V, D, full=True):
    """All estimators on the resample idx. V, D: arrays over the analytic cohort (D NaN when V = 0)."""
    X, v, t, sc = Xd[idx], V[idx], T[idx], d.logit.to_numpy()[idx]
    dd = D[idx]
    ver = v == 1
    out = {"naive": metrics(t[ver], dd[ver], np.ones(ver.sum()), sc[ver], 0)}
    p = fit_predict(X, v, X[ver])
    w = 1 / p
    cap = np.quantile(w, TRUNC_Q)
    wt = np.minimum(w, cap)
    out["ipw"] = metrics(t[ver], dd[ver], wt, sc[ver], 1)
    out["delta"] = {m: out["naive"][m] - out["ipw"][m] for m in ["sens", "spec", "ppv", "npv", "auroc"]}
    if not full:
        return out
    setting = d.setting.to_numpy()[idx][ver]
    out["ipw_by_setting"] = {g: metrics(t[ver][setting == g], dd[ver][setting == g], wt[setting == g], sc[ver][setting == g], 1)
                             for g in ["outpatient", "ED", "inpatient"]}
    out["naive_by_setting"] = {g: metrics(t[ver][setting == g], dd[ver][setting == g], np.ones((setting == g).sum()), sc[ver][setting == g], 0)
                               for g in ["outpatient", "ED", "inpatient"]}
    pd_ = fit_predict(X[ver], dd[ver].astype(int), X[~ver])
    lg = np.log(pd_ / (1 - pd_))
    for lam in LAMBDAS:
        pl = 1 / (1 + np.exp(-(lg + np.log(lam))))
        Tm = np.concatenate([t[ver], t[~ver], t[~ver]])
        Dm = np.concatenate([dd[ver], np.ones((~ver).sum()), np.zeros((~ver).sum())])
        wm = np.concatenate([np.ones(ver.sum()), pl, 1 - pl])
        sm = np.concatenate([sc[ver], sc[~ver], sc[~ver]])
        out[f"mnar_{lam}"] = metrics(Tm, Dm, wm, sm, 1)
    out["weights"] = {"pct_truncated": 100 * np.mean(w > cap), "cap": cap,
                      "quantiles_untruncated": dict(zip(["min", "p50", "p90", "p99", "max"],
                                                        np.quantile(w, [0, .5, .9, .99, 1]).round(4).tolist()))}
    return out


V30 = d.V30.to_numpy(int); D30 = d.d_v30.to_numpy(float)
V365 = d.V365.to_numpy(int); D365 = d.d_v365.to_numpy(float)
D30s = d.d_sens_v30.to_numpy(float)
allidx = np.arange(N)


def one(b):
    idx = np.random.default_rng([SEED, b]).integers(0, N, N)
    return {"main": estimate(idx, V30, D30, full=True),
            "sens_365": estimate(idx, V365, D365, full=False),
            "sens_straddle": estimate(idx, V30, D30s, full=False)}


point = {"main": estimate(allidx, V30, D30), "sens_365": estimate(allidx, V365, D365, full=False),
         "sens_straddle": estimate(allidx, V30, D30s, full=False)}
boot = Parallel(n_jobs=30, verbose=0)(delayed(one)(b) for b in range(B))


def ci(path):
    vals = []
    for r in boot:
        v = r
        for kx in path:
            v = v[kx]
        vals.append(v)
    return np.percentile(vals, [2.5, 97.5]).round(4).tolist()


def walk(pt, path=()):
    if isinstance(pt, dict):
        return {kx: walk(vx, path + (kx,)) for kx, vx in pt.items()}
    return {"est": round(float(pt), 4), "ci": ci(path)}


res = walk({kx: {mk: mv for mk, mv in v.items() if mk != "weights"} for kx, v in point.items()})
res["main"]["weights"] = point["main"]["weights"]

# verification model diagnostics (full data)
p_all = fit_predict(Xd, V30, Xd)
dec = pd.qcut(d.logit, 10, labels=False)
res["verification_model"] = {
    "c_statistic": round(roc_auc_score(V30, p_all), 4),
    "V30_by_score_decile_pct": (100 * pd.Series(V30).groupby(dec).mean()).round(1).tolist(),
    "V30_by_setting_pct": (100 * pd.Series(V30).groupby(d.setting).mean()).round(1).to_dict(),
    "V30_by_test_result_pct": {"T=1": round(100 * V30[T == 1].mean(), 1), "T=0": round(100 * V30[T == 0].mean(), 1)},
    "n_design_columns": int(Xd.shape[1]),
}

# pre-specified readout (§4)
ds, dp = res["main"]["delta"]["sens"], res["main"]["delta"]["spec"]
excl0 = lambda c: c[0] > 0 or c[1] < 0
material = (abs(ds["est"]) >= MATERIAL and excl0(ds["ci"])) or (abs(dp["est"]) >= MATERIAL and excl0(dp["ci"]))
none_material = abs(ds["est"]) < MATERIAL and abs(dp["est"]) < MATERIAL
naive = res["main"]["naive"]


def material_under(lam):
    m = res["main"][f"mnar_{lam}"]
    return abs(naive["sens"]["est"] - m["sens"]["est"]) >= MATERIAL or abs(naive["spec"]["est"] - m["spec"]["est"]) >= MATERIAL


flips = len({material_under(l) for l in LAMBDAS}) > 1
res["readout"] = {
    "delta_sens": ds, "delta_spec": dp,
    "category": ("Assumption-dependent; report as bounds only" if flips else
                 "Verified-only validation misstates intended-use accuracy" if material else
                 "No material distortion detected" if none_material else "Indeterminate (>= 5 points but CI includes 0)"),
    "material_by_lambda": {str(l): material_under(l) for l in LAMBDAS},
    "H1_naive_sens_gt_corrected": ds["est"] > 0, "H1_naive_spec_lt_corrected": dp["est"] < 0,
    "H2_corrected_ppv_lt_naive": res["main"]["delta"]["ppv"]["est"] > 0,
    "lambda1_minus_ipw": {m: round(res["main"]["mnar_1.0"][m]["est"] - res["main"]["ipw"][m]["est"], 4)
                          for m in ["sens", "spec", "ppv", "npv", "prev", "auroc"]},
}
res["meta"] = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"), "N_analytic": N,
               "verified": int(V30.sum()), "B": B, "seed": SEED, "threshold": float(s.threshold.iloc[0]) if "threshold" in s else None}
(OUT / ("wp3_results.json" if B == 1000 else f"wp3_results_B{B}_test.json")).write_text(json.dumps(res, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o)))
print(json.dumps(res["readout"], indent=2, default=str))
