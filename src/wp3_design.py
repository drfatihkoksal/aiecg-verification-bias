"""Shared WP3 data/design/estimator code, copied verbatim from src/07_wp3_analysis.py so that exploratory
analyses (src/09) use the identical analytic cohort, design matrix and estimators. 09 asserts that this module
reproduces the 07 point estimates before doing anything else."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parents[1] / "outputs"
Q = [.05, .275, .5, .725, .95]
FLAGS = ["ecg_af_flutter", "ecg_lbbb", "ecg_rbbb", "ecg_infarct_q", "ecg_lvh", "ecg_low_voltage",
         "prior_admission", "index_admitted", "prior_cad_mi", "prior_af", "prior_valve",
         "hx_htn", "hx_dm", "hx_ckd", "hx_copd", "prior_tte_365",
         "ed_cc_dyspnoea", "ed_cc_chest_pain", "ed_cc_syncope", "ed_cc_oedema", "ed_cc_other", "ed_cc_missing"]


def load():
    k = pd.read_parquet(OUT / "wp1_cohort.parquet")
    x = pd.read_parquet(OUT / "wp1b_covariates.parquet")
    s = pd.read_parquet(OUT / "wp2_scores.parquet")
    d = (k.merge(x, on=["subject_id", "study_id"], validate="1:1")
          .merge(s[["subject_id", "logit", "score", "positive", "technical_exclusion", "threshold"]], on="subject_id", validate="1:1"))
    return d[~d.technical_exclusion].reset_index(drop=True)


def rcs(v, knots):
    t = knots; kk = len(t)
    cols = [v]
    for j in range(kk - 2):
        cols.append((np.maximum(v - t[j], 0) ** 3
                     - np.maximum(v - t[kk - 2], 0) ** 3 * (t[kk - 1] - t[j]) / (t[kk - 1] - t[kk - 2])
                     + np.maximum(v - t[kk - 1], 0) ** 3 * (t[kk - 2] - t[j]) / (t[kk - 1] - t[kk - 2])) / (t[kk - 1] - t[0]) ** 2)
    return np.column_stack(cols)


def design(d, extra=None):
    kn_s, kn_a = np.quantile(d.logit, Q), np.quantile(d.age, Q)
    era = d.era.astype(str).replace({"2020-2022": "2017-2019"})
    cols = [rcs(d.logit.to_numpy(float), kn_s), rcs(d.age.to_numpy(float), kn_a),
            (d.gender == "M").astype(float),
            pd.get_dummies(era, drop_first=True).astype(float).to_numpy(),
            pd.get_dummies(d.setting).drop(columns="outpatient").astype(float).to_numpy(),
            d[FLAGS].astype(float).to_numpy(),
            d.hr.fillna(d.hr.median()).to_numpy(float), d.qrs_ms.fillna(d.qrs_ms.median()).to_numpy(float)]
    if extra is not None:
        cols.append(np.asarray(extra, float))
    X = np.column_stack(cols)
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    assert np.isfinite(X).all()
    return X


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


def truncate(w, q):
    return w if q is None else np.minimum(w, np.quantile(w, q))
