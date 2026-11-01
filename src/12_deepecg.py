"""Second model, DeepECG-SL LVEF regression (concept.md §6.8). EXPLORATORY; the primary analysis is unchanged.

Model, loader and preprocessing are reused from rise-spec without change:
  analysis/14_deepecg_check.py (strict load), analysis/15_deepecg_score.py (ported DeepECG_Docker preprocessing,
  reader, scorer), MIMIC preprocessing parameters frozen on 30 Oct 2026 in results/wp_d_g2.json.
Guard: rescoring 300 rise-spec ECGs must reproduce rise-spec's predicted LVEF (max |diff| < 0.05, mean < 0.005 LVEF
points). Inference runs in full FP32 (TF32 off), which is batch-size invariant; rise-spec ran with TF32 on, whose
batch-dependent rounding explains differences up to ~0.015 points (deviation logged 1 Nov 2026, before any cohort score).

Index test, two operating points (fixed before any DeepECG score was computed in this project):
  primary   matched positivity: positive if predicted LVEF <= the value that flags the same share of the
            intended-use analytic cohort as EchoNext-Mini (scores only, no outcome data);
  secondary predicted LVEF <= 45%.
Score for splines/AUROC = -predicted LVEF. Estimators as src/07 (naive; IPW, 99th-percentile truncation; MNAR
lambda grid), verification model with the DeepECG score spline. Published-design comparator: E1b (Otabor) pairs.
If DeepECG cannot read some cohort ECGs, EchoNext naive/IPW are re-estimated on the same patients.
Bootstrap: 1000 patient resamples, seed 20261104.
"""
import sys, json, importlib, importlib.util, datetime, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=UserWarning)
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import wp3_design as W  # noqa: E402

OUT = W.OUT
RS = Path("/home/fatih/projects/rise-spec")
ECG = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c/mimic-iv-ecg/1.0")
B, SEED, LAMBDAS, MATERIAL = 1000, 20261104, [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 5.0


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def ci(v):
    return np.percentile(v, [2.5, 97.5]).round(4).tolist()


def score_deepecg(pairs, guard_only=False):
    import torch
    chk = _mod("deepecg_check", RS / "analysis/14_deepecg_check.py")
    m15 = _mod("deepecg_score", RS / "analysis/15_deepecg_score.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _ = chk.load_model(device)
    prm = json.loads((RS / "results/wp_d_g2.json").read_text())["preprocessing_frozen"]["mimic"]
    prep = m15.Prep()
    prep.scale = prm["scale"]
    prep.peaks = [tuple(x) for x in prm["peak_ranges_hz"]]
    prep.ranges = [[tuple(r) for r in lead] for lead in prm["flatten_ranges_hz_by_lead"]] if prm["flatten_ranges_hz_by_lead"] else None
    rl = pd.read_csv(ECG / "record_list.csv", usecols=["subject_id", "study_id", "path"])
    # guard: reproduce rise-spec
    ref = pd.read_parquet(RS / "work/wp_d_g2_mimic.parquet").sample(300, random_state=SEED).merge(rl, on=["subject_id", "study_id"])
    rr = m15.score(ref.rename(columns={"pred_lvef": "pred_ref"}).assign(src="mimic"), "src", "path", {"mimic": prep}, model, device)
    dif = np.abs(rr.pred_lvef - rr.pred_ref)
    guard = {"max": float(dif.max()), "mean": float(dif.mean())}
    assert len(rr) == len(ref) and guard["max"] < 0.05 and guard["mean"] < 0.005, guard
    if guard_only:
        return None, guard, None
    df = pairs[["subject_id", "study_id"]].drop_duplicates().merge(rl, on=["subject_id", "study_id"]).assign(src="mimic")
    s = m15.score(df, "src", "path", {"mimic": prep}, model, device)
    return s[["subject_id", "study_id", "pred_lvef"]], guard, {"scale": prm["scale"], "peak_ranges_hz": prm["peak_ranges_hz"]}


def main():
    e1b = importlib.import_module("11_e1b_e5")
    d = W.load()
    pairs, _ = e1b.build_e1b_pairs()
    tech = pd.read_parquet(OUT / "wp3_e1b_e5_scores.parquet")[["subject_id", "study_id", "technical_exclusion"]]
    pairs = pairs.merge(tech, on=["subject_id", "study_id"])
    pairs = pairs[~pairs.technical_exclusion].reset_index(drop=True)
    to_score = pd.concat([d[["subject_id", "study_id"]], pairs[["subject_id", "study_id"]]]).drop_duplicates()
    cache = OUT / "wp3_deepecg_scores.parquet"  # patient-level, local only
    res = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"), "status": "EXPLORATORY (post hoc), concept.md §6.8"}
    if cache.exists() and len(pd.read_parquet(cache).merge(to_score, on=["subject_id", "study_id"])) >= 0.99 * len(to_score):
        sc = pd.read_parquet(cache)
        _, guard, _ = score_deepecg(None, guard_only=True)
        res["guard_rise_spec_max_abs_diff"] = guard
    else:
        sc, guard, prm = score_deepecg(to_score)
        sc.to_parquet(cache, index=False)
        res["guard_rise_spec_max_abs_diff"] = guard
    res["scored"] = {"requested": int(len(to_score)), "readable": int(len(sc))}

    # ---------------- intended-use cohort
    dk = d.merge(sc, on=["subject_id", "study_id"], how="inner").reset_index(drop=True)
    pos_share = float(d.positive.mean())
    thr_match = float(np.quantile(dk.pred_lvef, pos_share))
    res["cohort"] = {"echonext_analytic": int(len(d)), "deepecg_readable": int(len(dk)), "lost": int(len(d) - len(dk)),
                     "matched_positive_share": round(100 * pos_share, 4), "threshold_matched_pred_lvef": round(thr_match, 3),
                     "pct_positive_lvef45": round(100 * (dk.pred_lvef <= 45).mean(), 4),
                     "pred_lvef_quartiles": [round(float(q), 4) for q in np.quantile(dk.pred_lvef, [.25, .5, .75])]}
    rules = {"matched": (dk.pred_lvef <= thr_match).astype(int).to_numpy(), "lvef45": (dk.pred_lvef <= 45).astype(int).to_numpy()}
    xD = dk.copy(); xD["logit"] = -dk.pred_lvef
    XD, XE = W.design(xD), W.design(dk)
    S_D, S_E = -dk.pred_lvef.to_numpy(float), dk.logit.to_numpy(float)
    T_E = dk.positive.astype(int).to_numpy()
    V, D = dk.V30.to_numpy(int), dk.d_v30.to_numpy(float)
    N = len(dk)
    same = len(dk) == len(d)

    def run(idx, full):
        v, dd = V[idx], D[idx]
        vv = v == 1
        out = {}
        w = W.truncate(1 / W.fit_predict(XD[idx], v, XD[idx][vv]), 0.99)
        if full:
            pdv = W.fit_predict(XD[idx][vv], dd[vv].astype(int), XD[idx][~vv])
            lg = np.log(pdv / (1 - pdv))
        for rn, T in rules.items():
            t, s_ = T[idx], S_D[idx]
            nv = W.metrics(t[vv], dd[vv], np.ones(vv.sum()), s_[vv], 0)
            ip = W.metrics(t[vv], dd[vv], w, s_[vv], 1)
            out[f"{rn}_naive"] = {k: nv[k] for k in ["sens", "spec", "ppv", "npv", "prev", "auroc"]}
            out[f"{rn}_ipw"] = {k: ip[k] for k in ["sens", "spec", "ppv", "npv", "prev", "auroc", "echos_per_case"]}
            out[f"{rn}_delta"] = {k: nv[k] - ip[k] for k in ["sens", "spec", "ppv", "auroc"]}
            if full:
                for lam in LAMBDAS:
                    pl = 1 / (1 + np.exp(-(lg + np.log(lam))))
                    m = W.metrics(np.concatenate([t[vv], t[~vv], t[~vv]]),
                                  np.concatenate([dd[vv], np.ones((~vv).sum()), np.zeros((~vv).sum())]),
                                  np.concatenate([np.ones(vv.sum()), pl, 1 - pl]),
                                  np.concatenate([s_[vv], s_[~vv], s_[~vv]]), 1)
                    out[f"{rn}_mnar_{lam}"] = {k: m[k] for k in ["sens", "spec", "ppv", "npv", "prev", "auroc"]}
        if full and not same:  # EchoNext on the same patients
            wE = W.truncate(1 / W.fit_predict(XE[idx], v, XE[idx][vv]), 0.99)
            t, s_ = T_E[idx], S_E[idx]
            nv = W.metrics(t[vv], dd[vv], np.ones(vv.sum()), s_[vv], 0)
            ip = W.metrics(t[vv], dd[vv], wE, s_[vv], 1)
            out["echonext_same_patients_delta"] = {k: nv[k] - ip[k] for k in ["sens", "spec", "ppv", "auroc"]}
        return out

    pt = run(np.arange(N), True)
    bs = Parallel(n_jobs=30)(delayed(run)(np.random.default_rng([SEED, b]).integers(0, N, N), True) for b in range(B))
    res["intended_use"] = {blk: {k: {"est": round(float(v), 4), "ci": ci([b[blk][k] for b in bs])} for k, v in vals.items()}
                           for blk, vals in pt.items()}
    res["descriptive_gate_naive_auroc_ge_70"] = bool(pt["matched_naive"]["auroc"] >= 70)

    # readout (§4) at the primary operating point
    ds, dp = res["intended_use"]["matched_delta"]["sens"], res["intended_use"]["matched_delta"]["spec"]
    excl0 = lambda c: c[0] > 0 or c[1] < 0
    nv = pt["matched_naive"]
    mat = {str(l): abs(nv["sens"] - pt[f"matched_mnar_{l}"]["sens"]) >= MATERIAL or abs(nv["spec"] - pt[f"matched_mnar_{l}"]["spec"]) >= MATERIAL
           for l in LAMBDAS}
    flips = len(set(mat.values())) > 1
    material = (abs(ds["est"]) >= MATERIAL and excl0(ds["ci"])) or (abs(dp["est"]) >= MATERIAL and excl0(dp["ci"]))
    res["readout_matched"] = {"delta_sens": ds, "delta_spec": dp, "material_by_lambda": mat,
                              "category": ("Assumption-dependent; report as bounds only" if flips else
                                           "Verified-only validation misstates intended-use accuracy" if material else
                                           "No material distortion detected" if abs(ds["est"]) < MATERIAL and abs(dp["est"]) < MATERIAL
                                           else "Indeterminate")}

    # ---------------- published-design comparator (E1b pairs)
    pk = pairs.merge(sc, on=["subject_id", "study_id"], how="inner").reset_index(drop=True)
    y, s_ = pk.d.to_numpy(float), -pk.pred_lvef.to_numpy(float)
    rng = np.random.default_rng(SEED)
    idxs = [rng.integers(0, len(pk), len(pk)) for _ in range(B)]
    res["E1b_published_design"] = {"n": int(len(pk)), "lost_unreadable": int(len(pairs) - len(pk))}
    for rn, T in {"matched": (pk.pred_lvef <= thr_match).astype(int).to_numpy(), "lvef45": (pk.pred_lvef <= 45).astype(int).to_numpy()}.items():
        m = W.metrics(T, y, np.ones(len(y)), s_, 0)
        bs2 = [W.metrics(T[i], y[i], np.ones(len(i)), s_[i], 0) for i in idxs]
        res["E1b_published_design"][rn] = {k: {"est": round(m[k], 4), "ci": ci([b[k] for b in bs2])} for k in ["sens", "spec", "ppv", "npv", "prev", "auroc"]}
    (OUT / "wp3_deepecg.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: res[k] for k in ["scored", "cohort", "readout_matched", "descriptive_gate_naive_auroc_ge_70"]}, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
