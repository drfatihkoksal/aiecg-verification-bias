"""EXPLORATORY (post hoc, 1 Nov 2026; analysis by sex, added after the primary results were known). The primary
analysis is unchanged.

Sex = as recorded in the MIMIC-IV patients table (female, male). For each model the verification model is fitted
once in the whole analytic cohort, exactly as in the primary analysis (sex is one of its covariates; weights
truncated at the 99th percentile of all verified patients); naive and IPW accuracy are then computed within
women and within men, as for care settings. Reported per sex: verification rate, naive and IPW sensitivity,
specificity, PPV, delta (naive - IPW); and the between-sex difference of each delta (women - men).
Models: EchoNext-Mini (frozen threshold; WP2 analytic cohort) and DeepECG-SL (matched-positivity threshold;
readable-ECG cohort, src/12). Bootstrap: 1000 patient resamples, seed 20261105, models refitted.
"""
import sys, json, datetime, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=UserWarning)
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import wp3_design as W  # noqa: E402

OUT = W.OUT
B, SEED = 1000, 20261105
K = ["sens", "spec", "ppv", "prev"]


def setup():
    d = W.load()
    dg = json.loads((OUT / "wp3_deepecg.json").read_text())
    sc = pd.read_parquet(OUT / "wp3_deepecg_scores.parquet")
    dk = d.merge(sc, on=["subject_id", "study_id"], how="inner").reset_index(drop=True)
    thr = dg["cohort"]["threshold_matched_pred_lvef"]
    xD = dk.copy(); xD["logit"] = -dk.pred_lvef
    return {
        "EchoNext-Mini": dict(X=W.design(d), T=d.positive.astype(int).to_numpy(), S=d.logit.to_numpy(float),
                              V=d.V30.to_numpy(int), D=d.d_v30.to_numpy(float), F=(d.gender == "F").to_numpy()),
        "DeepECG-SL": dict(X=W.design(xD), T=(dk.pred_lvef <= thr).astype(int).to_numpy(), S=-dk.pred_lvef.to_numpy(float),
                           V=dk.V30.to_numpy(int), D=dk.d_v30.to_numpy(float), F=(dk.gender == "F").to_numpy()),
    }


def run(M, idx):
    out = {}
    for name, m in M.items():
        X, T, S, V, D, F = (m[k][idx] for k in ["X", "T", "S", "V", "D", "F"])
        vv = V == 1
        w = W.truncate(1 / W.fit_predict(X, V, X[vv]), 0.99)
        Tv, Dv, Sv, Fv = T[vv], D[vv], S[vv], F[vv]
        r = {}
        for sex, sel in [("women", Fv), ("men", ~Fv)]:
            nv = W.metrics(Tv[sel], Dv[sel], np.ones(sel.sum()), Sv[sel], 0)
            ip = W.metrics(Tv[sel], Dv[sel], w[sel], Sv[sel], 1)
            r[sex] = {"verified_pct": 100 * V[F if sex == "women" else ~F].mean(),
                      **{f"naive_{k}": nv[k] for k in K}, **{f"ipw_{k}": ip[k] for k in K},
                      **{f"delta_{k}": nv[k] - ip[k] for k in ["sens", "spec", "ppv"]}}
        r["women_minus_men"] = {f"delta_{k}": r["women"][f"delta_{k}"] - r["men"][f"delta_{k}"] for k in ["sens", "spec", "ppv"]}
        out[name] = r
    return out


def main():
    M = setup()
    N = {k: len(v["V"]) for k, v in M.items()}
    pt = {name: run({name: m}, np.arange(N[name]))[name] for name, m in M.items()}

    def one(b):
        return {name: run({name: m}, np.random.default_rng([SEED, b, j]).integers(0, N[name], N[name]))[name]
                for j, (name, m) in enumerate(M.items())}

    bs = Parallel(n_jobs=30)(delayed(one)(b) for b in range(B))
    res = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"), "status": "EXPLORATORY (post hoc)",
           "n": N, "women_n": {k: int(v["F"].sum()) for k, v in M.items()}}
    for name in M:
        res[name] = {grp: {k: {"est": round(float(v), 4), "ci": np.percentile([b[name][grp][k] for b in bs], [2.5, 97.5]).round(4).tolist()}
                           for k, v in vals.items()} for grp, vals in pt[name].items()}
    (OUT / "wp3_sex.json").write_text(json.dumps(res, indent=2))
    for name in M:
        print(name)
        for grp in ["women", "men", "women_minus_men"]:
            print("  ", grp, {k: f"{v['est']} ({v['ci'][0]}, {v['ci'][1]})" for k, v in res[name][grp].items() if k.startswith(("delta", "ipw_sens", "ipw_spec", "naive_sens", "naive_spec", "verified"))})


if __name__ == "__main__":
    main()
