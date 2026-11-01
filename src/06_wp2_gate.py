"""WP2 - technical exclusion flow and G2 transport gate (concept.md §7).

G2: naive AUROC (verified patients only, V30 = 1, primary D rule) >= 0.80.
Only the gate is computed here. No sensitivity/specificity/PPV and no corrected estimate (WP3).
"""
import json, datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parents[1] / "outputs"
s = pd.read_parquet(OUT / "wp2_scores.parquet")
k = pd.read_parquet(OUT / "wp1_cohort.parquet", columns=["subject_id", "V30", "d_v30"])
d = k.merge(s, on="subject_id", how="left", validate="1:1")
assert d.score.isna().sum() == d.load_failed.sum()  # flat-lead records are scored but excluded

a = d[~d.technical_exclusion]
v = a[a.V30 == 1]
y, x = v.d_v30.astype(int).to_numpy(), v.score.to_numpy()
auc = roc_auc_score(y, x)
rng = np.random.default_rng(20261101)
bs = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    bs.append(roc_auc_score(y[i], x[i]))
lo, hi = np.percentile(bs, [2.5, 97.5])

res = {
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "technical_exclusion": {
        "cohort_wp1": int(len(d)),
        "load_failed": int(d.load_failed.sum()),
        "flat_lead_records": int((d.flat_leads.fillna(0) > 0).sum()),
        "missing_lead_records": int((d.nan_leads.fillna(0) > 0).sum()),
        "excluded_total": int(d.technical_exclusion.sum()),
        "excluded_V30": int((d.technical_exclusion & (d.V30 == 1)).sum()),
        "load_errors": d.error.dropna().str.split(":").str[0].value_counts().to_dict(),
        "analytic_cohort": int(len(a)),
    },
    "verified": {"n": int(len(v)), "d1": int(y.sum()), "prevalence_pct": round(100 * y.mean(), 2)},
    "G2": {"rule": "naive AUROC >= 0.80 (point estimate)", "naive_auroc": round(auc, 4),
           "ci95_bootstrap_1000": [round(lo, 4), round(hi, 4)], "passed": bool(auc >= 0.80)},
}
(OUT / "wp2_summary.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
