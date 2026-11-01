"""Render outputs/wp3_results.json into outputs/wp3_results.md (every number traceable to the JSON)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs"
r = json.loads((OUT / "wp3_results.json").read_text())
f = lambda x, nd=1: "—" if x["est"] != x["est"] else f'{x["est"]:.{nd}f} ({x["ci"][0]:.{nd}f}–{x["ci"][1]:.{nd}f})'
M = [("sens", "Sensitivity"), ("spec", "Specificity"), ("ppv", "PPV"), ("npv", "NPV"), ("prev", "Prevalence"),
     ("auroc", "AUROC"), ("ecgs_per_case", "ECGs per detected case"), ("echos_per_case", "Echos per detected case")]
L = []
mt = r["meta"]; vm = r["verification_model"]; w = r["main"]["weights"]
L += [f"# WP3 results — VERIFY-GAP", "",
      f"Generated {mt['generated_at']} by `src/07_wp3_analysis.py` (B = {mt['B']}, seed {mt['seed']}). "
      f"Analytic cohort {mt['N_analytic']:,}; verified (V30) {mt['verified']:,}. Threshold: score ≥ {mt['threshold']}. "
      "Values are % (95% bootstrap CI) unless stated.", "",
      "## Verification model", "",
      f"- C-statistic {vm['c_statistic']}; {vm['n_design_columns']} design columns.",
      f"- V30 by AI-score decile (low → high): {', '.join(str(x) for x in vm['V30_by_score_decile_pct'])}%.",
      f"- V30 by test result: T = 1 {vm['V30_by_test_result_pct']['T=1']}%, T = 0 {vm['V30_by_test_result_pct']['T=0']}%.",
      f"- V30 by setting: " + ", ".join(f"{k} {v}%" for k, v in vm["V30_by_setting_pct"].items()) + ".",
      f"- IPW weights (untruncated): median {w['quantiles_untruncated']['p50']}, p90 {w['quantiles_untruncated']['p90']}, "
      f"p99 {w['quantiles_untruncated']['p99']}, max {w['quantiles_untruncated']['max']}; "
      f"{w['pct_truncated']:.1f}% truncated at {w['cap']:.1f}.", "",
      "## Main estimators", "",
      "| Metric | Naive (verified only) | IPW (MAR, primary) | Δ (naive − IPW) |", "|---|---|---|---|"]
for m, lab in M:
    dl = f(r["main"]["delta"][m]) if m in r["main"]["delta"] else ""
    L.append(f"| {lab} | {f(r['main']['naive'][m])} | {f(r['main']['ipw'][m])} | {dl} |")
L += ["", "Naive 'ECGs per detected case' is undefined (only verified patients are observed).", "",
      "## MNAR range (outcome-model imputation)", "",
      "| λ | Sensitivity | Specificity | PPV | NPV | Prevalence | AUROC | Echos per case |", "|---|---|---|---|---|---|---|---|"]
for lam in ["0.25", "0.5", "0.75", "1.0", "1.5", "2.0"]:
    e = r["main"][f"mnar_{lam}"]
    L.append(f"| {lam} | " + " | ".join(f(e[m]) for m in ["sens", "spec", "ppv", "npv", "prev", "auroc", "echos_per_case"]) + " |")
L += ["", f"λ = 1 minus IPW (points): " + ", ".join(f"{k} {v:+.2f}" for k, v in r["readout"]["lambda1_minus_ipw"].items()) + ".", "",
      "## By setting", "", "| Setting | Estimator | Sensitivity | Specificity | PPV | Prevalence | AUROC |", "|---|---|---|---|---|---|---|"]
for g in ["outpatient", "ED", "inpatient"]:
    for est in ["naive", "ipw"]:
        e = r["main"][f"{est}_by_setting"][g]
        L.append(f"| {g} | {est} | " + " | ".join(f(e[m]) for m in ["sens", "spec", "ppv", "prev", "auroc"]) + " |")
L += ["", "## Sensitivity analyses", "", "| Analysis | Δsens | Δspec | ΔPPV | IPW sens | IPW spec |", "|---|---|---|---|---|---|"]
for sa, lab in [("main", "Primary (0–30 d, midpoint rule)"), ("sens_365", "0–365-day window"), ("sens_straddle", "Straddlers as D = 1")]:
    e = r[sa]
    L.append(f"| {lab} | {f(e['delta']['sens'])} | {f(e['delta']['spec'])} | {f(e['delta']['ppv'])} | {f(e['ipw']['sens'])} | {f(e['ipw']['spec'])} |")
ro = r["readout"]
L += ["", "## Pre-specified readout (§4)", "",
      f"- Δsens {f(ro['delta_sens'])}, Δspec {f(ro['delta_spec'])}.",
      f"- **Category: {ro['category']}.** Material (≥5 points) at every λ: {all(ro['material_by_lambda'].values())}.",
      f"- H1: naive sens > corrected {ro['H1_naive_sens_gt_corrected']}; naive spec < corrected {ro['H1_naive_spec_lt_corrected']}.",
      f"- H2 (MAR): corrected PPV < naive {ro['H2_corrected_ppv_lt_naive']}."]
(OUT / "wp3_results.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
