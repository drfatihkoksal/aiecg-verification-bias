"""Freeze the design BEFORE any MIMIC index ECG is scored (concept.md §7, WP2).

Writes cohort_spec.yaml (decisions + SHA256 of every input, output and script) and a
verbatim snapshot of concept.md. Any later change to a frozen item must be logged with
its reason in concept.md §10 and the spec re-issued under a new version.
"""
import hashlib, json, shutil, datetime as dt
from pathlib import Path
import yaml

P = Path(__file__).resolve().parents[1]
OUT = P / "outputs"
AFEF = Path("/home/fatih/projects/afef")
TODAY = dt.date.today().isoformat()


def sha(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(buf):
            h.update(b)
    return h.hexdigest()


VERSION = "v2"
snap = P / f"concept_frozen_{VERSION}_{TODAY}.md"
shutil.copyfile(P / "concept.md", snap)

thr = json.loads((OUT / "source_threshold.json").read_text())
wp1 = json.loads((OUT / "wp1_summary.json").read_text())

FILES = [
    snap,
    *sorted((P / "src").glob("*.py")),
    OUT / "source_threshold.json", OUT / "decision_inputs.json",
    OUT / "wp1_tte_d45.parquet", OUT / "wp1_cohort.parquet", OUT / "wp1_flow.csv", OUT / "wp1_summary.json",
    OUT / "wp1b_covariates.parquet", OUT / "wp1b_summary.json",
    OUT / "codebook_icd.csv", OUT / "codebook_ecg_statements.csv", OUT / "codebook_ed_cc.csv",
    AFEF / "outputs/echo_ef_studies.parquet", AFEF / "outputs/ecg_rhythm_candidates.parquet",
    AFEF / "outputs/ecg_tabular_features.parquet",
    AFEF / "src/02_rhythm_screen.py", AFEF / "src/mimic_to_echonext.py", AFEF / "src/08_mimic_tabular.py",
    AFEF / "models/echonext_minimodel/weights.pt",
    AFEF / "models/echonext_minimodel/waveform_normalization_params.json",
    AFEF / "models/echonext_minimodel/tabular_params.npz",
]

spec = {
    "spec_version": f"verify_gap_spec_{VERSION}",
    "supersedes": {"version": "verify_gap_spec_v1", "file": "cohort_spec_v1.yaml", "reason": "HTN/DM/CKD/COPD also from the index admission (hx_*) + index_admitted indicator; WP2 script added. No MIMIC score existed when v1 was replaced."},
    "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
    "frozen_before": "WP2: scoring of MIMIC index ECGs with EchoNext-Mini. No MIMIC score exists at freeze time.",
    "concept_snapshot": snap.name,
    "index_test": {
        "model": "EchoNext-Mini (echonext_multilabel_minimodel), output lvef_lte_45 (index 0), frozen",
        "pipeline": "afef src/mimic_to_echonext.py + src/08_mimic_tabular.py, unchanged (afef cohort_spec_v1)",
        "threshold": {"rule": thr["rule"], "positive_if_sigmoid_gte": thr["lvef_lte_45"]["cutoff_prob"],
                      "source_split_metrics": {k: thr["lvef_lte_45"][k] for k in ("sensitivity", "specificity", "ppv", "npv")},
                      "rule_check_composite_shd_reproduced": thr["sanity_check_composite_shd"]["reproduced"]},
        "technical_exclusion": "load/format failure (afef checks), or any of 12 raw leads flat (SD < 1e-6 ADC) or entirely missing; applied in WP2 (src/05_wp2_score.py)",
    },
    "cohort": {
        "unit": "first ECG per patient in MIMIC-IV-ECG v1.0 (ecg_time, then study_id)",
        "exclusions": ["not linked to MIMIC-IV patients or sex missing", "age < 18 at index ECG",
                       "index ECG paced (any pacing statement, afef rhythm_rules_v2)",
                       "prior resting TTE with D = 1 (section 6.3 rule)",
                       "I50/428/I42/425 on an admission with dischtime < index ECG time"],
        "not_excluded": ["prosthetic valves (pre-index evidence enters X)", "any non-paced rhythm"],
        "setting_rule": "ED if within an ED stay, else inpatient if within an admission, else outpatient",
        "flow": wp1["flow"],
    },
    "verification": {
        "V": "first resting TTE with LVEF information, 0-30 days after the index ECG",
        "D_primary": "biplane > reported range (midpoint rule for closed straddlers, lower-bound rule if open-ended) > 3D > text; LVEF <= 45",
        "D_sensitivity": "all straddling studies D = 1",
        "window_sensitivity": "0-365 days",
        "V30": wp1["verification"]["V30_n"], "V30_pct": wp1["verification"]["V30_pct"],
        "G1": wp1["G1"],
    },
    "verification_model": {
        "form": "logistic regression; AI score and age as restricted cubic splines",
        "X": ["AI score", "age", "sex", "era (approx year from anchor_year_group)", "setting",
              "ecg_af_flutter", "ecg_lbbb", "ecg_rbbb", "ecg_infarct_q", "ecg_lvh", "ecg_low_voltage", "qrs_ms", "hr",
              "prior_admission", "index_admitted", "prior_cad_mi", "prior_af", "prior_valve (incl. pre-index prosthesis)",
              "hx_htn", "hx_dm", "hx_ckd", "hx_copd", "prior_tte_365",
              "ed_cc_dyspnoea", "ed_cc_chest_pain", "ed_cc_syncope", "ed_cc_oedema", "ed_cc_other", "ed_cc_missing"],
        "codebooks": ["outputs/codebook_icd.csv", "outputs/codebook_ecg_statements.csv", "outputs/codebook_ed_cc.csv"],
    },
    "estimators": {
        "naive": "verified only",
        "primary": "IPW 1/P(V=1|X), weights truncated at 99th percentile",
        "mnar": {"lambda": [0.25, 0.5, 0.75, 1, 1.5, 2], "consistency_check": "lambda = 1 ~ IPW"},
        "bootstrap": "1000 patient resamples, models refitted, threshold fixed",
        "primary_estimand": "delta sens and delta spec (naive - IPW); materiality 5 percentage points",
        "gate_G2": "naive AUROC >= 0.80",
    },
    "files_sha256": {str(f.relative_to(P) if f.is_relative_to(P) else f): sha(f) for f in FILES},
}
(P / "cohort_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=120))
print(f"frozen: {len(FILES)} files hashed -> cohort_spec.yaml; snapshot {snap.name}")
