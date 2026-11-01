"""Freeze the second-model analysis (concept.md §6.8) BEFORE any DeepECG score is computed in this project.
Writes cohort_spec_addendum_deepecg.yaml and a verbatim concept snapshot."""
import hashlib, importlib.util, shutil, sys, datetime as dt
from pathlib import Path
import yaml

P = Path(__file__).resolve().parents[1]
RS = Path("/home/fatih/projects/rise-spec")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


spec = importlib.util.spec_from_file_location("deepecg_check", RS / "analysis/14_deepecg_check.py")
chk = importlib.util.module_from_spec(spec); sys.modules["deepecg_check"] = chk; spec.loader.exec_module(chk)
snap = P / f"concept_frozen_deepecg_{dt.date.today().isoformat()}.md"
assert not (P / "outputs/wp3_deepecg_scores.parquet").exists(), "DeepECG scores already exist; freeze must precede scoring"
shutil.copyfile(P / "concept.md", snap)
files = [snap, P / "src/12_deepecg.py", P / "src/11_e1b_e5.py", P / "src/wp3_design.py", P / "cohort_spec.yaml",
         RS / "analysis/14_deepecg_check.py", RS / "analysis/15_deepecg_score.py", RS / "results/wp_d_g2.json", Path(chk.WEIGHTS)]
spec_out = {
    "addendum_to": "verify_gap_spec_v2 (cohort_spec.yaml)",
    "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
    "frozen_before": "any DeepECG-SL score computed in this project",
    "status": "exploratory, added after the primary results were known (concept.md §6.8)",
    "model": "heartwise/deepecg-sl_finetuned_LVEF_MHI; rise-spec loader and preprocessing; MIMIC parameters frozen 30 Oct 2026",
    "operating_points": {"primary": "matched positivity: predicted LVEF <= value flagging the EchoNext-Mini share of the intended-use analytic cohort (scores only)",
                         "secondary": "predicted LVEF <= 45%"},
    "estimators": "as spec v2; verification model with the DeepECG score spline; naive, IPW (99th pct truncation), MNAR lambda grid, 1000 bootstrap resamples (seed 20261104)",
    "comparator": "E1b (Otabor et al. 2026 design) pairs",
    "descriptive_gate": "naive AUROC < 0.70 -> descriptive only",
    "prior_knowledge": "rise-spec G2-D (prevalent LVSD code) AUROC 0.886; predicted LVEF quartiles 58.1/59.8/60.6",
    "files_sha256": {str(f.relative_to(P) if f.is_relative_to(P) else f): sha(f) for f in files},
}
(P / "cohort_spec_addendum_deepecg.yaml").write_text(yaml.safe_dump(spec_out, sort_keys=False, allow_unicode=True, width=120))
print("frozen:", len(files), "files; snapshot", snap.name)
