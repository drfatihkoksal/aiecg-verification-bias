"""Source-anchored operating threshold for EchoNext-Mini `lvef_lte_45` (concept.md §6.1, rule a).

Rule = the EchoNext-Mini authors' own rule [12], as implemented in
`IntroECG/7-EchoNext Minimodel/EchoNext_Mini_example.ipynb`, cell 19:
    roc = roc_curve(y, yh)  (sklearn defaults)
    cutoff = first ROC point with sensitivity >= 0.70
    positive = yh >= cutoff
applied to the EchoNext TEST split. Touches NO MIMIC data.

Sanity check: the same rule on the composite SHD output must reproduce the published
70.1 / 77.9 / 70.2 / 77.8 (sens / spec / PPV / NPV) [12].
"""
import sys, json, hashlib, datetime
import numpy as np, pandas as pd, torch, sklearn
from sklearn.metrics import roc_curve, roc_auc_score
from pathlib import Path

EN = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c/echonext")
MD = Path("/home/fatih/projects/afef/models/echonext_minimodel")
CODE = "/home/fatih/projects/afef/code/echonext"
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)
TARGET_SENS = 0.70
LABELS = ['lvef_lte_45_flag', 'lvwt_gte_13_flag', 'aortic_stenosis_moderate_or_greater_flag',
          'aortic_regurgitation_moderate_or_greater_flag', 'mitral_regurgitation_moderate_or_greater_flag',
          'tricuspid_regurgitation_moderate_or_greater_flag', 'pulmonary_regurgitation_moderate_or_greater_flag',
          'rv_systolic_dysfunction_moderate_or_greater_flag', 'pericardial_effusion_moderate_large_flag',
          'pasp_gte_45_flag', 'tr_max_gte_32_flag', 'shd_moderate_or_greater_flag']
IDX = {"lvef_lte_45_flag": 0, "shd_moderate_or_greater_flag": 11}

sys.path.insert(0, CODE)
from cradlenet.models.resnet1d_tabular import ResNet1dWithTabular

sha = hashlib.sha256((MD / "weights.pt").read_bytes()).hexdigest()
sd = torch.load(MD / "weights.pt", map_location="cpu", weights_only=False)["model"]
model = ResNet1dWithTabular(len_tabular_feature_vector=7, filter_size=16, num_classes=12)
model.load_state_dict(sd, strict=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.eval().to(dev)

wf = np.load(EN / "EchoNext_test_waveforms.npy", mmap_mode="r")
tab = np.load(EN / "EchoNext_test_tabular_features.npy")
meta = pd.read_csv(EN / "echonext_metadata_100k.csv")
meta = meta[meta.split == "test"].reset_index(drop=True)
assert len(meta) == wf.shape[0] == tab.shape[0]

probs = []
with torch.no_grad():
    for i in range(0, len(wf), 256):
        x = torch.from_numpy(np.ascontiguousarray(wf[i:i + 256])).float().to(dev)
        t = torch.from_numpy(tab[i:i + 256]).float().to(dev)
        probs.append(torch.sigmoid(model((x, t))).cpu().numpy())
P = np.concatenate(probs)


# --- metric helpers, mirroring the notebook (percent scale, bootstrap n=1000, rng seed 0) ---
def sensitivity(y, yh): return (yh & y).sum() / y.sum()
def specificity(y, yh): return (~yh & ~y).sum() / (~y).sum()
def ppv(y, yh): return (yh & y).sum() / yh.sum()
def npv(y, yh): return (~yh & ~y).sum() / (~yh).sum()


def boot(y, yh, fn, n=1000):
    y, yh = np.asarray(y, bool), np.asarray(yh, bool)
    est = 100 * fn(y, yh)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y))
        bs.append(100 * fn(y[idx], yh[idx]))
    return [round(est, 1), round(np.percentile(bs, 2.5), 1), round(np.percentile(bs, 97.5), 1)]


def operating_point(label, drop_intermediate=True):
    y = meta[label].values.astype(bool)
    yh = P[:, IDX[label]]
    fpr, tpr, thr = roc_curve(y, yh, drop_intermediate=drop_intermediate)
    cutoff = float(thr[np.argmax(tpr >= TARGET_SENS)])  # first point with sens >= 0.70
    pos = yh >= cutoff
    return {
        "label": label, "n": int(len(y)), "positives": int(y.sum()),
        "auroc": round(100 * roc_auc_score(y, yh), 1),
        "cutoff_prob": cutoff,
        "cutoff_logit": float(np.log(cutoff / (1 - cutoff))),
        "cutoff_score_0_100": round(100 * cutoff, 4),
        "flagged_pct": round(100 * pos.mean(), 1),
        "sensitivity": boot(y, pos, sensitivity), "specificity": boot(y, pos, specificity),
        "ppv": boot(y, pos, ppv), "npv": boot(y, pos, npv),
    }


shd = operating_point("shd_moderate_or_greater_flag")
lv = operating_point("lvef_lte_45_flag")
lv_nodrop = operating_point("lvef_lte_45_flag", drop_intermediate=False)
published = {"sensitivity": 70.1, "specificity": 77.9, "ppv": 70.2, "npv": 77.8}
check = all(shd[k][0] == v for k, v in published.items())

res = {
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "rule": "EchoNext-Mini notebook cell 19: first roc_curve point with sensitivity >= 0.70, positive = score >= cutoff; EchoNext test split",
    "weights_sha256": sha, "device": dev,
    "versions": {"torch": torch.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
    "sanity_check_composite_shd": {"result": shd, "published": published, "reproduced": check},
    "lvef_lte_45": lv,
    "lvef_lte_45_drop_intermediate_false": {"cutoff_prob": lv_nodrop["cutoff_prob"],
                                            "same_cutoff": lv_nodrop["cutoff_prob"] == lv["cutoff_prob"]},
}
(OUT / "source_threshold.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
