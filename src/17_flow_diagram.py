"""Study flow diagram (STARD 2015 item 19). Counts read from the WP1/WP2/DeepECG outputs; deaths among
unverified analytic patients computed from the cohort file. Aggregates only."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

P = Path(__file__).resolve().parents[1]; OUT = P / "outputs"; FIG = OUT / "figures"
wp1 = json.loads((OUT / "wp1_summary.json").read_text()); wp2 = json.loads((OUT / "wp2_summary.json").read_text())
dg = json.loads((OUT / "wp3_deepecg.json").read_text()); e1b = json.loads((OUT / "wp3_e1b_e5.json").read_text())
fl = wp1["flow"]
k = pd.read_parquet(OUT / "wp1_cohort.parquet", columns=["subject_id", "V30", "d_v30", "died_30d"])
s = pd.read_parquet(OUT / "wp2_scores.parquet", columns=["subject_id", "technical_exclusion"])
a = k.merge(s, on="subject_id"); a = a[~a.technical_exclusion]
n_unv = int((a.V30 == 0).sum()); died = int(((a.V30 == 0) & a.died_30d).sum())
n_ver = int((a.V30 == 1).sum()); pos = int((a.d_v30 == 1).sum())
INK, INK2, EDGE = "#0b0b0b", "#52514e", "#8a8984"

fig, ax = plt.subplots(figsize=(7.2, 8.2)); ax.set_xlim(0, 101); ax.set_ylim(-2, 100); ax.axis("off")
plt.rcParams.update({"font.family": "DejaVu Sans"})


def box(x, y, w, h, text, bold=False, fs=7.4):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=EDGE, lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=0.8, shrinkA=0, shrinkB=0))


main_x = 36
box(main_x, 95, 50, 6, f"Patients with ≥1 ECG in MIMIC-IV-ECG\nfirst ECG per patient, n = {fl[0]['n_remaining']:,}")
excl = [(f"Not in MIMIC-IV or sex missing, n = {fl[1]['n_excluded']:,}", 85),
        (f"Age <18 years, n = {fl[2]['n_excluded']:,}", 79),
        (f"Paced ECG, n = {fl[3]['n_excluded']:,}", 73),
        (f"Earlier TTE with LVEF ≤45%, n = {fl[4]['n_excluded']:,}", 67),
        (f"Earlier HF or cardiomyopathy code, n = {fl[5]['n_excluded']:,}", 61),
        (f"Flat or missing lead, n = {wp2['technical_exclusion']['excluded_total']:,}", 55)]
arrow(main_x, 92, main_x, 52)
for t, y in excl:
    ax.plot([main_x, 58.5], [y, y], color=EDGE, lw=0.8)
    box(78.5, y, 40, 4.2, t, fs=6.9)
box(main_x, 48, 50, 6, f"Intended-use cohort, n = {wp2['technical_exclusion']['analytic_cohort']:,}\n(index test performed in all)", bold=True)
arrow(main_x, 45, main_x, 42)
ax.plot([18, 58], [42, 42], color=EDGE, lw=0.8)
arrow(18, 42, 18, 37); arrow(58, 42, 58, 37)
box(18, 32, 32, 9, f"TTE with LVEF within 30 days\n(verified), n = {n_ver:,}")
box(58, 32, 34, 9, f"No TTE within 30 days\n(unverified), n = {n_unv:,}\ndied within 30 days, n = {died:,}")
arrow(18, 27.5, 18, 22.5)
box(18, 17, 32, 9, f"LVEF ≤45%, n = {pos:,}\nLVEF >45%, n = {n_ver - pos:,}")
box(58, 17, 34, 9, "Reference standard missing;\naccuracy estimated by IPW\nand MNAR outcome models")
arrow(58, 27.5, 58, 21.5)
box(50, 4.0, 94, 7.5, f"Second model (DeepECG-SL): readable index ECG in n = {dg['cohort']['deepecg_readable']:,} of the intended-use cohort.\n"
    f"Published validation design (Otabor et al., replicated): separate echo-anchored cohort,\nn = {e1b['E1b_flow']['after_technical_exclusion']:,} (exploratory).")
FIG.mkdir(exist_ok=True)
for ext in ["png", "pdf"]:
    fig.savefig(FIG / f"figS1_flow.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print("written", FIG / "figS1_flow.png", "| unverified died:", died)
