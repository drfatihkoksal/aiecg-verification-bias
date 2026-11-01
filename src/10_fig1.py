"""Figure 1 (static) from outputs/fig1_data.json (aggregate data only).

A  Verification gradient: V30 by AI-score decile (95% Wilson CI).
B  ROC curves: published validation design (Otabor et al. 2026, replicated; exploratory E1b), naive (verified
   only), IPW (primary),
   with operating points at the frozen threshold and the EchoNext-Mini source-split operating point.
C  Sensitivity and specificity across the MNAR lambda range, with naive and IPW references.
D  Second model (exploratory, concept.md §6.8): sensitivity and specificity under the published design, naive and
   IPW for EchoNext-Mini (frozen threshold) and DeepECG-SL (matched-positivity threshold), 95% bootstrap CIs.
   Read from outputs/wp3_results.json, wp3_e1b_e5.json and wp3_deepecg.json (aggregates only).
Colour = estimator identity throughout (validated palette slots 1-3, all-pairs PASS; slot 3 contrast WARN ->
direct labels + distinct line styles and marker shapes as secondary encoding).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "outputs"
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
F = json.loads((OUT / "fig1_data.json").read_text())
E = json.loads((OUT / "fig1_e1b_roc.json").read_text())  # E1b replaces the +-30-day E1 comparator
F["roc"]["conventional"], F["operating_points"]["conventional"] = E["roc"], E["operating_point"]
F["auroc"]["conventional"], F["n"]["conventional"] = E["auroc"], E["n"]

IPW, NAIVE, CONV = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, BAR = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#9c9a94"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "legend.fontsize": 7})
LW = 1.5


def grid(ax):
    ax.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


fig = plt.figure(figsize=(7.2, 8.4))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.62], hspace=0.62, wspace=0.3)

# ---------------- A
ax = fig.add_subplot(gs[0, 0]); grid(ax)
g = F["verification_gradient"]
x = np.array([r["decile"] for r in g]); y = np.array([r["v30_pct"] for r in g])
lo = np.array([r["lo"] for r in g]); hi = np.array([r["hi"] for r in g])
ax.bar(x, y, width=0.82, color=BAR, edgecolor="white", linewidth=1.0, zorder=2)
ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.8, zorder=3)
# threshold position on the decile axis: inside the first decile that contains any model-positive ECG
k = next(i for i, r in enumerate(g) if r["pct_test_positive"] > 0)
xthr = x[k] - 0.5 + (1 - g[k]["pct_test_positive"] / 100)
ax.axvline(xthr, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
ax.text(xthr - 0.12, max(hi) * 1.13, "frozen threshold", color=INK2, fontsize=6.5, va="center", ha="right")
for xi, yi in [(x[0], y[0]), (x[-1], y[-1])]:
    ax.text(xi, hi[list(x).index(xi)] + 1.0, f"{yi:.1f}%", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.set_xticks(x); ax.set_xlabel("AI-score decile (EchoNext-Mini, LVEF ≤45%)")
ax.set_ylabel("Echo within 30 days (%)"); ax.set_ylim(0, max(hi) * 1.22)
ax.set_title("A   Echo referral rises with the model's score")

# ---------------- B
ax = fig.add_subplot(gs[0, 1]); grid(ax)
ax.plot([0, 1], [0, 1], color=GRID, linewidth=0.8, zorder=1)
sty = {"conventional": (CONV, (0, (5, 1.5, 1, 1.5)), "s", "Published design*"),
       "naive": (NAIVE, (0, (4, 2)), "o", "Naive"),
       "ipw": (IPW, "-", "D", "Corrected (IPW)")}
for k, (col, ls, mk, lab) in sty.items():
    r = F["roc"][k]
    ax.plot(r["fpr"], r["tpr"], color=col, linestyle=ls, linewidth=LW, zorder=2, label=f"{lab}  {F['auroc'][k]:.1f}")
    op = F["operating_points"][k]
    ax.plot(op["fpr"], op["tpr"], marker=mk, markersize=6, color=col, markeredgecolor="white", markeredgewidth=1.0, zorder=4, linestyle="none")
sp = F["operating_points"]["source_split"]
ax.plot(sp["fpr"], sp["tpr"], marker="*", markersize=8.5, color=INK2, markeredgecolor="white", markeredgewidth=0.8, zorder=4, linestyle="none",
        label="Developer (source split)")
lab_off = {"conventional": (-0.2, 0.14), "naive": (0.07, -0.02), "ipw": (0.06, -0.09)}
for k in ["conventional", "naive", "ipw"]:
    op = F["operating_points"][k]
    ax.annotate(f"{100 * op['tpr']:.0f}% / {100 * (1 - op['fpr']):.0f}%", (op["fpr"], op["tpr"]),
                xytext=(op["fpr"] + lab_off[k][0], op["tpr"] + lab_off[k][1]), fontsize=6.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5))
ax.set_xlim(0, 1); ax.set_ylim(0, 1.0); ax.set_aspect("equal")
ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity")
ax.legend(loc="lower right", handlelength=2.4, borderaxespad=0.1, fontsize=6.3, labelspacing=0.3,
          title="AUROC; labels = sens / spec", title_fontsize=6.3)
ax.set_title("B   Similar curve, different operating point")

# ---------------- C
lam = np.array([r["lambda"] for r in F["mnar"]])
for j, (m, name) in enumerate([("sens", "Sensitivity"), ("spec", "Specificity")]):
    ax = fig.add_subplot(gs[1, j]); grid(ax)
    nv = F["naive"][m]["est"]
    yv = np.array([r[m]["est"] for r in F["mnar"]]); ylo = np.array([r[m]["ci"][0] for r in F["mnar"]]); yhi = np.array([r[m]["ci"][1] for r in F["mnar"]])
    ax.axhspan(nv - 5, nv + 5, color=GRID, alpha=0.7, linewidth=0, zorder=0)
    ax.axhline(nv, color=NAIVE, linestyle=(0, (4, 2)), linewidth=LW, zorder=2)
    ax.fill_between(lam, ylo, yhi, color=IPW, alpha=0.15, linewidth=0, zorder=1)
    ax.plot(lam, yv, color=IPW, linewidth=LW, marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ip = F["ipw"][m]
    ax.errorbar([1.0], [ip["est"]], yerr=[[ip["est"] - ip["ci"][0]], [ip["ci"][1] - ip["est"]]], fmt="D", color=IPW,
                markersize=6, markeredgecolor="white", markeredgewidth=1.0, elinewidth=0.9, capsize=2, zorder=4)
    ax.set_xscale("log"); ax.set_xticks(lam); ax.set_xticklabels([f"{v:g}" for v in lam]); ax.minorticks_off()
    ax.set_xlabel("λ  (odds of LVSD in unverified vs verified, given X)")
    ax.set_ylabel(f"{name} (%)")
    ax.text(lam[-1], nv + (1.2 if m == "sens" else -1.2), f"naive {nv:.1f}", color=INK, fontsize=6.5, ha="right",
            va="bottom" if m == "sens" else "top")
    ax.text(lam[0], nv + 5 - 0.4, "±5-point materiality band", color=INK2, fontsize=6, va="top")
    ax.annotate(f"IPW {ip['est']:.1f}", (1.0, ip["est"]), xytext=(1.12, ip["est"] + (3.0 if m == "sens" else -2.4)), fontsize=6.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5))
    if m == "sens":
        ax.set_ylim(35, 80)
        ax.set_title("C   Under MNAR: sensitivity range, specificity stable")
    else:
        ax.set_ylim(70, 98)
    ax.text(0.02, 0.03, "MNAR outcome-model estimate (95% CI)", transform=ax.transAxes, fontsize=6, color=INK2)

# ---------------- D  second model
R7 = json.loads((OUT / "wp3_results.json").read_text())["main"]
RB = json.loads((OUT / "wp3_e1b_e5.json").read_text())["E1b_primary_rule"]
RD = json.loads((OUT / "wp3_deepecg.json").read_text())
DE = {"EchoNext-Mini": {"pub": RB, "naive": R7["naive"], "ipw": R7["ipw"]},
      "DeepECG-SL": {"pub": RD["E1b_published_design"]["matched"], "naive": RD["intended_use"]["matched_naive"],
                     "ipw": RD["intended_use"]["matched_ipw"]}}
EST = [("pub", CONV, "s", "Published design*", 0.2), ("naive", NAIVE, "o", "Naive (referred only)", 0.0),
       ("ipw", IPW, "D", "Corrected (IPW)", -0.2)]
models = list(DE)
for j, (m, name, lim) in enumerate([("sens", "Sensitivity (%)", (45, 87)), ("spec", "Specificity (%)", (72, 95))]):
    ax = fig.add_subplot(gs[2, j]); grid(ax); ax.grid(False, axis="y")
    for yi, mod in enumerate(models[::-1]):
        for key, col, mk, lab, off in EST:
            v = DE[mod][key][m]
            ax.errorbar([v["est"]], [yi + off], xerr=[[v["est"] - v["ci"][0]], [v["ci"][1] - v["est"]]], fmt=mk, color=col,
                        markersize=5.5, markeredgecolor="white", markeredgewidth=0.8, elinewidth=0.9, capsize=1.6, zorder=3,
                        label=lab if (yi == 0 and j == 0) else None)
        dlt = DE[mod]["naive"][m]["est"] - DE[mod]["ipw"][m]["est"]
        tx, ha = (lim[1], "right") if m == "sens" else (lim[0] + 0.3, "left")  # keep the label clear of the IPW marker
        ax.text(tx, yi - 0.36, f"Δ naive − IPW {dlt:+.1f}".replace("-", "−"), ha=ha, va="center", fontsize=6.2, color=INK)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models[::-1] if j == 0 else [""] * len(models))
    ax.set_ylim(-0.55, len(models) - 0.45); ax.set_xlim(*lim); ax.set_xlabel(name)
    ax.tick_params(axis="y", length=0)
    if j == 0:
        ax.set_title("D   Same pattern in a second, independent model", pad=17)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=6.2, handletextpad=0.3, columnspacing=1.2,
                  borderaxespad=0.1, frameon=False)

n = F["n"]
fig.text(0.01, 0.005, f"Intended-use cohort n = {n['cohort']:,} (verified {n['verified']:,}); *published validation design = Otabor et al. 2026, replicated (n = {n['conventional']:,}; exploratory).\n"
         "Colour = estimator: blue corrected (IPW / MNAR), orange naive (referred only), green published design.\n"
         "D: DeepECG-SL (exploratory) at the threshold flagging the same share of the cohort as EchoNext-Mini (predicted LVEF ≤55.3%).", fontsize=6, color=INK2)
for ext in ["png", "pdf"]:
    fig.savefig(FIG / f"fig1.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print("written", FIG / "fig1.png")
