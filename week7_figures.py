# -*- coding: utf-8 -*-
"""
Week 7 — Publication Figures
Synaptic Degradation Thresholds and the Sequential Collapse of Engram Stability
in Alzheimer's Disease: Predictions from Attractor Network Modeling

Generates four publication figures from week5_full_results.csv and
week6_failure_sequence.csv. Outputs saved to figures/ as PNG + SVG.

Usage:
    python week7_figures.py
    python week7_figures.py --data_dir results/ --save_dir figures/
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="results/",
                    help="Directory containing week5_full_results.csv and "
                         "week6_failure_sequence.csv")
parser.add_argument("--save_dir", default="figures/",
                    help="Directory to save PNG and SVG outputs")
args, _ = parser.parse_known_args()

os.makedirs(args.save_dir, exist_ok=True)

# ============================================================
# Load Data
# ============================================================

print("Loading data...")
df5 = pd.read_csv(os.path.join(args.data_dir, "week5_full_results.csv"))
df6 = pd.read_csv(os.path.join(args.data_dir, "week6_failure_sequence.csv"))
print(f"  week5_full_results  : {df5.shape}")
print(f"  week6_failure_sequence : {df6.shape}")

# ============================================================
# Style
# ============================================================

plt.rcParams.update({
    "font.family":                   "sans-serif",
    "font.sans-serif":               ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":                     9,
    "axes.titlesize":                10,
    "axes.labelsize":                9,
    "xtick.labelsize":               8,
    "ytick.labelsize":               8,
    "legend.fontsize":               8,
    "legend.title_fontsize":         9,
    "lines.linewidth":               1.6,
    "lines.markersize":              5,
    "patch.linewidth":               0.8,
    "axes.linewidth":                0.8,
    "axes.spines.top":               False,
    "axes.spines.right":             False,
    "axes.grid":                     True,
    "grid.linewidth":                0.4,
    "grid.alpha":                    0.4,
    "grid.color":                    "#CCCCCC",
    "xtick.direction":               "out",
    "ytick.direction":               "out",
    "xtick.major.width":             0.8,
    "ytick.major.width":             0.8,
    "xtick.major.size":              3.5,
    "ytick.major.size":              3.5,
    "xtick.minor.visible":           False,
    "ytick.minor.visible":           False,
    "legend.frameon":                True,
    "legend.framealpha":             0.9,
    "legend.edgecolor":              "#CCCCCC",
    "legend.borderpad":              0.5,
    "figure.dpi":                    300,
    "savefig.dpi":                   300,
    "savefig.bbox":                  "tight",
    "savefig.transparent":           False,
    "figure.constrained_layout.use": True,
})

PALETTE = {
    "weak_remote":    "#E69F00",
    "weak_dense":     "#56B4E9",
    "strong_recent":  "#009E73",
    "strong_salient": "#CC79A7",
}
CLASS_ORDER = ["weak_remote", "weak_dense", "strong_recent", "strong_salient"]
CLASS_LABELS = {
    "weak_remote":    "Weak-Remote (Episodic)",
    "weak_dense":     "Weak-Dense (Semantic)",
    "strong_recent":  "Strong-Recent (Working)",
    "strong_salient": "Strong-Salient (Procedural/Emotional)",
}
R_STARS = {
    "weak_remote":    0.756,
    "weak_dense":     0.883,
    "strong_recent":  0.964,
    "strong_salient": 0.983,
}

# ============================================================
# Helpers
# ============================================================

def ci95(series):
    n  = len(series)
    se = stats.sem(series)
    h  = se * stats.t.ppf(0.975, df=n - 1)
    return series.mean() - h, series.mean() + h


def save_fig(fig, stem):
    for fmt in ("png", "svg"):
        path = os.path.join(args.save_dir, f"{stem}.{fmt}")
        fig.savefig(path, dpi=300 if fmt == "png" else None, bbox_inches="tight")
        print(f"  Saved: {path}")


# ============================================================
# Figure 1 — Retrieval Fidelity Trajectories
# ============================================================
# F(3, 536) confirmed from df5: 540 rows, 4 classes → df_within = 536.
# η² = 0.987 from original ANOVA on the sweep data.

print("\nFigure 1 — Fidelity Trajectories...")

fig1, ax = plt.subplots(figsize=(6.8, 3.8))

ax.axvspan(0.00, 0.33, color="#DDEEFF", alpha=0.35, zorder=0)
ax.axvspan(0.33, 0.66, color="#FFF3CC", alpha=0.35, zorder=0)
ax.axvspan(0.66, 1.00, color="#FFE0E0", alpha=0.35, zorder=0)

for x, label, col in [
    (0.165, "Stage 1\nTransentorhinal", "#5577AA"),
    (0.495, "Stage 2\nLimbic",           "#AA8800"),
    (0.830, "Stage 3\nIsocortical",      "#AA4444"),
]:
    ax.text(x, 1.04, label, ha="center", va="bottom", fontsize=6.5,
            color=col, transform=ax.get_xaxis_transform())

ax.axhline(0.50, color="#888888", linewidth=0.9, linestyle="--",
           zorder=1, label="Collapse threshold (0.50)")

for cls in CLASS_ORDER:
    sub    = df5[df5["Memory_Class"] == cls].copy()
    x_vals = np.array(sorted(sub["r_star"].unique()))
    means, lo, hi = [], [], []
    for x in x_vals:
        vals = sub[sub["r_star"] == x]["fidelity_at_r"]
        means.append(vals.mean())
        l, h = ci95(vals)
        lo.append(l)
        hi.append(h)
    means = np.array(means)
    lo    = np.array(lo)
    hi    = np.array(hi)

    ax.plot(x_vals, means, color=PALETTE[cls], linewidth=1.8,
            label=CLASS_LABELS[cls], zorder=3)
    ax.fill_between(x_vals, lo, hi, color=PALETTE[cls], alpha=0.15,
                    linewidth=0, zorder=2)

    r_star = R_STARS[cls]
    ax.axvline(r_star, color=PALETTE[cls], linewidth=0.7,
               linestyle=":", alpha=0.7, zorder=1)
    ax.text(r_star + 0.005, 0.53, f"r*={r_star}",
            color=PALETTE[cls], fontsize=6, va="bottom", rotation=90)

ax.set_xlim(0.0, 1.0)
ax.set_ylim(-0.05, 1.15)
ax.set_xlabel("Synaptic Degradation Parameter ($r$)")
ax.set_ylabel("Mean Retrieval Fidelity (cosine)")
ax.set_title(
    "Figure 1.  Retrieval Fidelity Trajectories under Simulated Braak-Stage Degradation",
    pad=10, fontweight="normal", loc="left",
)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.legend(loc="lower left", title="Memory Class", framealpha=0.92)
ax.text(
    0.98, 0.98,
    "$F$(3,536) = 13,187.93\n$p$ < 2.2×10⁻¹⁶,  η² = 0.987",
    transform=ax.transAxes, fontsize=7, ha="right", va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9),
)
save_fig(fig1, "fig1_fidelity_curves")
plt.show()

# ============================================================
# Figure 2 — Sensitivity Panel (2×2)
# ============================================================
# Panel C: Pearson r = 0.670 (confirmed from data; original script had 0.658).
# All other annotations confirmed.

print("\nFigure 2 — Sensitivity Panel...")

fig2, axes       = plt.subplots(2, 2, figsize=(6.8, 5.6))
short_labels     = ["Weak\nRemote", "Weak\nDense", "Strong\nRecent", "Strong\nSalient"]

# Panel A: Post-collapse floor
ax = axes[0, 0]
floors_mean = df5.groupby("Memory_Class")["post_floor"].mean().reindex(CLASS_ORDER)
floors_ci   = [ci95(df5[df5["Memory_Class"] == c]["post_floor"]) for c in CLASS_ORDER]

ax.bar(short_labels, floors_mean, color=[PALETTE[c] for c in CLASS_ORDER],
       width=0.55, edgecolor="white", linewidth=0.5, zorder=3)
for i, (lo, hi) in enumerate(floors_ci):
    ax.errorbar(i, floors_mean.iloc[i],
                yerr=[[floors_mean.iloc[i] - lo], [hi - floors_mean.iloc[i]]],
                fmt="none", color="#333333", capsize=3, linewidth=0.8, zorder=4)
ax.set_ylabel("Post-Collapse Floor (fidelity)")
ax.set_title("A.  Post-Collapse Floor by Memory Class",
             fontweight="normal", loc="left", pad=4)
ax.set_ylim(0, 0.65)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

# Panel B: AUC by N and class
ax       = axes[0, 1]
n_vals   = sorted(df5["N"].unique())
x        = np.arange(len(CLASS_ORDER))
width    = 0.22
n_colors = ["#BBBBBB", "#777777", "#333333"]

for i, (n, nc) in enumerate(zip(n_vals, n_colors)):
    sub   = df5[df5["N"] == n]
    means = [sub[sub["Memory_Class"] == c]["auc"].mean() for c in CLASS_ORDER]
    cis   = [ci95(sub[sub["Memory_Class"] == c]["auc"]) for c in CLASS_ORDER]
    offs  = x + (i - 1) * width
    ax.bar(offs, means, width, color=nc, label=f"$N$ = {n}",
           edgecolor="white", linewidth=0.4, zorder=3)
    for j, (lo, hi) in enumerate(cis):
        ax.errorbar(offs[j], means[j],
                    yerr=[[means[j] - lo], [hi - means[j]]],
                    fmt="none", color="#222222", capsize=2, linewidth=0.7, zorder=4)

ax.set_xticks(x)
ax.set_xticklabels(short_labels)
ax.set_ylabel("Area Under Fidelity Curve (AUC)")
ax.set_title("B.  AUC by Network Size and Memory Class",
             fontweight="normal", loc="left", pad=4)
ax.set_ylim(0.7, 1.02)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.legend(title="Network size", fontsize=7, title_fontsize=7)
ax.text(
    0.98, 0.04,
    "Interaction $F$(6,528)=131.92\n$p$=1.23×10⁻¹⁰¹,  η²=0.600",
    transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9),
)

# Panel C: Transition width by degradation rate
ax        = axes[1, 0]
deg_map   = {"slow": 0.025, "medium": 0.050, "fast": 0.100}
deg_order = sorted(df5["Deg_Rate"].unique(),
                   key=lambda d: deg_map[d])
deg_labels = {"fast": "Fast\n(0.100)", "medium": "Medium\n(0.050)",
              "slow": "Slow\n(0.025)"}

for cls in CLASS_ORDER:
    sub   = df5[df5["Memory_Class"] == cls]
    means = [sub[sub["Deg_Rate"] == d]["transition_width"].mean()
             for d in deg_order]
    ax.plot(range(len(deg_order)), means, color=PALETTE[cls], marker="o",
            markersize=5, linewidth=1.4, label=CLASS_LABELS[cls])

ax.set_xticks(range(len(deg_order)))
ax.set_xticklabels([deg_labels[d] for d in deg_order])
ax.set_ylabel("Transition Width ($r$-units)")
ax.set_title("C.  Collapse Sharpness by Degradation Rate",
             fontweight="normal", loc="left", pad=4)
# Pearson r confirmed from data as 0.670
ax.text(
    0.05, 0.92, "Pearson $r$ = +0.670\n(Deg_Rate → Trans_Width)",
    transform=ax.transAxes, fontsize=6.5, va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9),
)

# Panel D: Noise effect on plateau (weak_remote)
ax         = axes[1, 1]
noise_vals = sorted(df5["Noise_Std"].unique())
wr         = df5[df5["Memory_Class"] == "weak_remote"]

means_n = [wr[wr["Noise_Std"] == n]["plateau_height"].mean() for n in noise_vals]
cis_n   = [ci95(wr[wr["Noise_Std"] == n]["plateau_height"]) for n in noise_vals]

ax.plot(noise_vals, means_n, color=PALETTE["weak_remote"], marker="o",
        markersize=5, linewidth=1.4, label="Weak-Remote (most sensitive)")
for i, (lo, hi) in enumerate(cis_n):
    ax.errorbar(noise_vals[i], means_n[i],
                yerr=[[means_n[i] - lo], [hi - means_n[i]]],
                fmt="none", color=PALETTE["weak_remote"], capsize=3, linewidth=0.8)

for cls in ["weak_dense", "strong_recent", "strong_salient"]:
    ax.axhline(1.0, color=PALETTE[cls], linewidth=0.9, linestyle="--",
               alpha=0.6, label=CLASS_LABELS[cls])

ax.set_xlabel("Retrieval Noise ($\\sigma$)")
ax.set_ylabel("Pre-Collapse Plateau Height")
ax.set_title("D.  Noise Effect on Plateau (Weak-Remote most sensitive)",
             fontweight="normal", loc="left", pad=4)
ax.set_ylim(0.990, 1.005)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.003))
ax.text(
    0.98, 0.06,
    "Max Δ = 0.4% at σ=0.5\n$t$(52)=4.06,  $p_{Bonf}$=0.003",
    transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9),
)

fig2.suptitle(
    "Figure 2.  Parameter Sensitivity across 135 Simulation Configurations",
    fontsize=10, fontweight="normal", x=0.0, ha="left", y=1.01,
)
save_fig(fig2, "fig2_sensitivity_panel")
plt.show()

# ============================================================
# Figure 3 — Slopegraph + ADNI Trajectories
# ============================================================
# All values hardcoded from confirmed week6_failure_sequence.csv.
# rho = 0.50, p_min = 0.167 (minimum achievable p at n=3).

print("\nFigure 3 — Slopegraph + ADNI Trajectories...")

clinical = {
    "Episodic Memory": {
        "slope": -0.02487, "ci_lo": -0.03390, "ci_hi": -0.01584,
        "model_rank": 1, "emp_rank": 1, "class": "weak_remote",
    },
    "Working Memory": {
        "slope": -0.02420, "ci_lo": -0.03382, "ci_hi": -0.01458,
        "model_rank": 3, "emp_rank": 2, "class": "strong_recent",
    },
    "Semantic Memory": {
        "slope": -0.02184, "ci_lo": -0.02994, "ci_hi": -0.01375,
        "model_rank": 2, "emp_rank": 3, "class": "weak_dense",
    },
}

fig3, (ax_slope, ax_traj) = plt.subplots(
    1, 2, figsize=(6.8, 4.0),
    gridspec_kw={"width_ratios": [1, 1.6]},
)

# Panel A: Slopegraph
ax = ax_slope
for domain, vals in clinical.items():
    color = PALETTE[vals["class"]]
    mr, er = vals["model_rank"], vals["emp_rank"]
    ax.plot([0, 1], [mr, er], color=color, linewidth=2.0,
            solid_capstyle="round", zorder=3)
    ax.scatter([0, 1], [mr, er], color=color, s=70, zorder=4)
    ax.text(-0.06, mr, f"Rank {mr}", ha="right", va="center",
            fontsize=8, color=color)
    ax.text(1.06, er, f"Rank {er}", ha="left", va="center",
            fontsize=8, color=color)
    match = "✓" if mr == er else "↕"
    ax.text(0.5, (mr + er) / 2 + 0.09, f"{domain}\n{match}",
            ha="center", va="bottom", fontsize=7, color=color)

ax.set_xlim(-0.55, 1.55)
ax.set_ylim(3.7, 0.3)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Model\nPredicted", "ADNI\nEmpirical"], fontsize=8)
ax.set_yticks([])
ax.set_title("A.  Rank Alignment", fontweight="normal", loc="left", pad=4)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.grid(False)
ax.text(0.5, 0.02,
        "$\\rho_s$ = 0.500  ($n$=3,  $p_{min}$=0.167)",
        ha="center", va="bottom", fontsize=7, style="italic",
        transform=ax.transAxes)

# Panel B: ADNI decline trajectories
ax     = ax_traj
months = np.linspace(-36, 12, 200)

for domain, vals in clinical.items():
    color    = PALETTE[vals["class"]]
    slope    = vals["slope"]
    ci_lo    = vals["ci_lo"]
    ci_hi    = vals["ci_hi"]
    line     = slope * months
    lo_band  = np.minimum(ci_lo * months, ci_hi * months)
    hi_band  = np.maximum(ci_lo * months, ci_hi * months)
    ax.plot(months, line, color=color, linewidth=1.6,
            label=f"{domain}  ($M$={slope:.3f} z/mo)", zorder=3)
    ax.fill_between(months, lo_band, hi_band,
                    color=color, alpha=0.13, linewidth=0, zorder=2)

ax.axvline(0, color="#555555", linewidth=0.9, linestyle="--",
           zorder=1, label="AD conversion (month 0)")
ax.axhline(0, color="#AAAAAA", linewidth=0.5, linestyle=":", zorder=1)
ax.set_xlabel("Months Relative to AD Conversion")
ax.set_ylabel("Cognitive Score ($z$-units)")
ax.set_title(
    "B.  ADNI Longitudinal Decline Trajectories\n($n$ = 191 MCI→AD Converters)",
    fontweight="normal", loc="left", pad=4,
)
ax.set_xlim(-36, 12)
ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
ax.legend(loc="lower left", fontsize=7, framealpha=0.92)

fig3.suptitle(
    "Figure 3.  Clinical Validation: Model Predictions vs ADNI Empirical Decline",
    fontsize=10, fontweight="normal", x=0.0, ha="left", y=1.01,
)
save_fig(fig3, "fig3_slopegraph_adni")
plt.show()

# ============================================================
# Figure 4 — AUC × Network Size
# ============================================================
# AUC gains (N=500 → N=2000) confirmed from data:
#   weak_remote +12.0%, weak_dense +4.4%,
#   strong_recent +0.7%, strong_salient +2.1%
# N main effect F(2,528) confirmed: df_within = 540 - (3×4) = 528.

print("\nFigure 4 — AUC by Network Size...")

auc_gains = {
    "weak_remote":    12.0,
    "weak_dense":      4.4,
    "strong_recent":   0.7,
    "strong_salient":  2.1,
}

fig4, ax = plt.subplots(figsize=(5.0, 3.8))
n_vals   = sorted(df5["N"].unique())
x_pos    = np.arange(len(n_vals))
offsets  = np.linspace(-0.22, 0.22, len(CLASS_ORDER))

for i, cls in enumerate(CLASS_ORDER):
    sub   = df5[df5["Memory_Class"] == cls]
    means = [sub[sub["N"] == n]["auc"].mean() for n in n_vals]
    cis   = [ci95(sub[sub["N"] == n]["auc"]) for n in n_vals]
    xs    = x_pos + offsets[i]

    ax.plot(xs, means, color=PALETTE[cls], linewidth=1.4, marker="o",
            markersize=6, label=CLASS_LABELS[cls], zorder=3)
    for j, (lo, hi) in enumerate(cis):
        ax.errorbar(xs[j], means[j],
                    yerr=[[means[j] - lo], [hi - means[j]]],
                    fmt="none", color=PALETTE[cls], capsize=3,
                    linewidth=0.8, zorder=4)

    ax.annotate(f"+{auc_gains[cls]}%",
                xy=(xs[-1], means[-1]),
                xytext=(xs[-1] + 0.04, means[-1] + 0.003),
                fontsize=6.5, color=PALETTE[cls], va="bottom")

ax.set_xticks(x_pos)
ax.set_xticklabels([f"$N$ = {n}" for n in n_vals])
ax.set_xlabel("Network Size ($N$)")
ax.set_ylabel("Area Under Fidelity Curve (AUC)")
ax.set_title(
    "Figure 4.  Network Size Preferentially Benefits Most Vulnerable Memory Class",
    fontweight="normal", loc="left", pad=6,
)
ax.legend(loc="lower right", title="Memory Class", fontsize=7,
          title_fontsize=7, framealpha=0.92)
ax.text(
    0.02, 0.06,
    "$N$ main effect: $F$(2,528)=655.00\n$p$=9.73×10⁻¹⁴⁴,  η²=0.713",
    transform=ax.transAxes, fontsize=7, va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9),
)

save_fig(fig4, "fig4_auc_by_N")
plt.show()

print("\n" + "=" * 55)
print("All four figures complete.")
print(f"Saved to: {args.save_dir}")
print("=" * 55)
