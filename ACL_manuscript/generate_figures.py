"""
Generate all six paper figures for kcd_arxiv.tex.
Run from ACL_manuscript/:  python generate_figures.py
Outputs PDFs to figures/  (created if absent).
Requires: matplotlib, numpy, scipy
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrow
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec
from scipy.stats import norm

# ── output dir ──────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ── global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#cccccc",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#e8e8e8",
    "grid.linewidth":     0.5,
    "lines.linewidth":    1.6,
    "lines.markersize":   5.5,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# ── shared palette ────────────────────────────────────────────────────────────
C = {
    "uniform":   "#888787",
    "recency":   "#4e79a7",
    "rec_kcd":   "#76b7b2",
    "tfidf":     "#59a14f",
    "lllm":      "#e15759",   # LongLLMLingua
    "geo":       "#4e79a7",
    "sem":       "#f28e2b",
    "semqcg":    "#59a14f",
    "real":      "#4e79a7",
    "shuffled":  "#e15759",
}

BUDGETS = [0.20, 0.35, 0.50]
BUDGET_LABELS = ["0.20", "0.35", "0.50"]

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Geometry-Behavior Divergence Schematic
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_divergence_schematic():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # --- left panel: scatter plot conceptual (LongLLMLingua vs others) ---
    ax = axes[0]

    # Data points: (NLL improvement, Logit-L2 degradation) relative to uniform
    # negative NLL = better;  positive L2 = worse
    points = {
        "Uniform":        (0.0,    0.0),
        "Recency":        (-0.27,  -58.8),
        "Recency-KCD":    (-0.31,  -31.9),
        "TF-IDF":         (-0.16,  -2.3),
        "LongLLMLingua":  (-0.19,  +143.2),   # large L2 increase, small NLL gain
        "Sem-KCD":        (-0.23,  +153.8),
        "Sem-QCG":        (-0.28,  -46.4),
    }
    colors_scatter = {
        "Uniform":       C["uniform"],
        "Recency":       C["recency"],
        "Recency-KCD":   C["rec_kcd"],
        "TF-IDF":        C["tfidf"],
        "LongLLMLingua": C["lllm"],
        "Sem-KCD":       C["sem"],
        "Sem-QCG":       C["semqcg"],
    }
    markers_scatter = {
        "Uniform": "s", "Recency": "o", "Recency-KCD": "D",
        "TF-IDF": "^", "LongLLMLingua": "X", "Sem-KCD": "P", "Sem-QCG": "*",
    }
    for name, (dnll, dl2) in points.items():
        ax.scatter(dnll, dl2,
                   color=colors_scatter[name],
                   marker=markers_scatter[name],
                   s=60 if name != "Sem-QCG" else 100,
                   zorder=5, label=name)

    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")

    # quadrant labels
    ax.text(-0.44, -75, "better both (SW)", fontsize=7, color="#555555", style="italic")
    ax.text(0.04,  160, "worse geometry /\nbetter behavior (NE)", fontsize=7,
            color=C["lllm"], style="italic")

    ax.set_xlabel(r"$\Delta$ Answer NLL (↓ = better behavior)")
    ax.set_ylabel(r"$\Delta$ Logit $\ell_2$ (↑ = worse geometry)")
    ax.set_title("(a) Geometry-behavior divergence at $B=0.50$")
    ax.legend(loc="lower left", fontsize=6.5, ncol=2, handlelength=1.2,
              columnspacing=0.8, labelspacing=0.3)

    # --- right panel: conceptual dual-objective illustration ---
    ax2 = axes[1]
    budgets = np.array([0.20, 0.35, 0.50])

    # LongLLMLingua: good NLL, bad L2 (normalized to 0-1 scale for illustration)
    lllm_nll = np.array([0.85, 0.88, 0.92])    # near 1 = good
    lllm_l2  = np.array([0.22, 0.20, 0.18])    # near 0 = bad (high L2)
    # SemQCG: good both at higher budgets
    sqcg_nll = np.array([0.78, 0.87, 0.91])
    sqcg_l2  = np.array([0.80, 0.84, 0.98])

    ax2.plot(budgets, lllm_nll, color=C["lllm"],   marker="X", label="LLMLingua NLL (↑)")
    ax2.plot(budgets, lllm_l2,  color=C["lllm"],   marker="X", linestyle="--",
             label="LLMLingua Geom (↑)")
    ax2.plot(budgets, sqcg_nll, color=C["semqcg"], marker="*", label="Sem-QCG NLL (↑)")
    ax2.plot(budgets, sqcg_l2,  color=C["semqcg"], marker="*", linestyle="--",
             label="Sem-QCG Geom (↑)")

    ax2.set_xlabel("Budget fraction $B$")
    ax2.set_ylabel("Relative score (↑ better)")
    ax2.set_title("(b) Dual-objective trade-off (schematic)")
    ax2.set_xticks(budgets)
    ax2.set_xticklabels(BUDGET_LABELS)
    ax2.legend(fontsize=6.5, ncol=1, loc="lower right")

    # annotation
    ax2.annotate("Geometry collapses\nfor LLMLingua",
                 xy=(0.50, 0.18), xytext=(0.36, 0.10),
                 fontsize=7, color=C["lllm"],
                 arrowprops=dict(arrowstyle="->", color=C["lllm"], lw=0.8))

    plt.tight_layout(pad=1.0)
    path = os.path.join(OUT, "fig1_divergence_schematic.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Geometry Characterization: Rank-95 and Correlation
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_geometry_characterization():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))

    # --- left: Rank-95 real vs shuffled ---
    ax = axes[0]
    models = ["Qwen2.5\n0.5B", "Qwen2.5\n1.5B", "SmolLM2\n1.7B"]
    rank95_real     = [1.049, 1.167, 1.528]
    rank95_shuffled = [1.486, 1.458, 1.799]
    ci_real     = [(0.000, 0.062), (0.000, 0.104), (0.000, 0.048)]
    ci_shuffled = [(0.000, 0.060), (0.000, 0.060), (0.000, 0.040)]

    x = np.arange(len(models))
    w = 0.32

    bars_r = ax.bar(x - w/2, rank95_real,
                    width=w, color=C["real"], label="Real", alpha=0.88,
                    yerr=[[r[0] for r in ci_real], [r[1] for r in ci_real]],
                    capsize=3, error_kw={"linewidth": 0.8})
    bars_s = ax.bar(x + w/2, rank95_shuffled,
                    width=w, color=C["shuffled"], label="Shuffled (control)", alpha=0.88,
                    yerr=[[r[0] for r in ci_shuffled], [r[1] for r in ci_shuffled]],
                    capsize=3, error_kw={"linewidth": 0.8})

    # significance brackets
    for i, (rv, sv) in enumerate(zip(rank95_real, rank95_shuffled)):
        top = max(rv, sv) + 0.12
        ax.plot([x[i]-w/2, x[i]-w/2, x[i]+w/2, x[i]+w/2],
                [top-0.04, top, top, top-0.04], color="#444444", linewidth=0.8)
        pstar = "***" if i < 2 else "*"
        ax.text(x[i], top+0.01, pstar, ha="center", fontsize=8, color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Rank-95 (normalized, ↓ = more compact)")
    ax.set_title("(a) Conversation trajectory compactness")
    ax.legend()
    ax.set_ylim(0, 2.15)

    # --- right: correlation bar chart ρ(d_geo, Δℓ) ---
    ax2 = axes[1]
    models2  = ["Qwen2.5\n0.5B", "Qwen2.5\n1.5B", "SmolLM2\n1.7B"]
    rho_vals = [0.989, 0.994, 0.989]

    bars = ax2.barh(models2, rho_vals, color=C["semqcg"], alpha=0.85, height=0.45)

    for bar, rho in zip(bars, rho_vals):
        ax2.text(rho - 0.003, bar.get_y() + bar.get_height()/2,
                 f"{rho:.3f}", ha="right", va="center", fontsize=8,
                 color="white", fontweight="bold")

    ax2.set_xlim(0.97, 1.002)
    ax2.set_xlabel(r"Pearson $\rho$ ($d_\mathrm{geo}$, $\Delta\ell_2$)")
    ax2.set_title(r"(b) Geometry predicts decoder drift")
    ax2.axvline(1.0, color="#aaaaaa", linestyle="--", linewidth=0.8)

    # p-value annotation
    for i, m in enumerate(models2):
        ax2.text(0.9985, i, "$p{<}0.001$", va="center", fontsize=7,
                 color="#555555")

    plt.tight_layout(pad=1.0)
    path = os.path.join(OUT, "fig2_geometry_characterization.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Budget vs Logit L2: MSC Signal Variants
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_budget_logit_l2():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=False)

    # --- left: MSC 1000-conv baselines ---
    ax = axes[0]
    data_baselines_l2 = {
        "Uniform":       [627.2, 613.8, 596.3],
        "Recency":       [605.3, 572.6, 537.5],
        "Recency-KCD":   [604.9, 590.9, 564.4],
        "TF-IDF":        [617.4, 610.5, 594.0],
        "LongLLMLingua": [730.4, 741.0, 757.2],
    }
    styles_b = {
        "Uniform":       (C["uniform"], "s",  "-"),
        "Recency":       (C["recency"], "o",  "-"),
        "Recency-KCD":   (C["rec_kcd"], "D",  "-"),
        "TF-IDF":        (C["tfidf"],   "^",  "--"),
        "LongLLMLingua": (C["lllm"],    "X",  "-"),
    }
    for name, vals in data_baselines_l2.items():
        c, m, ls = styles_b[name]
        ax.plot(BUDGETS, vals, color=c, marker=m, linestyle=ls,
                label=name, linewidth=1.8 if name == "LongLLMLingua" else 1.5)

    ax.set_xlabel("Budget fraction $B$")
    ax.set_ylabel(r"Logit $\ell_2$ $\downarrow$")
    ax.set_title("(a) External baselines ($n{=}1{,}000$ conv)")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels(BUDGET_LABELS)
    ax.legend(fontsize=7.5)
    ax.annotate("LongLLMLingua\ncatastrophically\nhigher L2",
                xy=(0.50, 757.2), xytext=(0.35, 740),
                fontsize=6.5, color=C["lllm"],
                arrowprops=dict(arrowstyle="->", color=C["lllm"], lw=0.7))

    # --- right: MSC 50-conv KCD variants ---
    ax2 = axes[1]
    data_variants_l2 = {
        "Uniform":   [705.2, 692.2, 682.8],
        "Geo-KCD":   [709.8, 724.4, 737.0],
        "Sem-KCD":   [713.4, 740.0, 750.1],
        "Sem-QCG":   [717.8, 723.0, 636.4],
    }
    styles_v = {
        "Uniform": (C["uniform"], "s",  "--", 1.4),
        "Geo-KCD": (C["geo"],     "o",  "-",  1.6),
        "Sem-KCD": (C["sem"],     "D",  "-",  1.6),
        "Sem-QCG": (C["semqcg"], "*",  "-",  2.0),
    }
    for name, vals in data_variants_l2.items():
        c, m, ls, lw = styles_v[name]
        ax2.plot(BUDGETS, vals, color=c, marker=m, linestyle=ls,
                 label=name, linewidth=lw)

    # annotate the "below uniform" point
    ax2.annotate("Below uniform\n($p{<}0.001$)",
                 xy=(0.50, 636.4), xytext=(0.35, 648),
                 fontsize=6.5, color=C["semqcg"],
                 arrowprops=dict(arrowstyle="->", color=C["semqcg"], lw=0.7))
    ax2.axhline(682.8, color=C["uniform"], linestyle=":", linewidth=0.9, alpha=0.7)

    ax2.set_xlabel("Budget fraction $B$")
    ax2.set_ylabel(r"Logit $\ell_2$ $\downarrow$")
    ax2.set_title("(b) KCD signal variants ($n{=}50$ conv)")
    ax2.set_xticks(BUDGETS)
    ax2.set_xticklabels(BUDGET_LABELS)
    ax2.legend(fontsize=7.5)

    plt.tight_layout(pad=1.0)
    path = os.path.join(OUT, "fig3_budget_logit_l2.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Budget vs Answer NLL: MSC Signal Variants
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_budget_nll():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=False)

    # --- left: MSC 1000-conv baselines ---
    ax = axes[0]
    data_baselines_nll = {
        "Uniform":       [3.361, 3.269, 3.171],
        "Recency":       [3.236, 3.050, 2.901],
        "Recency-KCD":   [3.219, 3.021, 2.860],
        "TF-IDF":        [3.249, 3.074, 2.930],
        "LongLLMLingua": [3.114, 3.070, 3.013],
    }
    styles_b = {
        "Uniform":       (C["uniform"], "s",  "-"),
        "Recency":       (C["recency"], "o",  "-"),
        "Recency-KCD":   (C["rec_kcd"], "D",  "-"),
        "TF-IDF":        (C["tfidf"],   "^",  "--"),
        "LongLLMLingua": (C["lllm"],    "X",  "-"),
    }
    for name, vals in data_baselines_nll.items():
        c, m, ls = styles_b[name]
        ax.plot(BUDGETS, vals, color=c, marker=m, linestyle=ls, label=name)

    ax.set_xlabel("Budget fraction $B$")
    ax.set_ylabel("Answer NLL $\\downarrow$")
    ax.set_title("(a) External baselines ($n{=}1{,}000$ conv)")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels(BUDGET_LABELS)
    ax.legend(fontsize=7.5)

    # --- right: MSC 50-conv KCD variants ---
    ax2 = axes[1]
    data_variants_nll = {
        "Uniform": [3.251, 3.166, 3.071],
        "Geo-KCD": [3.201, 3.052, 2.923],
        "Sem-KCD": [3.176, 2.993, 2.845],
        "Sem-QCG": [3.143, 2.944, 2.789],
    }
    styles_v = {
        "Uniform": (C["uniform"], "s",  "--", 1.4),
        "Geo-KCD": (C["geo"],     "o",  "-",  1.6),
        "Sem-KCD": (C["sem"],     "D",  "-",  1.6),
        "Sem-QCG": (C["semqcg"], "*",  "-",  2.0),
    }
    for name, vals in data_variants_nll.items():
        c, m, ls, lw = styles_v[name]
        ax2.plot(BUDGETS, vals, color=c, marker=m, linestyle=ls,
                 label=name, linewidth=lw)

    ax2.set_xlabel("Budget fraction $B$")
    ax2.set_ylabel("Answer NLL $\\downarrow$")
    ax2.set_title("(b) KCD signal variants ($n{=}50$ conv)")
    ax2.set_xticks(BUDGETS)
    ax2.set_xticklabels(BUDGET_LABELS)
    ax2.legend(fontsize=7.5)

    # annotate sem-qcg wins at all budgets
    ax2.annotate("Sem-QCG best\nat all budgets",
                 xy=(0.35, 2.944), xytext=(0.22, 2.87),
                 fontsize=6.5, color=C["semqcg"],
                 arrowprops=dict(arrowstyle="->", color=C["semqcg"], lw=0.7))

    plt.tight_layout(pad=1.0)
    path = os.path.join(OUT, "fig4_budget_nll.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Support-Turn Rescue Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_support_turn_rescue():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # --- left: rescue counts by outcome ---
    ax = axes[0]
    outcomes   = ["Policy rescues\nmore support\nturns",
                  "Uniform beats\npolicy on\nsupport turns"]
    geo_counts = [17, 2]
    sem_counts = [7, 5]
    total      = 36

    x = np.arange(len(outcomes))
    w = 0.32

    b1 = ax.bar(x - w/2, [v/total*100 for v in geo_counts],
                width=w, color=C["geo"],   label="Geo-KCD", alpha=0.88)
    b2 = ax.bar(x + w/2, [v/total*100 for v in sem_counts],
                width=w, color=C["sem"],   label="Sem-KCD", alpha=0.88)

    def autolabel(bars, counts):
        for bar, count in zip(bars, counts):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f"{count}/36", ha="center", va="bottom", fontsize=7.5)

    autolabel(b1, geo_counts)
    autolabel(b2, sem_counts)

    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, fontsize=8)
    ax.set_ylabel("Conversations (%)")
    ax.set_title("(a) Support-turn rescue rate\n(budget $B{=}0.35$, $n{=}36$ conv)")
    ax.legend()
    ax.set_ylim(0, 60)

    # --- right: breakdown by turn family ---
    ax2 = axes[1]
    families    = ["Constraint\n(instruct, code)", "Base-memory\n(persona, fact)"]
    geo_rescued = [13, 6]
    sem_rescued = [4,  3]

    x2 = np.arange(len(families))
    b3 = ax2.bar(x2 - w/2, geo_rescued, width=w,
                 color=C["geo"], label="Geo-KCD", alpha=0.88)
    b4 = ax2.bar(x2 + w/2, sem_rescued, width=w,
                 color=C["sem"], label="Sem-KCD", alpha=0.88)

    def autolabel2(bars):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                     str(int(h)), ha="center", va="bottom", fontsize=8)

    autolabel2(b3)
    autolabel2(b4)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(families, fontsize=8)
    ax2.set_ylabel("Conversations rescued")
    ax2.set_title("(b) Rescue breakdown by turn family")
    ax2.legend()
    ax2.set_ylim(0, 18)

    # mechanism annotation
    fig.text(0.5, -0.02,
             "Geometry identifies constraint-critical turns (2.4× more rescues).\n"
             "Mechanism: user instructions displace hidden state more than "
             "semantically similar assistant echoes.",
             ha="center", fontsize=7.5, color="#555555", style="italic")

    plt.tight_layout(pad=1.0)
    path = os.path.join(OUT, "fig5_support_turn_rescue.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Scope Condition Map (2×2 + evidence strips)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_scope_conditions():
    fig = plt.figure(figsize=(6.5, 4.2))
    gs  = GridSpec(2, 3, figure=fig, width_ratios=[1.2, 1.2, 1.0],
                   hspace=0.55, wspace=0.45)

    # ── top-left: Qwen + short (MSC 50-conv) — sem_qcg wins both ──
    ax00 = fig.add_subplot(gs[0, 0])
    budgets = np.array(BUDGETS)
    ax00.plot(budgets, [705.2, 692.2, 682.8], color=C["uniform"], marker="s",
              linestyle="--", linewidth=1.2, label="Uniform")
    ax00.plot(budgets, [713.4, 740.0, 750.1], color=C["sem"],     marker="D",
              linewidth=1.4, label="Sem-KCD")
    ax00.plot(budgets, [717.8, 723.0, 636.4], color=C["semqcg"], marker="*",
              linewidth=1.8, label="Sem-QCG")
    ax00.axhline(682.8, color=C["uniform"], linestyle=":", linewidth=0.7, alpha=0.6)
    ax00.set_title("Qwen / Short conv\n(MSC, 20–60 turns)", fontsize=8, pad=3)
    ax00.set_ylabel(r"Logit $\ell_2$ $\downarrow$", fontsize=7.5)
    ax00.set_xticks(budgets); ax00.set_xticklabels(BUDGET_LABELS, fontsize=7)
    ax00.legend(fontsize=6.5, loc="upper right")
    ax00.text(0.05, 0.07, "Sem-QCG wins\nboth objectives",
              transform=ax00.transAxes, fontsize=6.5, color=C["semqcg"],
              style="italic",
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C["semqcg"], lw=0.6))

    # ── top-right: Qwen + long (LME 100-conv) — geometry degrades ──
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.plot(budgets, [1763.1, 1763.1, 1705.1], color=C["uniform"], marker="s",
              linestyle="--", linewidth=1.2, label="Uniform")
    ax01.plot(budgets, [1740.4, 1693.6, 1714.2], color=C["sem"],     marker="D",
              linewidth=1.4, label="Sem-KCD")
    ax01.plot(budgets, [1740.7, 1743.9, 1696.9], color=C["semqcg"], marker="*",
              linewidth=1.8, label="Sem-QCG")
    ax01.set_title("Qwen / Long conv\n(LME, 400–600 turns)", fontsize=8, pad=3)
    ax01.set_ylabel(r"Logit $\ell_2$ $\downarrow$", fontsize=7.5)
    ax01.set_xticks(budgets); ax01.set_xticklabels(BUDGET_LABELS, fontsize=7)
    ax01.legend(fontsize=6.5)
    ax01.annotate("Sem-QCG worse\nat $B{=}0.35$\n($p{=}0.011$)",
                 xy=(0.35, 1743.9), xytext=(0.22, 1760),
                 fontsize=6, color="#cc3300",
                 arrowprops=dict(arrowstyle="->", color="#cc3300", lw=0.7))

    # ── bottom-left: Llama + short (MSC 50-conv) — semantic wins ──
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.plot(budgets, [689.7, 669.4, 631.0], color=C["uniform"], marker="s",
              linestyle="--", linewidth=1.2, label="Uniform")
    ax10.plot(budgets, [690.6, 643.2, 592.2], color=C["geo"],     marker="o",
              linewidth=1.4, label="Geo-KCD")
    ax10.plot(budgets, [671.9, 633.5, 586.1], color=C["sem"],     marker="D",
              linewidth=1.8, label="Sem-KCD")
    ax10.set_title("Llama-3.2-3B / Short conv\n(MSC, 20–60 turns)", fontsize=8, pad=3)
    ax10.set_xlabel("Budget $B$", fontsize=7.5)
    ax10.set_ylabel(r"Logit $\ell_2$ $\downarrow$", fontsize=7.5)
    ax10.set_xticks(budgets); ax10.set_xticklabels(BUDGET_LABELS, fontsize=7)
    ax10.legend(fontsize=6.5)
    ax10.text(0.05, 0.07, "Semantic wins\nNLL ($p≤0.026$)",
              transform=ax10.transAxes, fontsize=6.5, color=C["sem"],
              style="italic",
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C["sem"], lw=0.6))

    # ── bottom-right: Llama + long — semantic wins NLL ──
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(budgets, [3.722, 3.657, 3.524], color=C["uniform"], marker="s",
              linestyle="--", linewidth=1.2, label="Uniform")
    ax11.plot(budgets, [3.617, 3.416, 3.260], color=C["geo"],     marker="o",
              linewidth=1.4, label="Geo-KCD")
    ax11.plot(budgets, [3.526, 3.317, 3.139], color=C["sem"],     marker="D",
              linewidth=1.8, label="Sem-KCD")
    ax11.set_title("Llama-3.2-3B NLL\n(MSC, 20–60 turns)", fontsize=8, pad=3)
    ax11.set_xlabel("Budget $B$", fontsize=7.5)
    ax11.set_ylabel("Answer NLL $\\downarrow$", fontsize=7.5)
    ax11.set_xticks(budgets); ax11.set_xticklabels(BUDGET_LABELS, fontsize=7)
    ax11.legend(fontsize=6.5)

    # ── right column: 2×2 summary heatmap ──
    ax_heat = fig.add_subplot(gs[:, 2])
    # rows: model family (Qwen, Llama)
    # cols: length (short, long)
    # values: 0=sem wins, 1=tie, 2=sem_qcg wins, 3=geo wins
    # Green = sem_qcg wins both; Blue = geo advantage; Orange = sem wins; Gray = tie/unclear
    labels = [
        ["Sem-QCG\nwins both", "Semantic\nbest\n(NLL)"],
        ["Semantic\nbest\n(NLL only)", "Not\ntested"],
    ]
    colors_heat = [
        [C["semqcg"], C["sem"]],
        [C["sem"],    "#cccccc"],
    ]
    for r in range(2):
        for c in range(2):
            rect = plt.Rectangle([c, 1-r], 1, 1,
                                  facecolor=colors_heat[r][c], alpha=0.75,
                                  edgecolor="white", linewidth=2)
            ax_heat.add_patch(rect)
            ax_heat.text(c + 0.5, 1 - r + 0.5, labels[r][c],
                        ha="center", va="center", fontsize=7.5,
                        color="white" if colors_heat[r][c] != "#cccccc" else "#555555",
                        fontweight="bold")

    ax_heat.set_xlim(0, 2)
    ax_heat.set_ylim(0, 2)
    ax_heat.set_xticks([0.5, 1.5])
    ax_heat.set_xticklabels(["Short\n(20–60 turns)", "Long\n(400–600 turns)"], fontsize=7.5)
    ax_heat.set_yticks([0.5, 1.5])
    ax_heat.set_yticklabels(["Llama-3.2-3B", "Qwen2.5"], fontsize=7.5)
    ax_heat.set_title("Scope condition\nsummary", fontsize=8, pad=4)
    ax_heat.tick_params(left=False, bottom=False)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.grid(False)

    path = os.path.join(OUT, "fig6_scope_conditions.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS — Hardset NLL damage by scale (for §5 inline figure if needed)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_hardset_scale():
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    budgets = np.array(BUDGETS)
    # delta NLL (sem - geo): positive = geometry is better
    nll_15b = np.array([0.195, 0.424, 0.628])
    nll_3b  = np.array([0.494, 0.853, 1.126])

    ax.plot(budgets, nll_15b, color="#4e79a7", marker="o", linewidth=1.8,
            label="Qwen2.5-1.5B")
    ax.plot(budgets, nll_3b,  color="#e15759", marker="s", linewidth=1.8,
            label="Qwen2.5-3B")

    ax.fill_between(budgets, nll_15b, nll_3b, alpha=0.10, color="#e15759")

    # significance markers
    sigs_15b = ["*",  "*",  "**"]
    sigs_3b  = ["**", "**", "***"]
    for b, s15, s3 in zip(budgets, sigs_15b, sigs_3b):
        ax.text(b, nll_15b[list(budgets).index(b)] + 0.02, s15,
                ha="center", fontsize=9, color="#4e79a7")
        ax.text(b, nll_3b[list(budgets).index(b)] + 0.02, s3,
                ha="center", fontsize=9, color="#e15759")

    ax.set_xlabel("Budget fraction $B$")
    ax.set_ylabel(r"$\Delta$ Answer NLL (sem $-$ geo, $\uparrow$ = geo better)")
    ax.set_title("NLL damage from wrong signal\ngrows with model scale")
    ax.set_xticks(budgets)
    ax.set_xticklabels(BUDGET_LABELS)
    ax.legend(fontsize=8)
    ax.axhline(0, color="#aaaaaa", linestyle="--", linewidth=0.8)

    plt.tight_layout(pad=0.8)
    path = os.path.join(OUT, "fig_hardset_scale.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures …")
    fig1_divergence_schematic()
    fig2_geometry_characterization()
    fig3_budget_logit_l2()
    fig4_budget_nll()
    fig5_support_turn_rescue()
    fig6_scope_conditions()
    fig_hardset_scale()
    print("Done. All PDFs written to figures/")
