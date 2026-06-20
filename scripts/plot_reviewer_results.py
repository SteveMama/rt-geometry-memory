#!/usr/bin/env python3
"""
plot_reviewer_results.py

Reads all CSVs written by run_reviewer_fixes_multigpu.sh and generates
publication-quality figures.

Usage:
    python scripts/plot_reviewer_results.py \
        --results-root results/reviewer_fixes \
        --output-dir  results/reviewer_fixes/plots

Outputs (all PNG + one combined PDF):
    01_score_curves.png         score vs compression budget, per benchmark
    02_score_boxplots.png       score distribution by policy × budget
    03_kcd_action_breakdown.png keep / compress / evict fractions
    04_budget_adherence.png     target vs achieved token fraction
    05_head_to_head.png         geometry_KCD vs longllmlingua per-conversation
    06_geometry_signal.png      geometry score distribution by action
    07_behavior_logprob.png     answer neg-logprob delta by policy (behavior rows)
    combined_report.pdf         all panels in one file
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[plot] ERROR: matplotlib not installed — run: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

try:
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

POLICY_COLORS = {
    "uniform":                                          "#888888",
    "longllmlingua":                                    "#e07b39",
    "semantic":                                         "#4e9af1",
    "semantic_keep_compress_drop":                      "#2166ac",
    "geometry_keep_compress_drop":                      "#d6004c",
    "semantic_query_conditioned_geometry_keep_compress_drop": "#6a0dad",
    "budget_aware_semantic_keep_compress_drop":         "#2ca25f",
}
POLICY_LABELS = {
    "uniform":                                          "Uniform (baseline)",
    "longllmlingua":                                    "LongLLMLingua",
    "semantic":                                         "Semantic",
    "semantic_keep_compress_drop":                      "Semantic KCD",
    "geometry_keep_compress_drop":                      "Geometry KCD ★",
    "semantic_query_conditioned_geometry_keep_compress_drop": "Signal-Cond. KCD ★",
    "budget_aware_semantic_keep_compress_drop":         "Budget-Aware Semantic KCD",
}
BENCHMARK_LABELS = {
    "hardset":          "Hard Stress Set",
    "baselines_hardset":"Hard Stress Set",
    "fullLME":          "LongMemEval-S (Full)",
    "msc":              "MSC Valid",
    "baselines_msc":    "MSC Valid",
    "scale_llama32_3b": "Llama-3.2-3B (Scale Val.)",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ── data loading ──────────────────────────────────────────────────────────────

def load_all_csvs(results_root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for csv in sorted(results_root.rglob(filename)):
        try:
            df = pd.read_csv(csv)
            # infer benchmark tag from path
            parts = csv.relative_to(results_root).parts
            tag = parts[0] if parts else "unknown"
            df["_benchmark_tag"] = tag
            df["_source_file"] = str(csv.relative_to(results_root))
            frames.append(df)
        except Exception as exc:
            print(f"[plot] skip {csv}: {exc}", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # deduplicate on natural key if columns present
    key_cols = [c for c in
                ["model_key", "conversation_id", "target_turn", "policy_name",
                 "budget_fraction", "_benchmark_tag"]
                if c in combined.columns]
    if key_cols:
        combined = combined.drop_duplicates(subset=key_cols)
    return combined


def policy_color(name: str) -> str:
    for k, v in POLICY_COLORS.items():
        if k in name:
            return v
    return "#333333"


def policy_label(name: str) -> str:
    for k, v in POLICY_LABELS.items():
        if k == name:
            return v
    return name


def benchmark_label(tag: str) -> str:
    for k, v in BENCHMARK_LABELS.items():
        if k in tag:
            return v
    return tag


def policy_order(policies):
    order = list(POLICY_COLORS.keys())
    known = [p for p in order if p in policies]
    unknown = sorted(set(policies) - set(known))
    return known + unknown


# ── Figure 1: Score curves ─────────────────────────────────────────────────────

def fig_score_curves(eval_df: pd.DataFrame, out: Path):
    if eval_df.empty or "top1_match" not in eval_df.columns:
        print("[plot] skip score curves — no top1_match column", file=sys.stderr)
        return

    df = eval_df.copy()
    df["score"] = df["top1_match"]
    tags = sorted(df["_benchmark_tag"].unique())
    n = len(tags)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(5 * max(n, 1), 4), squeeze=False)

    for ax, tag in zip(axes[0], tags):
        sub = df[df["_benchmark_tag"] == tag]
        budgets = sorted(sub["budget_fraction"].unique())
        policies = policy_order(sub["policy_name"].unique())

        for pol in policies:
            psub = sub[sub["policy_name"] == pol]
            means, cis = [], []
            for b in budgets:
                vals = psub[psub["budget_fraction"] == b]["score"].dropna()
                if len(vals) == 0:
                    means.append(np.nan); cis.append(0); continue
                m = vals.mean()
                se = vals.sem() * 1.96
                means.append(m); cis.append(se)
            means, cis = np.array(means), np.array(cis)
            ax.plot(budgets, means, "o-", color=policy_color(pol),
                    label=policy_label(pol), linewidth=2, markersize=5)
            ax.fill_between(budgets, means - cis, means + cis,
                            color=policy_color(pol), alpha=0.12)

        ax.set_title(benchmark_label(tag), fontsize=12, fontweight="bold")
        ax.set_xlabel("Compression budget")
        ax.set_ylabel("top1_match" if ax == axes[0][0] else "")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(budgets)
        ax.legend(fontsize=8, loc="lower right")
        ax.axhline(1.0, color="#cccccc", linewidth=0.8, linestyle="--")

    fig.suptitle("Score vs Compression Budget", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "01_score_curves.png")
    plt.close(fig)
    print(f"[plot] saved 01_score_curves.png")


# ── Figure 2: Score box plots ──────────────────────────────────────────────────

def fig_score_boxplots(eval_df: pd.DataFrame, out: Path):
    if eval_df.empty or "top1_match" not in eval_df.columns:
        return

    df = eval_df.copy()
    df["score"] = df["top1_match"]
    budgets = sorted(df["budget_fraction"].unique())
    tags = sorted(df["_benchmark_tag"].unique())

    fig, axes = plt.subplots(len(tags), len(budgets),
                             figsize=(4 * len(budgets), 3.5 * len(tags)),
                             squeeze=False)

    for row, tag in enumerate(tags):
        for col, b in enumerate(budgets):
            ax = axes[row][col]
            sub = df[(df["_benchmark_tag"] == tag) & (df["budget_fraction"] == b)]
            policies = policy_order(sub["policy_name"].unique())
            data = [sub[sub["policy_name"] == p]["score"].dropna().values for p in policies]
            colors = [policy_color(p) for p in policies]

            bp = ax.boxplot(data, patch_artist=True, notch=False,
                            medianprops={"color": "white", "linewidth": 2},
                            whiskerprops={"linewidth": 1.2},
                            capprops={"linewidth": 1.2},
                            flierprops={"marker": ".", "markersize": 3})
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.8)

            ax.set_xticks(range(1, len(policies) + 1))
            ax.set_xticklabels([policy_label(p) for p in policies],
                               rotation=35, ha="right", fontsize=8)
            ax.set_ylim(-0.05, 1.1)
            ax.set_ylabel("top1_match" if col == 0 else "")
            if row == 0:
                ax.set_title(f"budget={b}", fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{benchmark_label(tag)}\ntop1_match", fontsize=9)

    fig.suptitle("Score Distribution by Policy × Budget", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "02_score_boxplots.png")
    plt.close(fig)
    print("[plot] saved 02_score_boxplots.png")


# ── Figure 3: KCD action breakdown ────────────────────────────────────────────

def fig_kcd_actions(eval_df: pd.DataFrame, out: Path):
    need = {"kept_segment_count", "compressed_segment_count", "evicted_segment_count"}
    if eval_df.empty or not need.issubset(eval_df.columns):
        print("[plot] skip KCD breakdown — missing segment count columns", file=sys.stderr)
        return

    df = eval_df.copy()
    df["total_seg"] = df["kept_segment_count"] + df["compressed_segment_count"] + df["evicted_segment_count"]
    df = df[df["total_seg"] > 0].copy()
    df["frac_keep"]     = df["kept_segment_count"]       / df["total_seg"]
    df["frac_compress"] = df["compressed_segment_count"] / df["total_seg"]
    df["frac_evict"]    = df["evicted_segment_count"]    / df["total_seg"]

    policies = policy_order(df["policy_name"].unique())
    tags = sorted(df["_benchmark_tag"].unique())

    fig, axes = plt.subplots(1, max(len(tags), 1), figsize=(5 * max(len(tags), 1), 4), squeeze=False)
    for ax, tag in zip(axes[0], tags):
        sub = df[df["_benchmark_tag"] == tag]
        agg = sub.groupby("policy_name")[["frac_keep", "frac_compress", "frac_evict"]].mean()
        agg = agg.reindex([p for p in policies if p in agg.index])
        xs = np.arange(len(agg))
        ax.bar(xs, agg["frac_keep"],     color="#2ca25f", label="Keep")
        ax.bar(xs, agg["frac_compress"], color="#feb24c", label="Compress",
               bottom=agg["frac_keep"])
        ax.bar(xs, agg["frac_evict"],    color="#de2d26", label="Drop",
               bottom=agg["frac_keep"] + agg["frac_compress"])
        ax.set_xticks(xs)
        ax.set_xticklabels([policy_label(p) for p in agg.index],
                           rotation=35, ha="right", fontsize=8)
        ax.set_title(benchmark_label(tag), fontsize=11, fontweight="bold")
        ax.set_ylabel("Fraction of segments" if ax == axes[0][0] else "")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle("KCD Action Breakdown by Policy", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "03_kcd_action_breakdown.png")
    plt.close(fig)
    print("[plot] saved 03_kcd_action_breakdown.png")


# ── Figure 4: Budget adherence ────────────────────────────────────────────────

def fig_budget_adherence(eval_df: pd.DataFrame, out: Path):
    if eval_df.empty or "token_fraction" not in eval_df.columns:
        return

    df = eval_df.copy()
    policies = policy_order(df["policy_name"].unique())

    fig, ax = plt.subplots(figsize=(6, 5))
    for pol in policies:
        sub = df[df["policy_name"] == pol]
        ax.scatter(sub["budget_fraction"], sub["token_fraction"],
                   color=policy_color(pol), alpha=0.35, s=18,
                   label=policy_label(pol))

    lims = [0, 1]
    ax.plot(lims, lims, "k--", linewidth=1, label="perfect adherence")
    ax.fill_between(lims, [x - 0.1 for x in lims], [x + 0.1 for x in lims],
                    color="#cccccc", alpha=0.25, label="±10% band")
    ax.set_xlabel("Target budget fraction")
    ax.set_ylabel("Achieved token fraction")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Budget Adherence (target vs achieved)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "04_budget_adherence.png")
    plt.close(fig)
    print("[plot] saved 04_budget_adherence.png")


# ── Figure 5: Head-to-head ────────────────────────────────────────────────────

def fig_head_to_head(eval_df: pd.DataFrame, out: Path):
    if eval_df.empty or "top1_match" not in eval_df.columns:
        return

    # pivot: one row per (conv, budget, benchmark), columns = policy scores
    df = eval_df.copy()
    key_cols = [c for c in ["model_key", "conversation_id", "target_turn",
                             "budget_fraction", "_benchmark_tag"] if c in df.columns]
    pivot = df.pivot_table(index=key_cols, columns="policy_name",
                           values="top1_match", aggfunc="mean").reset_index()

    geom_col = next((c for c in pivot.columns
                     if "geometry_keep_compress_drop" in str(c)), None)
    ling_col  = next((c for c in pivot.columns if "longllmlingua" in str(c)), None)

    if geom_col is None or ling_col is None:
        print("[plot] skip head-to-head — geometry_KCD or longllmlingua not in results yet",
              file=sys.stderr)
        return

    valid = pivot[[geom_col, ling_col, "budget_fraction"]].dropna()
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(valid[ling_col], valid[geom_col],
                    c=valid["budget_fraction"], cmap="viridis",
                    alpha=0.55, s=22, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Budget")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.fill_between([0, 1], [0, 1], [1, 1], color="#d6004c", alpha=0.06,
                    label="Geometry KCD wins")
    ax.fill_between([0, 1], [0, 0], [0, 1], color="#e07b39", alpha=0.06,
                    label="LongLLMLingua wins")
    ax.set_xlabel("LongLLMLingua  top1_match")
    ax.set_ylabel("Geometry KCD  top1_match")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.set_title("Head-to-Head: Geometry KCD vs LongLLMLingua\n(per conversation, matched budget)",
                 fontsize=11, fontweight="bold")

    geom_wins = (valid[geom_col] > valid[ling_col]).mean() * 100
    ax.text(0.02, 0.97, f"Geometry KCD wins: {geom_wins:.0f}%",
            transform=ax.transAxes, fontsize=9, va="top", color="#d6004c")

    fig.tight_layout()
    fig.savefig(out / "05_head_to_head.png")
    plt.close(fig)
    print(f"[plot] saved 05_head_to_head.png  (geometry wins {geom_wins:.0f}%)")


# ── Figure 6: Geometry signal distributions ────────────────────────────────────

def fig_geometry_signal(cand_df: pd.DataFrame, out: Path):
    if cand_df.empty:
        return
    has_geom = "geometry_score" in cand_df.columns
    has_curv = "query_geom_v2_curvature" in cand_df.columns
    has_action = "action" in cand_df.columns

    if not (has_geom or has_curv):
        print("[plot] skip geometry signal — no geometry_score / curvature columns", file=sys.stderr)
        return

    metrics = []
    if has_geom:   metrics.append(("geometry_score",         "Geometry Score κ"))
    if has_curv:   metrics.append(("query_geom_v2_curvature","Query-Conditioned Curvature"))
    if "segment_mean_stabilized_curvature" in cand_df.columns:
        metrics.append(("segment_mean_stabilized_curvature", "Stabilized Curvature"))

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    keep_color = "#2ca25f"
    drop_color = "#de2d26"

    for ax, (col, label) in zip(axes[0], metrics):
        sub = cand_df[[col] + (["action"] if has_action else [])].dropna(subset=[col])
        vals = sub[col].clip(0, sub[col].quantile(0.99))

        if has_action:
            for act, color, zorder in [("keep_turn", keep_color, 3),
                                        ("compress_turn", "#feb24c", 2),
                                        ("drop_turn", drop_color, 1)]:
                v = sub.loc[sub["action"] == act, col].clip(0, vals.max())
                if len(v) == 0: continue
                ax.hist(v, bins=40, color=color, alpha=0.6,
                        density=True, label=act.replace("_turn", ""), zorder=zorder)
            ax.legend(fontsize=8)
        else:
            ax.hist(vals, bins=40, color="#4e9af1", alpha=0.75, density=True)

        ax.set_xlabel(label)
        ax.set_ylabel("Density" if ax == axes[0][0] else "")
        ax.set_title(label, fontsize=11, fontweight="bold")

    fig.suptitle("Geometry Signal Distributions by Codec Action",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "06_geometry_signal.png")
    plt.close(fig)
    print("[plot] saved 06_geometry_signal.png")


# ── Figure 7: Behavior (answer logprob delta) ─────────────────────────────────

def fig_behavior_logprob(behav_df: pd.DataFrame, out: Path):
    col = "answer_avg_neg_logprob_delta"
    if behav_df.empty or col not in behav_df.columns:
        return

    df = behav_df.copy()
    tags = sorted(df["_benchmark_tag"].unique())
    policies = policy_order(df["policy_name"].unique())

    fig, axes = plt.subplots(1, max(len(tags), 1), figsize=(5 * max(len(tags), 1), 4), squeeze=False)
    for ax, tag in zip(axes[0], tags):
        sub = df[df["_benchmark_tag"] == tag]
        budgets = sorted(sub["budget_fraction"].unique())
        for pol in policies:
            psub = sub[sub["policy_name"] == pol]
            means = [psub[psub["budget_fraction"] == b][col].mean() for b in budgets]
            sems  = [psub[psub["budget_fraction"] == b][col].sem() * 1.96 for b in budgets]
            ax.plot(budgets, means, "o-", color=policy_color(pol),
                    label=policy_label(pol), linewidth=2)
            ax.fill_between(budgets,
                            np.array(means) - np.array(sems),
                            np.array(means) + np.array(sems),
                            color=policy_color(pol), alpha=0.12)

        ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")
        ax.set_title(benchmark_label(tag), fontsize=11, fontweight="bold")
        ax.set_xlabel("Budget")
        ax.set_ylabel("Answer neg-logprob delta\n(lower=less degradation)" if ax == axes[0][0] else "")
        ax.legend(fontsize=8)

    fig.suptitle("Answer Quality Degradation by Policy (behavior rows)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "07_behavior_logprob.png")
    plt.close(fig)
    print("[plot] saved 07_behavior_logprob.png")


# ── Combined PDF ───────────────────────────────────────────────────────────────

def combine_pdf(out: Path):
    if not HAS_PDF:
        return
    pngs = sorted(out.glob("0*.png"))
    if not pngs:
        return
    pdf_path = out / "combined_report.pdf"
    with PdfPages(pdf_path) as pdf:
        for png in pngs:
            img = plt.imread(str(png))
            h, w = img.shape[:2]
            fig, ax = plt.subplots(figsize=(w / 150, h / 150))
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"[plot] saved combined_report.pdf  ({len(pngs)} pages)")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/reviewer_fixes")
    ap.add_argument("--output-dir",   default="results/reviewer_fixes/plots")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[plot] scanning {results_root} ...")

    eval_df  = load_all_csvs(results_root, "evaluation_rows.csv")
    behav_df = load_all_csvs(results_root, "behavior_rows.csv")
    cand_df  = load_all_csvs(results_root, "candidate_rows.csv")

    print(f"[plot] evaluation_rows : {len(eval_df):,} rows  "
          f"({eval_df['_benchmark_tag'].nunique() if not eval_df.empty else 0} benchmarks)")
    print(f"[plot] behavior_rows   : {len(behav_df):,} rows")
    print(f"[plot] candidate_rows  : {len(cand_df):,} rows")

    if eval_df.empty and behav_df.empty:
        print("[plot] No data found yet — run experiments first", file=sys.stderr)
        sys.exit(0)

    if not eval_df.empty:
        print(f"[plot] policies found  : {sorted(eval_df['policy_name'].unique())}")
        print(f"[plot] budgets found   : {sorted(eval_df['budget_fraction'].unique())}")

    fig_score_curves(eval_df, out)
    fig_score_boxplots(eval_df, out)
    fig_kcd_actions(eval_df, out)
    fig_budget_adherence(eval_df, out)
    fig_head_to_head(eval_df, out)
    fig_geometry_signal(cand_df, out)
    fig_behavior_logprob(behav_df, out)
    combine_pdf(out)

    print(f"\n[plot] All plots written to {out.resolve()}")


if __name__ == "__main__":
    main()
