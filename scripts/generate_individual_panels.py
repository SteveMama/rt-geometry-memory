"""Generate every figure panel individually from raw data.

Output: figures_export/from_data/<figure_group>/<panel_name>.png
"""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUT = ROOT / "figures_export" / "from_data"

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")

def _parse_memory_report(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    fields = {
        "better": r"retains more support user turns than uniform: (\d+)",
        "worse": r"Uniform retains more support user turns than .*: (\d+)",
        "latest": r"keeps the latest support turn while uniform drops it: (\d+)",
    }
    out: dict[str, int] = {}
    for key, pattern in fields.items():
        m = re.search(pattern, text)
        out[key] = int(m.group(1)) if m else 0
    return out


# ── load paper1 records (CSV rows + JSON summary attached) ───────────────────

def _load_paper1_records() -> tuple[list[dict], list[dict]]:
    csv_path = ARTIFACTS / "paper1" / "expanded_v8_final" / "conversation_summary.csv"
    baseline_path = ARTIFACTS / "paper1" / "expanded_v8_final" / "baseline_conversation_summary.csv"

    records = []
    for row in _read_csv(csv_path):
        rec = dict(row)
        if Path(rec.get("output_json", "")).exists():
            payload = _load_json(rec["output_json"])
            rec["summary"] = payload["summary"]
            rec["_series"] = payload["series"]
            rec["_segments"] = payload.get("segments", [])
        records.append(rec)

    # Baseline CSV is already flat (no output_json pointer)
    baseline = [dict(row) for row in _read_csv(baseline_path)]
    return records, baseline


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 1 — per-family trace panels
# ══════════════════════════════════════════════════════════════════════════════

def _p1_family_trace(records: list[dict], series_key: str, ylabel: str, title_prefix: str, out_dir: Path) -> None:
    families = sorted({r["family"] for r in records})
    for family in families:
        fig, ax = plt.subplots(figsize=(8, 4))
        for rec in records:
            if rec["family"] != family or "_series" not in rec:
                continue
            values = np.asarray(rec["_series"].get(series_key, []), dtype=np.float32)
            if values.size == 0:
                continue
            ax.plot(np.arange(1, values.size + 1), values, marker="o", linewidth=1.5, alpha=0.8)
        ax.set_title(f"{title_prefix}: {family}")
        ax.set_xlabel("Interior turn index")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        _save(fig, out_dir / f"{family}.png")


def generate_p1_traces(records: list[dict]) -> None:
    print("\n[Paper 1] Trace plots per family")
    _p1_family_trace(records, "curvatures",          "Curvature",          "Curvature Traces",          OUT / "curvature_traces")
    _p1_family_trace(records, "boundary_scores",     "Hybrid score",       "Boundary Score Traces",     OUT / "boundary_score_traces")
    _p1_family_trace(records, "boundary_prominences","Boundary prominence", "Boundary Prominence Traces",OUT / "boundary_prominence_traces")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 1 — rank95 by family (single bar chart)
# ══════════════════════════════════════════════════════════════════════════════

def generate_p1_rank95(records: list[dict]) -> None:
    print("\n[Paper 1] Rank95 by family")
    grouped: dict[tuple[str, str], list[float]] = {}
    for rec in records:
        if "summary" not in rec:
            continue
        key = (rec["model_key"], rec["family"])
        grouped.setdefault(key, []).append(float(rec["summary"]["mean_rank95"]))

    model_keys = sorted({rec["model_key"] for rec in records})
    families   = sorted({rec["family"]    for rec in records})
    x = np.arange(len(families), dtype=np.float32)
    width = min(0.75 / max(len(model_keys), 1), 0.28)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for idx, mk in enumerate(model_keys):
        offsets = x + (idx - (len(model_keys) - 1) / 2.0) * width
        vals = [_mean(grouped.get((mk, f), [])) for f in families]
        ax.bar(offsets, vals, width=width, label=mk)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=20, ha="right")
    ax.set_ylabel("Mean rank95")
    ax.set_title("Segment Rank95 by Family")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, OUT / "rank95_by_family" / "rank95_by_family.png")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 1 — family correlation heatmap (split into 2 panels)
# ══════════════════════════════════════════════════════════════════════════════

def generate_p1_correlation_heatmap(records: list[dict]) -> None:
    print("\n[Paper 1] Family correlation heatmap (individual panels)")
    model_keys = sorted({r["model_key"] for r in records})
    families   = sorted({r["family"]    for r in records})

    logit_matrix = np.zeros((len(model_keys), len(families)), dtype=np.float32)
    kl_matrix    = np.zeros((len(model_keys), len(families)), dtype=np.float32)
    for mi, mk in enumerate(model_keys):
        for fi, fam in enumerate(families):
            rows = [r for r in records if r["model_key"] == mk and r["family"] == fam]
            logit_matrix[mi, fi] = _mean([float(r["corr_geodesic_vs_logit_l2"]) for r in rows])
            kl_matrix   [mi, fi] = _mean([float(r["corr_geodesic_vs_kl"])       for r in rows])

    for matrix, name, title in [
        (logit_matrix, "corr_geodesic_logitL2", "Mean corr(geodesic, logit L2)"),
        (kl_matrix,    "corr_geodesic_KL",      "Mean corr(geodesic, KL)"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_xticks(np.arange(len(families)))
        ax.set_xticklabels(families, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(model_keys)))
        ax.set_yticklabels(model_keys)
        ax.set_title(title)
        for ri in range(matrix.shape[0]):
            for ci in range(matrix.shape[1]):
                ax.text(ci, ri, f"{matrix[ri, ci]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        _save(fig, OUT / "family_correlation_heatmap" / f"{name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 1 — boundary eval heatmap (4 panels split)
# ══════════════════════════════════════════════════════════════════════════════

def _heatmap_panel(model_keys: list[str], families: list[str], matrix: np.ndarray,
                   title: str, cmap: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels(families, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(model_keys)))
    ax.set_yticklabels(model_keys)
    ax.set_title(title)
    for ri in range(matrix.shape[0]):
        for ci in range(matrix.shape[1]):
            ax.text(ci, ri, f"{matrix[ri, ci]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, out_path)


def generate_p1_boundary_heatmap(records: list[dict]) -> None:
    print("\n[Paper 1] Boundary eval heatmap (individual panels)")
    model_keys = sorted({r["model_key"] for r in records})
    families   = sorted({r["family"]    for r in records})

    def _build(col: str) -> np.ndarray:
        m = np.zeros((len(model_keys), len(families)), dtype=np.float32)
        for mi, mk in enumerate(model_keys):
            for fi, fam in enumerate(families):
                rows = [r for r in records if r["model_key"] == mk and r["family"] == fam]
                m[mi, fi] = _mean([float(r.get(col, 0.0)) for r in rows])
        return m

    for col, name, title in [
        ("boundary_f1_exact", "boundary_f1_exact", "Mean boundary F1 exact"),
        ("boundary_f1_tol1",  "boundary_f1_tol1",  "Mean boundary F1 tol1"),
        ("boundary_f1_tol2",  "boundary_f1_tol2",  "Mean boundary F1 tol2"),
        ("boundary_f1_tol3",  "boundary_f1_tol3",  "Mean boundary F1 tol3"),
    ]:
        _heatmap_panel(model_keys, families, _build(col), title, "YlGnBu",
                       OUT / "boundary_eval_heatmap" / f"{name}.png")


def generate_p1_baseline_heatmap(baseline: list[dict]) -> None:
    print("\n[Paper 1] Baseline eval heatmap (individual panels)")
    baseline_names = sorted({r.get("baseline_name", r.get("model_key","")) for r in baseline})
    families       = sorted({r["family"] for r in baseline})

    def _build(col: str) -> np.ndarray:
        m = np.zeros((len(baseline_names), len(families)), dtype=np.float32)
        for bi, bn in enumerate(baseline_names):
            for fi, fam in enumerate(families):
                rows = [r for r in baseline
                        if r.get("baseline_name", r.get("model_key","")) == bn and r["family"] == fam]
                m[bi, fi] = _mean([float(r.get(col, 0.0)) for r in rows])
        return m

    for col, name, title in [
        ("boundary_f1_exact", "baseline_f1_exact", "Baseline boundary F1 exact"),
        ("boundary_f1_tol1",  "baseline_f1_tol1",  "Baseline boundary F1 tol1"),
        ("boundary_f1_tol2",  "baseline_f1_tol2",  "Baseline boundary F1 tol2"),
        ("boundary_f1_tol3",  "baseline_f1_tol3",  "Baseline boundary F1 tol3"),
    ]:
        _heatmap_panel(baseline_names, families, _build(col), title, "YlOrBr",
                       OUT / "baseline_eval_heatmap" / f"{name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 1 — geometry vs decoder scatter (2 panels split)
# ══════════════════════════════════════════════════════════════════════════════

def generate_p1_geometry_vs_decoder(records: list[dict]) -> None:
    print("\n[Paper 1] Geometry vs decoder scatter (individual panels)")
    families  = sorted({r["family"] for r in records})
    color_map = plt.get_cmap("tab10")

    for axis_idx, (y_key, ylabel, name, title) in enumerate([
        ("logit_l2", "Logit L2",       "geodesic_vs_logitL2", "Geodesic Error vs Logit Drift"),
        ("kl",       "KL divergence",  "geodesic_vs_KL",      "Geodesic Error vs KL"),
    ]):
        fig, ax = plt.subplots(figsize=(7, 5))
        for fi, fam in enumerate(families):
            geo_vals: list[float] = []
            y_vals:   list[float] = []
            for rec in records:
                if rec["family"] != fam or "_series" not in rec:
                    continue
                geo_vals.extend(rec["_series"].get("state_geodesic_errors", []))
                y_vals.extend(rec["_series"].get(y_key, []))
            if geo_vals:
                ax.scatter(geo_vals, y_vals, s=18, alpha=0.7, label=fam, color=color_map(fi % 10))
        ax.set_title(title)
        ax.set_xlabel("State geodesic error")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        _save(fig, OUT / "geometry_vs_decoder" / f"{name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 1 — rank energy curves (single)
# ══════════════════════════════════════════════════════════════════════════════

def generate_p1_rank_energy(records: list[dict]) -> None:
    print("\n[Paper 1] Rank energy curves")
    grouped: dict[tuple[str, str], list[list[float]]] = {}
    for rec in records:
        if "_segments" not in rec:
            continue
        key = (rec["model_key"], rec["family"])
        for seg in rec["_segments"]:
            ce = seg.get("cumulative_energy", [])
            if ce:
                grouped.setdefault(key, []).append(ce)

    keys = sorted(grouped)
    color_map = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for idx, key in enumerate(keys):
        curves = [np.asarray(c, dtype=np.float32) for c in grouped[key] if c]
        if not curves:
            continue
        max_len = max(c.size for c in curves)
        padded = np.ones((len(curves), max_len), dtype=np.float32)
        for ri, c in enumerate(curves):
            padded[ri, :c.size] = c
            if c.size < max_len:
                padded[ri, c.size:] = c[-1]
        mean_curve = padded.mean(axis=0)
        ax.plot(np.arange(1, max_len + 1), mean_curve, marker="o", linewidth=1.5,
                alpha=0.85, color=color_map(idx % 20), label=f"{key[0]} / {key[1]}")
    ax.set_title("Average Rank-Energy Curves")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cumulative energy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, OUT / "rank_energy_curves" / "rank_energy_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 2 — per-model budget curves (3 metrics × 3 models = 9 panels)
# ══════════════════════════════════════════════════════════════════════════════

_POLICY_COLORS = {
    "uniform":                   "#5f5f5f",
    "uniform_segment_actions":   "#8a8a8a",
    "lexical":                   "#1f77b4",
    "geometry":                  "#d95f02",
    "geometry_lexical":          "#2ca02c",
    "geometry_segment_actions":  "#7b3294",
}
_POLICY_ORDER = list(_POLICY_COLORS)


def generate_p2_budget_curves(rows: list[dict[str, str]]) -> None:
    print("\n[Paper 2] Budget curves per model (logit, kl, token)")
    model_keys   = sorted({r["model_key"]      for r in rows})
    budget_keys  = sorted({r["budget_fraction"] for r in rows}, key=float)
    present      = {r["policy_name"] for r in rows}
    policy_names = [p for p in _POLICY_ORDER if p in present]

    for metric, ylabel, title_prefix, out_dir in [
        ("mean_logit_l2",       "Mean logit L2",      "Logit Budget Curves",  OUT / "logit_budget_curves"),
        ("mean_kl",             "Mean KL divergence", "KL Budget Curves",     OUT / "kl_budget_curves"),
        ("mean_token_fraction", "Actual token frac",  "Token Budget Curves",  OUT / "token_budget_curves"),
    ]:
        for mk in model_keys:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            x = np.asarray([float(b) for b in budget_keys], dtype=np.float32)
            for policy in policy_names:
                y = np.asarray([
                    _mean([float(r[metric]) for r in rows
                           if r["model_key"] == mk and r["policy_name"] == policy
                           and r["budget_fraction"] == bk])
                    for bk in budget_keys
                ], dtype=np.float32)
                ax.plot(x, y, marker="o", linewidth=2.0, label=policy,
                        color=_POLICY_COLORS[policy])
            ax.set_title(f"{title_prefix} — {mk}")
            ax.set_xlabel("Budget fraction")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            _save(fig, out_dir / f"{mk}.png")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 2 — family logit heatmap (single)
# ══════════════════════════════════════════════════════════════════════════════

def generate_p2_family_heatmap() -> None:
    print("\n[Paper 2] Family logit heatmap")
    # Load per-model evaluation_rows and combine with model_key tag
    base = ROOT / "results" / "paper2" / "studies" / "behavior_stress_v1"
    model_keys = ["qwen25_05b", "qwen25_15b", "smollm2_17b"]
    all_rows: list[dict[str, str]] = []
    for mk in model_keys:
        for row in _read_csv(base / mk / "evaluation_rows.csv"):
            row["model_key"] = mk
            all_rows.append(row)

    families         = sorted({r["family"]      for r in all_rows})
    model_policy_keys= sorted({(r["model_key"], r["policy_name"]) for r in all_rows})
    budget_fraction  = "0.35"

    matrix = np.zeros((len(model_policy_keys), len(families)), dtype=np.float32)
    for ri, (mk, pn) in enumerate(model_policy_keys):
        for ci, fam in enumerate(families):
            vals = [float(r["logit_l2"]) for r in all_rows
                    if r["model_key"] == mk and r["policy_name"] == pn
                    and r["family"] == fam and r["budget_fraction"] == budget_fraction]
            matrix[ri, ci] = float(np.mean(vals)) if vals else 0.0

    fig, ax = plt.subplots(figsize=(10.5, max(4.6, 0.6 * len(model_policy_keys))))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels(families, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(model_policy_keys)))
    ax.set_yticklabels([f"{mk}:{pn}" for mk, pn in model_policy_keys])
    ax.set_title(f"Mean logit L2 by model/policy × family @ budget {budget_fraction}")
    for ri in range(matrix.shape[0]):
        for ci in range(matrix.shape[1]):
            ax.text(ci, ri, f"{matrix[ri, ci]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, OUT / "family_logit_heatmap" / "family_logit_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 2 checkpoint — panels B and C (A=logit curves, D=token curves already done)
# ══════════════════════════════════════════════════════════════════════════════

def generate_p2_checkpoint_panels() -> None:
    print("\n[Paper 2 checkpoint] Individual panels B and C")
    sig = _load_json(ARTIFACTS / "paper2" / "blazing_study_v3_confidence" / "significance_summary.json")
    mechanism = _parse_memory_report(
        ARTIFACTS / "paper2" / "behavior_stress_qwen_cases" / "memory_critical_qwen25_05b_b035.md"
    )

    # Panel B — support-turn rescue bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Geometry better", "Uniform better", "Latest support rescued"]
    vals   = [mechanism["better"], mechanism["worse"], mechanism["latest"]]
    colors = ["#2ca02c", "#d62728", "#1f77b4"]
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, max(vals) + 4)
    ax.set_ylabel("Cases out of 36")
    ax.set_title("B. Support-Turn Rescue at Budget 0.35")
    ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    _save(fig, OUT / "paper2_checkpoint" / "B_support_turn_rescue.png")

    # Panel C — cross-model geometry vs uniform barh with CI
    models   = ["qwen25_05b", "qwen25_15b", "smollm2_17b"]
    means    = [sig[m]["0.35"]["geometry"]["delta_logit_l2"]["mean"]    for m in models]
    lo       = [sig[m]["0.35"]["geometry"]["delta_logit_l2"]["ci_low"]  for m in models]
    hi       = [sig[m]["0.35"]["geometry"]["delta_logit_l2"]["ci_high"] for m in models]
    y        = np.arange(len(models))
    err_low  = [m - l for m, l in zip(means, lo)]
    err_high = [h - m for m, h in zip(means, hi)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(y, means, color="#1f77b4", alpha=0.85)
    ax.errorbar(means, y, xerr=[err_low, err_high], fmt="none", ecolor="black", capsize=4)
    ax.axvline(0.0, color="black", linewidth=1, alpha=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(["Qwen-0.5B", "Qwen-1.5B", "SmolLM2-1.7B"])
    ax.set_title("C. Geometry vs Uniform at Budget 0.35")
    ax.set_xlabel("Mean Δ logit L2 (negative is better)")
    fig.tight_layout()
    _save(fig, OUT / "paper2_checkpoint" / "C_geometry_vs_uniform.png")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER 3 checkpoint — all 4 panels
# ══════════════════════════════════════════════════════════════════════════════

def generate_p3_checkpoint_panels() -> None:
    print("\n[Paper 3 checkpoint] Individual panels A, B, C, D")
    fair_eval    = _read_csv(ARTIFACTS / "paper3" / "paper3_batch_v1_fairness" / "evaluation_rows.csv")
    probe_eval   = _read_csv(ARTIFACTS / "paper3" / "paper3_batch_v1_3b"       / "evaluation_rows.csv")
    fair_summary = _load_json(ARTIFACTS / "paper3" / "paper3_batch_v1_fairness" / "study_summary.json")
    probe_summary= _load_json(ARTIFACTS / "paper3" / "paper3_batch_v1_3b"       / "study_summary.json")

    fair_budgets  = [0.24, 0.28, 0.32, 0.35, 0.38, 0.42, 0.46, 0.50]
    probe_budgets = [0.20, 0.35, 0.50]
    policies = ["geometry", "geometry_segment_actions", "geometry_keep_compress_drop"]
    colors   = {"geometry": "#1f77b4", "geometry_segment_actions": "#d62728",
                 "geometry_keep_compress_drop": "#2ca02c"}
    labels   = {"geometry": "Geometry", "geometry_segment_actions": "Segment Actions",
                 "geometry_keep_compress_drop": "Keep/Compress/Drop"}

    def _delta_logit(eval_rows, model, budgets, policy):
        vals = []
        for b in budgets:
            sub = [float(r["logit_l2"]) for r in eval_rows
                   if r["model_key"] == model and r["policy_name"] == policy
                   and math.isclose(float(r["budget_fraction"]), b)]
            uni = [float(r["logit_l2"]) for r in eval_rows
                   if r["model_key"] == model and r["policy_name"] == "uniform"
                   and math.isclose(float(r["budget_fraction"]), b)]
            vals.append(np.mean(sub) - np.mean(uni) if sub and uni else 0.0)
        return vals

    # Panel A — qwen25_15b fairness sweep
    fig, ax = plt.subplots(figsize=(7, 5))
    for pol in policies:
        ax.plot(fair_budgets, _delta_logit(fair_eval, "qwen25_15b", fair_budgets, pol),
                marker="o", color=colors[pol], alpha=0.9, label=labels[pol])
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    ax.set_title("A. qwen25_15b Fairness Sweep")
    ax.set_xlabel("Budget Fraction")
    ax.set_ylabel("Mean Δ logit L2")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, OUT / "paper3_checkpoint" / "A_15b_fairness_sweep.png")

    # Panel B — qwen25_3b probe
    fig, ax = plt.subplots(figsize=(7, 5))
    for pol in policies:
        ax.plot(probe_budgets, _delta_logit(probe_eval, "qwen25_3b", probe_budgets, pol),
                marker="o", color=colors[pol], label=labels[pol])
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    ax.set_title("B. qwen25_3b Probe")
    ax.set_xlabel("Budget Fraction")
    ax.set_ylabel("Mean Δ logit L2")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, OUT / "paper3_checkpoint" / "B_3b_probe.png")

    # Panel C — active compression bar chart
    width = 0.25
    xs = np.arange(2)
    fair_pl  = fair_summary["models"]["qwen25_15b"]["aggregate"]["geometry_keep_compress_drop"]
    probe_pl = probe_summary["models"]["qwen25_3b"]["aggregate"]["geometry_keep_compress_drop"]
    payloads = [fair_pl, probe_pl]

    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, budget in enumerate([0.35, 0.50]):
        vals = [pl[f"{budget:.2f}"]["mean_compressed_segments"] for pl in payloads]
        ax.bar(xs + (idx - 0.5) * width, vals, width=width, label=f"Budget {budget:.2f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(["Qwen-1.5B", "Qwen-3B"])
    ax.set_title("C. Active Compression in Keep/Compress/Drop")
    ax.set_ylabel("Mean compressed segments")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, OUT / "paper3_checkpoint" / "C_active_compression.png")

    # Panel D — memory-critical mechanism grouped bar
    report_specs = [
        ("qwen25_15b", "keep_compress_drop",  "paper3_batch_v1_fairness", "Qwen-1.5B K/C/D"),
        ("qwen25_3b",  "keep_compress_drop",  "paper3_batch_v1_3b",       "Qwen-3B K/C/D"),
        ("qwen25_15b", "segment_actions",     "paper3_batch_v1_fairness", "Qwen-1.5B Seg"),
        ("qwen25_3b",  "segment_actions",     "paper3_batch_v1_3b",       "Qwen-3B Seg"),
    ]
    better_vals, latest_vals, worse_vals, names = [], [], [], []
    for model, suffix, folder, label in report_specs:
        path = ARTIFACTS / "paper3" / folder / f"memory_critical_{model}_{suffix}_b035.md"
        pl = _parse_memory_report(path)
        names.append(label)
        better_vals.append(pl["better"])
        latest_vals.append(pl["latest"])
        worse_vals.append(pl["worse"])

    xs = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(xs - width, better_vals, width=width, color="#2ca02c", label="Better than uniform")
    ax.bar(xs,         latest_vals, width=width, color="#1f77b4", label="Latest support rescued")
    ax.bar(xs + width, worse_vals,  width=width, color="#d62728", label="Worse than uniform")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title("D. Memory-Critical Mechanism at Budget 0.35")
    ax.set_ylabel("Cases out of 36")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, OUT / "paper3_checkpoint" / "D_memory_critical_mechanism.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"Output → {OUT}")
    if OUT.exists():
        shutil.rmtree(OUT)

    # Paper 1
    records, baseline = _load_paper1_records()
    print(f"Loaded {len(records)} paper1 records, {len(baseline)} baseline records")
    generate_p1_traces(records)
    generate_p1_rank95(records)
    generate_p1_correlation_heatmap(records)
    generate_p1_boundary_heatmap(records)
    generate_p1_baseline_heatmap(baseline)
    generate_p1_geometry_vs_decoder(records)
    generate_p1_rank_energy(records)

    # Paper 2
    p2_rows = _read_csv(ARTIFACTS / "paper2" / "behavior_stress_v1" / "policy_budget_summary.csv")
    print(f"\nLoaded {len(p2_rows)} paper2 policy_budget_summary rows")
    generate_p2_budget_curves(p2_rows)
    generate_p2_family_heatmap()

    # Paper 2 checkpoint panels
    generate_p2_checkpoint_panels()

    # Paper 3 checkpoint panels
    generate_p3_checkpoint_panels()

    # Count
    total = sum(1 for _ in OUT.rglob("*.png"))
    print(f"\nDone — {total} panels written to figures_export/from_data/")


if __name__ == "__main__":
    main()
