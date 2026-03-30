#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MANUSCRIPT = ROOT / "manuscript"
GENERATED = MANUSCRIPT / "generated"
FIGURES = MANUSCRIPT / "figures"


@dataclass
class Cell:
    x: float
    y: float
    w: float
    h: float
    title: str
    body: str
    face: str


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _tex(text: str) -> str:
    return text.replace("_", r"\_")


def _copy_figures() -> None:
    mapping = {
        ARTIFACTS / "shareable" / "paper2_checkpoint_overview.png": FIGURES / "paper2_checkpoint_overview.png",
        ARTIFACTS / "shareable" / "paper3_checkpoint_overview.png": FIGURES / "paper3_checkpoint_overview.png",
    }
    for src, dst in mapping.items():
        shutil.copy2(src, dst)


def _build_program_overview() -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cells = [
        Cell(
            0.04,
            0.17,
            0.25,
            0.66,
            "Paper 1: Geometry",
            "Turn-level state summaries are normalized to the sphere, transported into local tangent spaces, and analyzed for low-rank structure, curvature, and decoder relevance.",
            "#dfefff",
        ),
        Cell(
            0.375,
            0.17,
            0.25,
            0.66,
            "Paper 2: Control",
            "Geometry-derived risk scores allocate memory budget. The main result is lower decoder drift than uniform retention under conversational scarcity, plus support-turn rescue as the mechanism.",
            "#e8f7e4",
        ),
        Cell(
            0.71,
            0.17,
            0.25,
            0.66,
            "Paper 3: Sparse Codec",
            "A segment anchor plus sparse support memory enables keep/compress/drop decisions. Hard support-turn benchmarks favor geometry-aware codecs, while semantic-memory benchmarks favor semantic-led policies and also expose clear limits of geometry-only extensions.",
            "#fff0d9",
        ),
    ]

    for cell in cells:
        patch = FancyBboxPatch(
            (cell.x, cell.y),
            cell.w,
            cell.h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.5,
            edgecolor="#22313f",
            facecolor=cell.face,
        )
        ax.add_patch(patch)
        ax.text(cell.x + 0.02, cell.y + cell.h - 0.08, cell.title, fontsize=15, fontweight="bold", va="top")
        ax.text(
            cell.x + 0.02,
            cell.y + cell.h - 0.15,
            cell.body,
            fontsize=11.5,
            va="top",
            wrap=True,
        )

    arrow_1 = FancyArrowPatch((0.30, 0.5), (0.365, 0.5), arrowstyle="->", mutation_scale=18, linewidth=2.0)
    arrow_2 = FancyArrowPatch((0.635, 0.5), (0.70, 0.5), arrowstyle="->", mutation_scale=18, linewidth=2.0)
    ax.add_patch(arrow_1)
    ax.add_patch(arrow_2)

    ax.text(0.5, 0.92, "Checkpoint Research Program Overview", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(
        0.5,
        0.08,
        "Conversation states -> geometric characterization -> geometry-aware control -> sparse conversational memory codecs",
        ha="center",
        va="center",
        fontsize=11.5,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "program_overview.png", bbox_inches="tight")
    plt.close(fig)


def _build_paper3_regime_map() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8), dpi=220)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 4)
    ax.axis("off")

    palette = {
        "KCD": "#f4c27a",
        "semantic": "#9cc5e8",
        "semantic-family": "#b7d7f0",
        "geometry": "#a9d18e",
        "mixed": "#d9d9d9",
    }
    grid = [
        ("Hard stress", ["KCD", "KCD", "geometry"]),
        ("LongMemEval public", ["semantic", "mixed", "KCD"]),
        ("MSC", ["semantic-family", "semantic", "semantic"]),
        ("LoCoMo (bounded)", ["semantic-family", "semantic-family", "semantic-family"]),
    ]
    budgets = ["0.20", "0.35", "0.50"]

    for x, label in enumerate(budgets):
        ax.text(x + 0.5, 4.08, f"Budget {label}", ha="center", va="bottom", fontsize=11.5, fontweight="bold")

    for row_idx, (name, winners) in enumerate(grid):
        y = 3 - row_idx
        ax.text(-0.08, y + 0.5, name, ha="right", va="center", fontsize=11.0, fontweight="bold")
        for x, winner in enumerate(winners):
            rect = FancyBboxPatch(
                (x + 0.06, y + 0.08),
                0.88,
                0.84,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                linewidth=1.0,
                edgecolor="#25313d",
                facecolor=palette[winner],
            )
            ax.add_patch(rect)
            label = {
                "KCD": "KCD",
                "semantic": "semantic",
                "semantic-family": "semantic\nfamily",
                "geometry": "geometry",
                "mixed": "semantic /\nsegment",
            }[winner]
            ax.text(x + 0.5, y + 0.5, label, ha="center", va="center", fontsize=10.5)

    ax.text(
        1.5,
        -0.18,
        "Qualitative winner map from the current checkpoint: no universal best codec, only benchmark-dependent win regions.",
        ha="center",
        va="top",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "paper3_regime_map.png", bbox_inches="tight")
    plt.close(fig)


def _paper1_table() -> str:
    conf = _load_json(ARTIFACTS / "paper1" / "expanded_v8_final" / "confidence_summary.json")
    summary = _load_json(ARTIFACTS / "paper1" / "expanded_v8_final" / "study_summary.json")

    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Rank95 & 95\% CI & Shuffled Rank95 & $p_{\mathrm{H1}}$ & Corr$(d_{\mathrm{geo}}, \Delta \ell)$ & $p_{\mathrm{H3}}$ \\",
        r"\midrule",
    ]
    for mk in ["qwen25_05b", "qwen25_15b", "smollm2_17b"]:
        c = conf[mk]
        h1 = summary["null_controls"][mk]["h1_shuffled_turn_order"]
        h3 = summary["null_controls"][mk]["h3_permuted_alignment"]
        lines.append(
            f"{_tex(mk)} & "
            f"{_fmt(c['mean_rank95']['estimate'])} & "
            f"[{_fmt(c['mean_rank95']['ci_low'])}, {_fmt(c['mean_rank95']['ci_high'])}] & "
            f"{_fmt(h1['shuffled_mean_rank95'])} & "
            f"{_fmt(h1['p_value'], 4)} & "
            f"{_fmt(c['mean_corr_geodesic_vs_logit_l2']['estimate'])} & "
            f"{_fmt(h3['p_value'], 4)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _paper2_table() -> str:
    sig = _load_json(ARTIFACTS / "paper2" / "behavior_stress_v1" / "significance_summary.json")
    beh = _load_json(ARTIFACTS / "paper2" / "behavior_stress_v1" / "behavior_significance_summary.json")
    study = _load_json(ARTIFACTS / "paper2" / "behavior_stress_v1" / "study_summary.json")

    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & $\Delta$Logit@0.20 & $p$ & $\Delta$Logit@0.35 & $p$ & $\Delta$NLL@0.35 & $p$ \\",
        r"\midrule",
    ]
    for mk in ["qwen25_05b", "qwen25_15b", "smollm2_17b"]:
        d20 = sig[mk]["0.20"]["geometry"]["delta_logit_l2"]
        d35 = sig[mk]["0.35"]["geometry"]["delta_logit_l2"]
        b35 = beh[mk]["0.35"]["geometry"]["delta_answer_avg_neg_logprob"]
        lines.append(
            f"{_tex(mk)} & "
            f"{_fmt(d20['mean'])} & {_fmt(d20['p_value'], 4)} & "
            f"{_fmt(d35['mean'])} & {_fmt(d35['p_value'], 4)} & "
            f"{_fmt(b35['mean'])} & {_fmt(b35['p_value'], 4)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    return "\n".join(lines)


def _paper3_table() -> str:
    fair_sig = _load_json(ARTIFACTS / "paper3" / "paper3_batch_v1_fairness" / "significance_summary.json")
    fair_beh = _load_json(ARTIFACTS / "paper3" / "paper3_batch_v1_fairness" / "behavior_significance_summary.json")
    probe_sig = _load_json(ARTIFACTS / "paper3" / "paper3_batch_v1_3b" / "significance_summary.json")

    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Setting & Policy & Budget & $\Delta$Logit & $p$ & $\Delta$NLL & $p$ \\",
        r"\midrule",
        (
            "qwen25\\_15b fairness & geometry\\_keep\\_compress\\_drop & 0.24 & "
            f"{_fmt(fair_sig['qwen25_15b']['0.24']['geometry_keep_compress_drop']['delta_logit_l2']['mean'])} & "
            f"{_fmt(fair_sig['qwen25_15b']['0.24']['geometry_keep_compress_drop']['delta_logit_l2']['p_value'], 4)} & "
            f"{_fmt(fair_beh['qwen25_15b']['0.24']['geometry_keep_compress_drop']['delta_answer_avg_neg_logprob']['mean'])} & "
            f"{_fmt(fair_beh['qwen25_15b']['0.24']['geometry_keep_compress_drop']['delta_answer_avg_neg_logprob']['p_value'], 4)} \\\\"
        ),
        (
            "qwen25\\_15b fairness & geometry\\_keep\\_compress\\_drop & 0.32 & "
            f"{_fmt(fair_sig['qwen25_15b']['0.32']['geometry_keep_compress_drop']['delta_logit_l2']['mean'])} & "
            f"{_fmt(fair_sig['qwen25_15b']['0.32']['geometry_keep_compress_drop']['delta_logit_l2']['p_value'], 4)} & "
            f"{_fmt(fair_beh['qwen25_15b']['0.32']['geometry_keep_compress_drop']['delta_answer_avg_neg_logprob']['mean'])} & "
            f"{_fmt(fair_beh['qwen25_15b']['0.32']['geometry_keep_compress_drop']['delta_answer_avg_neg_logprob']['p_value'], 4)} \\\\"
        ),
        (
            "qwen25\\_15b fairness & geometry\\_keep\\_compress\\_drop & 0.50 & "
            f"{_fmt(fair_sig['qwen25_15b']['0.50']['geometry_keep_compress_drop']['delta_logit_l2']['mean'])} & "
            f"{_fmt(fair_sig['qwen25_15b']['0.50']['geometry_keep_compress_drop']['delta_logit_l2']['p_value'], 4)} & "
            f"{_fmt(fair_beh['qwen25_15b']['0.50']['geometry_keep_compress_drop']['delta_answer_avg_neg_logprob']['mean'])} & "
            f"{_fmt(fair_beh['qwen25_15b']['0.50']['geometry_keep_compress_drop']['delta_answer_avg_neg_logprob']['p_value'], 4)} \\\\"
        ),
        (
            "qwen25\\_3b probe & geometry\\_keep\\_compress\\_drop & 0.35 & "
            f"{_fmt(probe_sig['qwen25_3b']['0.35']['geometry_keep_compress_drop']['delta_logit_l2']['mean'])} & "
            f"{_fmt(probe_sig['qwen25_3b']['0.35']['geometry_keep_compress_drop']['delta_logit_l2']['p_value'], 4)} & "
            f"-- & -- \\\\"
        ),
        (
            "qwen25\\_3b probe & geometry & 0.50 & "
            f"{_fmt(probe_sig['qwen25_3b']['0.50']['geometry']['delta_logit_l2']['mean'])} & "
            f"{_fmt(probe_sig['qwen25_3b']['0.50']['geometry']['delta_logit_l2']['p_value'], 4)} & "
            f"-- & -- \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def _paper3_regime_table() -> str:
    lines = [
        r"\begin{tabular}{p{2.15cm}p{1.4cm}p{1.4cm}p{1.4cm}p{2.9cm}}",
        r"\toprule",
        r"Evaluation set & 0.20 & 0.35 & 0.50 & Reading \\",
        r"\midrule",
        r"Hard stress / fairness / 3B & KCD & KCD & geometry & Support-turn-critical memory; sparse geometry codecs help most under scarcity. \\",
        r"LongMemEval public & semantic & semantic / segment & KCD & Broader episode memory; geometry-family policies remain useful but ranking changes. \\",
        r"MSC & semantic family & semantic & semantic & Persona and conversational continuity dominate; geometry-KCD is the wrong family. \\",
        r"LoCoMo (bounded) & semantic family & semantic family & semantic family & Mixed temporal-semantic memory; semantic-led codecs remain strongest in current probes. \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def _negative_results_table() -> str:
    atlas = _load_json(ARTIFACTS / "paper1" / "regime_atlas_smoke_v4" / "atlas_summary.json")
    persona = _load_json(ARTIFACTS / "paper3" / "msc_persona_curvature_v1" / "summary.json")
    state_align = _load_json(ARTIFACTS / "paper3" / "state_update_alignment_smoke_qwen05b" / "summary.json")
    state_cross = _load_json(ARTIFACTS / "paper3" / "state_update_cross_control_qwen05b" / "summary.json")

    persona_ag = persona["aggregate"]
    align_ag = state_align["aggregate"]
    cross_ag = state_cross["aggregate"]
    retrieval_rows = [
        row for row in atlas["family_regime_summary"]
        if row["family"] == "retrieval_heavy"
    ]
    retrieval_summary = ", ".join(
        f"{row['count']} in regime {row['regime_id']}" for row in retrieval_rows
    )

    lines = [
        r"\begin{tabular}{p{2.6cm}p{2.0cm}p{3.0cm}p{2.3cm}}",
        r"\toprule",
        r"Check & Setting & Quantitative result & Reading \\",
        r"\midrule",
        (
            "MSC support vs filler curvature & "
            "5 conv. & "
            f"$\\Delta\\kappa={_fmt(persona_ag['mean_delta_curvature'])}$, "
            f"{persona_ag['positive_delta_count']}/5 positive & "
            "No robust within-topic curvature separation. \\\\"
        ),
        (
            "Regime atlas & "
            "208 segments & "
            f"retrieval\\_heavy: {retrieval_summary} & "
            "After stabilization, geometry-only regimes still mix stress and casual segments. \\\\"
        ),
        (
            "State-update alignment & "
            "10 synthetic conv. & "
            f"mean $A={_fmt(align_ag['mean_directional_alignment'])}$, "
            f"{align_ag['negative_alignment_count']}/10 negative & "
            "Same-sign increment alignment does not detect supersession. \\\\"
        ),
        (
            "State/increment cross-term & "
            "10 synthetic conv. & "
            f"update {_fmt(cross_ag['mean_state_update_entry_cross'])} vs control {_fmt(cross_ag['mean_control_entry_cross'])}; "
            f"{cross_ag['negative_state_update_entry_cross_count']}/10 negative & "
            "Only a weak ranking margin, not a usable sign-based detector. \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def _model_table() -> str:
    lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Preset & Hugging Face model & Current role \\",
        r"\midrule",
        r"qwen25\_05b & Qwen/Qwen2.5-0.5B-Instruct & Paper 1/Paper 2 local baseline \\",
        r"qwen25\_15b & Qwen/Qwen2.5-1.5B-Instruct & Main fairness-controlled model \\",
        r"qwen25\_3b & Qwen/Qwen2.5-3B-Instruct & First 3B Paper 2/Paper 3 validation \\",
        r"smollm2\_17b & HuggingFaceTB/SmolLM2-1.7B-Instruct & Non-Qwen compact control family \\",
        r"llama32\_3b & meta-llama/Llama-3.2-3B-Instruct & Non-Qwen 3B public-benchmark runner \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def _checkpoint_summary() -> dict:
    p1c = _load_json(ARTIFACTS / "paper1" / "expanded_v8_final" / "confidence_summary.json")
    p2s = _load_json(ARTIFACTS / "paper2" / "behavior_stress_v1" / "significance_summary.json")
    p3f = _load_json(ARTIFACTS / "paper3" / "paper3_batch_v1_fairness" / "significance_summary.json")
    p33 = _load_json(ARTIFACTS / "paper3" / "paper3_batch_v1_3b" / "significance_summary.json")
    return {
        "paper1": {
            mk: {
                "rank95": p1c[mk]["mean_rank95"]["estimate"],
                "corr_geodesic_logit": p1c[mk]["mean_corr_geodesic_vs_logit_l2"]["estimate"],
            }
            for mk in ["qwen25_05b", "qwen25_15b", "smollm2_17b"]
        },
        "paper2": {
            mk: {
                "delta_logit_020": p2s[mk]["0.20"]["geometry"]["delta_logit_l2"]["mean"],
                "delta_logit_035": p2s[mk]["0.35"]["geometry"]["delta_logit_l2"]["mean"],
            }
            for mk in ["qwen25_05b", "qwen25_15b", "smollm2_17b"]
        },
        "paper3": {
            "qwen25_15b_fairness": {
                "kcd_delta_logit_024": p3f["qwen25_15b"]["0.24"]["geometry_keep_compress_drop"]["delta_logit_l2"]["mean"],
                "kcd_delta_logit_032": p3f["qwen25_15b"]["0.32"]["geometry_keep_compress_drop"]["delta_logit_l2"]["mean"],
            },
            "qwen25_3b_probe": {
                "kcd_delta_logit_035": p33["qwen25_3b"]["0.35"]["geometry_keep_compress_drop"]["delta_logit_l2"]["mean"],
                "geometry_delta_logit_050": p33["qwen25_3b"]["0.50"]["geometry"]["delta_logit_l2"]["mean"],
            },
        },
    }


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    _copy_figures()
    _build_program_overview()
    _build_paper3_regime_map()
    _write(GENERATED / "table_models.tex", _model_table())
    _write(GENERATED / "table_paper1_results.tex", _paper1_table())
    _write(GENERATED / "table_paper2_results.tex", _paper2_table())
    _write(GENERATED / "table_paper3_results.tex", _paper3_table())
    _write(GENERATED / "table_paper3_regimes.tex", _paper3_regime_table())
    _write(GENERATED / "table_negative_results.tex", _negative_results_table())
    _write(GENERATED / "checkpoint_summary.json", json.dumps(_checkpoint_summary(), indent=2))


if __name__ == "__main__":
    main()
