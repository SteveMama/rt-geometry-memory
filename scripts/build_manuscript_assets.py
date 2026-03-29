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
            "A segment anchor plus sparse support memory enables keep/compress/drop decisions. The low/mid-budget winner is geometry_keep_compress_drop; plain geometry retakes the lead at looser budgets.",
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
    _write(GENERATED / "table_models.tex", _model_table())
    _write(GENERATED / "table_paper1_results.tex", _paper1_table())
    _write(GENERATED / "table_paper2_results.tex", _paper2_table())
    _write(GENERATED / "table_paper3_results.tex", _paper3_table())
    _write(GENERATED / "checkpoint_summary.json", json.dumps(_checkpoint_summary(), indent=2))


if __name__ == "__main__":
    main()
