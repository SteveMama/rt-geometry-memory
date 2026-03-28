from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT_DIR = ARTIFACTS / "shareable"


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if len(trial) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _add_header_and_caption(
    canvas: Image.Image,
    *,
    title: str,
    subtitle: str,
    caption: str,
) -> Image.Image:
    header_h = 210
    footer_h = 170
    framed = Image.new("RGB", (canvas.width, canvas.height + header_h + footer_h), "white")
    framed.paste(canvas, (0, header_h))
    draw = ImageDraw.Draw(framed)
    title_font = _load_font(42)
    subtitle_font = _load_font(24)
    caption_font = _load_font(24)
    draw.text((40, 28), title, fill="black", font=title_font)
    for idx, line in enumerate(_wrap_text(subtitle, 88)):
        draw.text((40, 92 + idx * 28), line, fill=(35, 35, 35), font=subtitle_font)
    footer_y = canvas.height + header_h + 18
    for idx, line in enumerate(_wrap_text(caption, 110)):
        draw.text((40, footer_y + idx * 28), line, fill=(45, 45, 45), font=caption_font)
    return framed


def _compose_existing_plots() -> Path:
    inputs = [
        (
            "A. Competitor Matrix",
            ARTIFACTS / "paper2" / "next_phase_v1_paper2_competitor_matrix" / "plots" / "logit_budget_curves.png",
        ),
        (
            "B. Family Heatmap",
            ARTIFACTS / "paper2" / "next_phase_v1_paper2_competitor_matrix" / "plots" / "family_logit_heatmap.png",
        ),
        (
            "C. 3B Probe",
            ARTIFACTS / "paper2" / "next_phase_v1_paper2_3b_probe" / "plots" / "logit_budget_curves.png",
        ),
        (
            "D. Fairness Sweep",
            ARTIFACTS / "paper2" / "next_phase_v1_paper2_fairness_sweep" / "plots" / "token_budget_curves.png",
        ),
    ]
    images = []
    label_font = _load_font(30)
    for label, path in inputs:
        image = Image.open(path).convert("RGB")
        image = image.resize((900, int(image.height * (900 / image.width))))
        panel = Image.new("RGB", (920, image.height + 64), "white")
        panel.paste(image, (10, 54))
        draw = ImageDraw.Draw(panel)
        draw.text((10, 10), label, fill="black", font=label_font)
        images.append(panel)
    panel_w = max(img.width for img in images)
    panel_h = max(img.height for img in images)
    canvas = Image.new("RGB", (panel_w * 2 + 30, panel_h * 2 + 30), "white")
    positions = [(0, 0), (panel_w + 30, 0), (0, panel_h + 30), (panel_w + 30, panel_h + 30)]
    for image, (x, y) in zip(images, positions):
        canvas.paste(image, (x, y))
    framed = _add_header_and_caption(
        canvas,
        title="Paper 2 Checkpoint: Geometry vs Competitors",
        subtitle=(
            "Geometry-guided retention remains the strongest decoder-fidelity policy on the hard stress set. "
            "The 3B probe is positive, and the fairness sweep shows the clearest gains in the low-to-mid budget regime."
        ),
        caption=(
            "A: cross-model competitor matrix on the hard stress set. B: family-level logit drift at the mid budget. "
            "C: first 3B generalization check. D: fairness sweep token fractions, showing that the strongest low-budget geometry gains occur when realized token gaps are still small."
        ),
    )
    output = OUTPUT_DIR / "paper2_checkpoint_overview.png"
    framed.save(output)
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_memory_report(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    fields = {
        "better": r"retains more support user turns than uniform: (\d+)",
        "worse": r"Uniform retains more support user turns than .*: (\d+)",
        "latest": r"keeps the latest support turn while uniform drops it: (\d+)",
        "not_worse": r"Compressed cases that are not worse than uniform on support retention: (\d+)",
    }
    out: dict[str, int] = {}
    for key, pattern in fields.items():
        match = re.search(pattern, text)
        out[key] = int(match.group(1)) if match else 0
    return out


def _plot_paper3_summary(tmp_path: Path) -> Path:
    eval_rows = _read_csv(ARTIFACTS / "paper3" / "next_phase_v1_paper3_head_to_head" / "evaluation_rows.csv")
    beh_rows = _read_csv(ARTIFACTS / "paper3" / "next_phase_v1_paper3_head_to_head" / "behavior_rows.csv")
    summary = json.loads((ARTIFACTS / "paper3" / "next_phase_v1_paper3_head_to_head" / "study_summary.json").read_text())
    models = ["qwen25_05b", "qwen25_15b", "smollm2_17b"]
    budgets = [0.20, 0.35, 0.50]
    colors = {
        "geometry": "#1f77b4",
        "geometry_segment_actions": "#d62728",
        "geometry_keep_compress_drop": "#2ca02c",
    }
    labels = {
        "geometry": "Geometry",
        "geometry_segment_actions": "Segment Actions",
        "geometry_keep_compress_drop": "Keep/Compress/Drop",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    for model in models:
        x = np.asarray(budgets)
        yvals = []
        for policy in ["geometry", "geometry_segment_actions", "geometry_keep_compress_drop"]:
            vals = []
            for budget in budgets:
                subset = [
                    float(row["logit_l2"])
                    for row in eval_rows
                    if row["model_key"] == model and row["policy_name"] == policy and math.isclose(float(row["budget_fraction"]), budget)
                ]
                uniform = [
                    float(row["logit_l2"])
                    for row in eval_rows
                    if row["model_key"] == model and row["policy_name"] == "uniform" and math.isclose(float(row["budget_fraction"]), budget)
                ]
                vals.append(np.mean(subset) - np.mean(uniform))
            axes[0, 0].plot(x, vals, marker="o", color=colors[policy], alpha=0.55)
            yvals.append((policy, vals))
        # annotate only once per model with best 0.35
        best_policy, best_vals = min(yvals, key=lambda item: item[1][1])
        axes[0, 0].text(0.355, best_vals[1], model, fontsize=9, color="black")
    axes[0, 0].axhline(0.0, color="black", linewidth=1, alpha=0.4)
    axes[0, 0].set_title("A. Logit Delta vs Uniform")
    axes[0, 0].set_xlabel("Budget Fraction")
    axes[0, 0].set_ylabel("Mean Δ logit L2")

    for policy in ["geometry", "geometry_segment_actions", "geometry_keep_compress_drop"]:
        vals = []
        for budget in budgets:
            per_model = []
            for model in models:
                subset = [
                    float(row["answer_avg_neg_logprob"])
                    for row in beh_rows
                    if row["model_key"] == model and row["policy_name"] == policy and math.isclose(float(row["budget_fraction"]), budget)
                ]
                uniform = [
                    float(row["answer_avg_neg_logprob"])
                    for row in beh_rows
                    if row["model_key"] == model and row["policy_name"] == "uniform" and math.isclose(float(row["budget_fraction"]), budget)
                ]
                per_model.append(np.mean(subset) - np.mean(uniform))
            vals.append(np.mean(per_model))
        axes[0, 1].plot(budgets, vals, marker="o", color=colors[policy], label=labels[policy])
    axes[0, 1].axhline(0.0, color="black", linewidth=1, alpha=0.4)
    axes[0, 1].set_title("B. Mean Behavior Delta vs Uniform")
    axes[0, 1].set_xlabel("Budget Fraction")
    axes[0, 1].set_ylabel("Mean Δ answer NLL")
    axes[0, 1].legend(frameon=False)

    width = 0.22
    xs = np.arange(len(models))
    for idx, budget in enumerate(budgets):
        vals = []
        for model in models:
            payload = summary["models"][model]["aggregate"]["geometry_keep_compress_drop"][f"{budget:.2f}"]
            vals.append(payload["mean_compressed_segments"])
        axes[1, 0].bar(xs + (idx - 1) * width, vals, width=width, label=f"{budget:.2f}")
    axes[1, 0].set_xticks(xs, models)
    axes[1, 0].set_title("C. Active Compression in Keep/Compress/Drop")
    axes[1, 0].set_ylabel("Mean compressed segments")
    axes[1, 0].legend(frameon=False, title="Budget")

    report_specs = [
        ("qwen25_05b", "geometry_keep_compress_drop", "Qwen-0.5B K/C/D"),
        ("qwen25_15b", "geometry_keep_compress_drop", "Qwen-1.5B K/C/D"),
        ("smollm2_17b", "geometry_keep_compress_drop", "SmolLM2 K/C/D"),
        ("qwen25_15b", "geometry_segment_actions", "Qwen-1.5B Seg"),
    ]
    better = []
    worse = []
    latest = []
    names = []
    for model, suffix, label in report_specs:
        path = ARTIFACTS / "paper3" / "next_phase_v1_paper3_head_to_head" / f"memory_critical_{model}_{'segment_actions' if suffix == 'geometry_segment_actions' else 'keep_compress_drop'}_b035.md"
        payload = _parse_memory_report(path)
        names.append(label)
        better.append(payload["better"])
        worse.append(payload["worse"])
        latest.append(payload["latest"])
    xs = np.arange(len(names))
    axes[1, 1].bar(xs - width, better, width=width, color="#2ca02c", label="Better than uniform")
    axes[1, 1].bar(xs, latest, width=width, color="#1f77b4", label="Latest support rescued")
    axes[1, 1].bar(xs + width, worse, width=width, color="#d62728", label="Worse than uniform")
    axes[1, 1].set_xticks(xs, names, rotation=15, ha="right")
    axes[1, 1].set_title("D. Memory-Critical Mechanism at Budget 0.35")
    axes[1, 1].set_ylabel("Cases out of 36")
    axes[1, 1].legend(frameon=False)

    fig.suptitle("Paper 3 Checkpoint: Two Viable Geometry-Driven Compression Families", fontsize=16, y=1.02)
    fig.savefig(tmp_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return tmp_path


def _build_paper3_overview() -> Path:
    tmp_path = OUTPUT_DIR / "_paper3_tmp.png"
    plot_path = _plot_paper3_summary(tmp_path)
    image = Image.open(plot_path).convert("RGB")
    framed = _add_header_and_caption(
        image,
        title="Paper 3 Checkpoint: Compression Is Real",
        subtitle=(
            "Both geometry_segment_actions and geometry_keep_compress_drop are viable. "
            "Segment actions are stronger at some higher-budget logit settings; keep/compress/drop is the cleaner low-to-mid-budget mechanism policy and preserves memory-critical support structure more often."
        ),
        caption=(
            "A: mean logit improvement over uniform across models. B: mean answer-NLL improvement over uniform. "
            "C: the keep/compress/drop codec now uses compression actively and more often as budget increases. "
            "D: at budget 0.35, keep/compress/drop has a stronger support-turn rescue mechanism than uniform, especially on the Qwen models."
        ),
    )
    output = OUTPUT_DIR / "paper3_checkpoint_overview.png"
    framed.save(output)
    try:
        plot_path.unlink()
    except FileNotFoundError:
        pass
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paper2 = _compose_existing_plots()
    paper3 = _build_paper3_overview()
    print(f"Wrote {paper2}")
    print(f"Wrote {paper3}")


if __name__ == "__main__":
    main()
