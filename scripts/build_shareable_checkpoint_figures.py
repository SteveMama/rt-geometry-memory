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
    tmp_path = OUTPUT_DIR / "_paper2_tmp.png"
    plot_path = _plot_paper2_summary(tmp_path)
    image = Image.open(plot_path).convert("RGB")
    framed = _add_header_and_caption(
        image,
        title="Paper 2 Checkpoint: Geometry-Aware Control",
        subtitle=(
            "Geometry-guided retention remains the strongest decoder-fidelity policy on the hard stress set. "
            "The figure now emphasizes stable plots plus the support-turn mechanism instead of the earlier weak heatmap panel."
        ),
        caption=(
            "A: hard-set logit budget curves. B: geometry rescues memory-critical support turns more often than uniform on the focused qwen25_05b mechanism pass. "
            "C: cross-model low-to-mid-budget geometry gains with bootstrap confidence on the three-model confidence study. "
            "D: token-fraction curves showing that the strongest geometry gains occur under real budget pressure."
        ),
    )
    output = OUTPUT_DIR / "paper2_checkpoint_overview.png"
    framed.save(output)
    try:
        plot_path.unlink()
    except FileNotFoundError:
        pass
    return output


def _plot_paper2_summary(tmp_path: Path) -> Path:
    sig = json.loads((ARTIFACTS / "paper2" / "blazing_study_v3_confidence" / "significance_summary.json").read_text())
    mechanism = _parse_memory_report(
        ARTIFACTS / "paper2" / "behavior_stress_qwen_cases" / "memory_critical_qwen25_05b_b035.md"
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    hard_logit = Image.open(ARTIFACTS / "paper2" / "behavior_stress_v1" / "plots" / "logit_budget_curves.png").convert("RGB")
    hard_token = Image.open(ARTIFACTS / "paper2" / "behavior_stress_v1" / "plots" / "token_budget_curves.png").convert("RGB")
    axes[0, 0].imshow(np.asarray(hard_logit))
    axes[0, 0].axis("off")
    axes[0, 0].set_title("A. Hard-Set Logit Budget Curves")

    labels = ["Geometry better", "Uniform better", "Latest support rescued"]
    vals = [mechanism["better"], mechanism["worse"], mechanism["latest"]]
    colors = ["#2ca02c", "#d62728", "#1f77b4"]
    axes[0, 1].bar(labels, vals, color=colors)
    axes[0, 1].set_ylim(0, max(vals) + 4)
    axes[0, 1].set_ylabel("Cases out of 36")
    axes[0, 1].set_title("B. Support-Turn Rescue at Budget 0.35")
    axes[0, 1].tick_params(axis="x", rotation=10)

    models = ["qwen25_05b", "qwen25_15b", "smollm2_17b"]
    means = [sig[m]["0.35"]["geometry"]["delta_logit_l2"]["mean"] for m in models]
    lo = [sig[m]["0.35"]["geometry"]["delta_logit_l2"]["ci_low"] for m in models]
    hi = [sig[m]["0.35"]["geometry"]["delta_logit_l2"]["ci_high"] for m in models]
    y = np.arange(len(models))
    err_low = [m - l for m, l in zip(means, lo)]
    err_high = [h - m for m, h in zip(means, hi)]
    axes[1, 0].barh(y, means, color="#1f77b4", alpha=0.85)
    axes[1, 0].errorbar(means, y, xerr=[err_low, err_high], fmt="none", ecolor="black", capsize=4)
    axes[1, 0].axvline(0.0, color="black", linewidth=1, alpha=0.4)
    axes[1, 0].set_yticks(y, ["Qwen-0.5B", "Qwen-1.5B", "SmolLM2-1.7B"])
    axes[1, 0].set_title("C. Geometry vs Uniform at Budget 0.35")
    axes[1, 0].set_xlabel("Mean Δ logit L2 (negative is better)")

    axes[1, 1].imshow(np.asarray(hard_token))
    axes[1, 1].axis("off")
    axes[1, 1].set_title("D. Realized Token Fractions by Policy")

    fig.suptitle("Paper 2 Checkpoint: Stable Geometry-Control Evidence", fontsize=16, y=1.02)
    fig.savefig(tmp_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return tmp_path


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
    fair_eval = _read_csv(ARTIFACTS / "paper3" / "paper3_batch_v1_fairness" / "evaluation_rows.csv")
    fair_beh = _read_csv(ARTIFACTS / "paper3" / "paper3_batch_v1_fairness" / "behavior_rows.csv")
    fair_summary = json.loads((ARTIFACTS / "paper3" / "paper3_batch_v1_fairness" / "study_summary.json").read_text())
    probe_eval = _read_csv(ARTIFACTS / "paper3" / "paper3_batch_v1_3b" / "evaluation_rows.csv")
    probe_beh = _read_csv(ARTIFACTS / "paper3" / "paper3_batch_v1_3b" / "behavior_rows.csv")
    probe_summary = json.loads((ARTIFACTS / "paper3" / "paper3_batch_v1_3b" / "study_summary.json").read_text())
    fair_budgets = [0.24, 0.28, 0.32, 0.35, 0.38, 0.42, 0.46, 0.50]
    probe_budgets = [0.20, 0.35, 0.50]
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

    for policy in ["geometry", "geometry_segment_actions", "geometry_keep_compress_drop"]:
        vals = []
        for budget in fair_budgets:
            subset = [
                float(row["logit_l2"])
                for row in fair_eval
                if row["model_key"] == "qwen25_15b" and row["policy_name"] == policy and math.isclose(float(row["budget_fraction"]), budget)
            ]
            uniform = [
                float(row["logit_l2"])
                for row in fair_eval
                if row["model_key"] == "qwen25_15b" and row["policy_name"] == "uniform" and math.isclose(float(row["budget_fraction"]), budget)
            ]
            vals.append(np.mean(subset) - np.mean(uniform))
        axes[0, 0].plot(fair_budgets, vals, marker="o", color=colors[policy], alpha=0.9, label=labels[policy])
    axes[0, 0].axhline(0.0, color="black", linewidth=1, alpha=0.4)
    axes[0, 0].set_title("A. qwen25_15b Fairness Sweep")
    axes[0, 0].set_xlabel("Budget Fraction")
    axes[0, 0].set_ylabel("Mean Δ logit L2")
    axes[0, 0].legend(frameon=False)

    for policy in ["geometry", "geometry_segment_actions", "geometry_keep_compress_drop"]:
        vals = []
        for budget in probe_budgets:
            subset = [
                float(row["logit_l2"])
                for row in probe_eval
                if row["model_key"] == "qwen25_3b" and row["policy_name"] == policy and math.isclose(float(row["budget_fraction"]), budget)
            ]
            uniform = [
                float(row["logit_l2"])
                for row in probe_eval
                if row["model_key"] == "qwen25_3b" and row["policy_name"] == "uniform" and math.isclose(float(row["budget_fraction"]), budget)
            ]
            vals.append(np.mean(subset) - np.mean(uniform))
        axes[0, 1].plot(probe_budgets, vals, marker="o", color=colors[policy], label=labels[policy])
    axes[0, 1].axhline(0.0, color="black", linewidth=1, alpha=0.4)
    axes[0, 1].set_title("B. qwen25_3b Probe")
    axes[0, 1].set_xlabel("Budget Fraction")
    axes[0, 1].set_ylabel("Mean Δ logit L2")
    axes[0, 1].legend(frameon=False)

    width = 0.25
    xs = np.arange(2)
    codec_labels = ["qwen25_15b", "qwen25_3b"]
    fair_payload = fair_summary["models"]["qwen25_15b"]["aggregate"]["geometry_keep_compress_drop"]
    probe_payload = probe_summary["models"]["qwen25_3b"]["aggregate"]["geometry_keep_compress_drop"]
    payloads = [fair_payload, probe_payload]
    for idx, budget in enumerate([0.35, 0.50]):
        vals = [payload[f"{budget:.2f}"]["mean_compressed_segments"] for payload in payloads]
        axes[1, 0].bar(xs + (idx - 0.5) * width, vals, width=width, label=f"{budget:.2f}")
    axes[1, 0].set_xticks(xs, ["Qwen-1.5B", "Qwen-3B"])
    axes[1, 0].set_title("C. Active Compression in Keep/Compress/Drop")
    axes[1, 0].set_ylabel("Mean compressed segments")
    axes[1, 0].legend(frameon=False, title="Budget")

    report_specs = [
        ("qwen25_15b", "geometry_keep_compress_drop", "Qwen-1.5B K/C/D"),
        ("qwen25_3b", "geometry_keep_compress_drop", "Qwen-3B K/C/D"),
        ("qwen25_15b", "geometry_segment_actions", "Qwen-1.5B Seg"),
        ("qwen25_3b", "geometry_segment_actions", "Qwen-3B Seg"),
    ]
    better = []
    worse = []
    latest = []
    names = []
    for model, suffix, label in report_specs:
        folder = "paper3_batch_v1_fairness" if model == "qwen25_15b" else "paper3_batch_v1_3b"
        path = ARTIFACTS / "paper3" / folder / f"memory_critical_{model}_{'segment_actions' if suffix == 'geometry_segment_actions' else 'keep_compress_drop'}_b035.md"
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
