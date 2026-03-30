from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import csv
import matplotlib.pyplot as plt
import numpy as np

from .conversations import ConversationRecord
from .geometry import (
    CURVATURE_ARCLENGTH_FLOOR,
    curvature_series,
    normalize_rows,
    sphere_distance,
    stabilized_curvature_series,
    turning_angle_series,
)
from .modeling import TrajectoryBatch
from .regime_atlas import AtlasSegmentRow


@dataclass(slots=True)
class ConversationSeriesDiagnostic:
    conversation_id: str
    family: str
    num_turns_used: int
    raw_curvatures: np.ndarray
    stabilized_curvatures: np.ndarray
    turning_angles: np.ndarray
    step_norms: np.ndarray
    excerpt: str


def build_conversation_series_diagnostic(
    conversation: ConversationRecord,
    batch: TrajectoryBatch,
) -> ConversationSeriesDiagnostic:
    states = np.asarray(batch.states, dtype=np.float32)
    unit_states, _ = normalize_rows(states)
    step_norms = np.asarray(
        [sphere_distance(unit_states[idx], unit_states[idx + 1]) for idx in range(max(unit_states.shape[0] - 1, 0))],
        dtype=np.float32,
    )
    raw_curvatures = curvature_series(unit_states)
    stabilized_curvatures = stabilized_curvature_series(
        unit_states,
        min_arclength=CURVATURE_ARCLENGTH_FLOOR,
    )
    turning_angles = turning_angle_series(unit_states)
    excerpt_turns = conversation.turns[: min(len(conversation.turns), batch.states.shape[0], 4)]
    excerpt = " || ".join(
        f"{turn.role}: {turn.content[:120].replace(chr(10), ' | ')}" for turn in excerpt_turns
    )
    return ConversationSeriesDiagnostic(
        conversation_id=conversation.conversation_id,
        family=conversation.family,
        num_turns_used=int(batch.states.shape[0]),
        raw_curvatures=raw_curvatures,
        stabilized_curvatures=stabilized_curvatures,
        turning_angles=turning_angles,
        step_norms=step_norms,
        excerpt=excerpt,
    )


def _family_priority_key(family: str) -> tuple[int, str]:
    priorities = {
        "msc_valid": 0,
        "locomo10": 1,
        "longmemeval_s_cleaned": 2,
        "retrieval_heavy": 3,
        "long_dependency": 4,
    }
    return (priorities.get(family, 99), family)


def select_representative_diagnostics(
    diagnostics: list[ConversationSeriesDiagnostic],
) -> list[ConversationSeriesDiagnostic]:
    selected: list[ConversationSeriesDiagnostic] = []
    seen: set[str] = set()
    for item in sorted(diagnostics, key=lambda row: (_family_priority_key(row.family), row.conversation_id)):
        if item.family in seen:
            continue
        selected.append(item)
        seen.add(item.family)
    return selected


def plot_representative_series(
    diagnostics: list[ConversationSeriesDiagnostic],
    output_path: Path,
    *,
    log_curvature: bool,
) -> None:
    if not diagnostics:
        return
    fig, axes = plt.subplots(
        len(diagnostics),
        2,
        figsize=(13, max(3.2, 2.8 * len(diagnostics))),
        squeeze=False,
    )
    for row_idx, diagnostic in enumerate(diagnostics):
        curvature_axis = axes[row_idx, 0]
        step_axis = axes[row_idx, 1]

        if diagnostic.raw_curvatures.size:
            curvature_axis.plot(
                np.arange(1, diagnostic.raw_curvatures.size + 1),
                diagnostic.raw_curvatures,
                marker="o",
                linewidth=1.4,
                label="raw",
            )
            curvature_axis.plot(
                np.arange(1, diagnostic.stabilized_curvatures.size + 1),
                diagnostic.stabilized_curvatures,
                marker="o",
                linewidth=1.2,
                alpha=0.8,
                label="stabilized",
            )
        if log_curvature:
            curvature_axis.set_yscale("symlog", linthresh=1e-3)
        curvature_axis.set_title(f"{diagnostic.family} / {diagnostic.conversation_id} curvature")
        curvature_axis.set_xlabel("Interior turn index")
        curvature_axis.set_ylabel("Curvature")
        curvature_axis.grid(alpha=0.25)
        curvature_axis.legend(loc="best", fontsize=8)

        if diagnostic.step_norms.size:
            step_axis.plot(
                np.arange(1, diagnostic.step_norms.size + 1),
                diagnostic.step_norms,
                marker="o",
                linewidth=1.4,
                color="#c45508",
            )
        step_axis.set_yscale("symlog", linthresh=1e-4)
        step_axis.set_title(f"{diagnostic.family} / {diagnostic.conversation_id} step norms")
        step_axis.set_xlabel("Turn step index")
        step_axis.set_ylabel("Geodesic step norm")
        step_axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_conversation_series_summary(
    diagnostics: list[ConversationSeriesDiagnostic],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "conversation_id",
                "family",
                "num_turns_used",
                "raw_mean_curvature",
                "raw_std_curvature",
                "raw_max_curvature",
                "stabilized_mean_curvature",
                "stabilized_std_curvature",
                "stabilized_max_curvature",
                "mean_step_norm",
                "min_step_norm",
                "max_step_norm",
                "raw_curvature_over_100_count",
                "raw_curvature_over_1000_count",
                "stabilized_curvature_over_10_count",
                "stabilized_curvature_over_25_count",
                "step_norm_below_1e-3_count",
                "step_norm_below_1e-2_count",
                "excerpt",
            ]
        )
        for item in diagnostics:
            writer.writerow(
                [
                    item.conversation_id,
                    item.family,
                    item.num_turns_used,
                    float(np.mean(item.raw_curvatures)) if item.raw_curvatures.size else 0.0,
                    float(np.std(item.raw_curvatures)) if item.raw_curvatures.size else 0.0,
                    float(np.max(item.raw_curvatures)) if item.raw_curvatures.size else 0.0,
                    float(np.mean(item.stabilized_curvatures)) if item.stabilized_curvatures.size else 0.0,
                    float(np.std(item.stabilized_curvatures)) if item.stabilized_curvatures.size else 0.0,
                    float(np.max(item.stabilized_curvatures)) if item.stabilized_curvatures.size else 0.0,
                    float(np.mean(item.step_norms)) if item.step_norms.size else 0.0,
                    float(np.min(item.step_norms)) if item.step_norms.size else 0.0,
                    float(np.max(item.step_norms)) if item.step_norms.size else 0.0,
                    int(np.sum(item.raw_curvatures > 100.0)),
                    int(np.sum(item.raw_curvatures > 1000.0)),
                    int(np.sum(item.stabilized_curvatures > 10.0)),
                    int(np.sum(item.stabilized_curvatures > 25.0)),
                    int(np.sum(item.step_norms < 1e-3)),
                    int(np.sum(item.step_norms < 1e-2)),
                    item.excerpt,
                ]
            )


def build_saturation_report(
    rows: list[AtlasSegmentRow],
    diagnostics: list[ConversationSeriesDiagnostic],
) -> str:
    suspicious_rows = [
        row
        for row in rows
        if row.mean_step_norm < 1e-3 and row.raw_mean_curvature > 100.0
    ]
    suspicious_rows.sort(key=lambda row: row.raw_mean_curvature, reverse=True)

    family_counts: dict[str, int] = {}
    for row in suspicious_rows:
        family_counts[row.family] = family_counts.get(row.family, 0) + 1

    lines = [
        "# Curvature Saturation Audit",
        "",
        "This report checks whether extreme curvature values are being driven by",
        "near-zero local step norms rather than meaningful geometric spikes.",
        "",
        f"- Conversations audited: {len(diagnostics)}",
        f"- Segments audited: {len(rows)}",
        f"- Suspicious segments (`mean_step_norm < 1e-3` and raw `mean_curvature > 100`): {len(suspicious_rows)}",
        f"- Suspicious family counts: {dict(sorted(family_counts.items()))}",
        "",
        "## Conversation-level summary",
        "",
        "| Family | Conversation | Raw mean curvature | Raw std curvature | Stabilized mean curvature | Mean step norm | Raw curvature > 1000 | Step norms < 1e-3 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in diagnostics:
        lines.append(
            f"| {item.family} | {item.conversation_id} | "
            f"{(float(np.mean(item.raw_curvatures)) if item.raw_curvatures.size else 0.0):.3f} | "
            f"{(float(np.std(item.raw_curvatures)) if item.raw_curvatures.size else 0.0):.3f} | "
            f"{(float(np.mean(item.stabilized_curvatures)) if item.stabilized_curvatures.size else 0.0):.3f} | "
            f"{(float(np.mean(item.step_norms)) if item.step_norms.size else 0.0):.6f} | "
            f"{int(np.sum(item.raw_curvatures > 1000.0))} | "
            f"{int(np.sum(item.step_norms < 1e-3))} |"
        )

    lines.extend(
        [
            "",
            "## Top suspicious segments",
            "",
            "| Family | Conversation | Turns | Raw mean curvature | Raw std curvature | Stabilized mean curvature | Mean step norm | Excerpt |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in suspicious_rows[:20]:
        lines.append(
            f"| {row.family} | {row.conversation_id} | {row.start_turn}-{row.end_turn} | "
            f"{row.raw_mean_curvature:.3f} | {row.raw_std_curvature:.3f} | {row.mean_curvature:.3f} | {row.mean_step_norm:.6f} | "
            f"{row.example_excerpt.replace('|', '\\|')} |"
        )

    return "\n".join(lines) + "\n"
