from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np

from .conversations import ConversationRecord
from .geometry import (
    EPS,
    boundary_score_series,
    choose_segments,
    curvature_series,
    effective_rank,
    low_rank_project,
    normalize_rows,
    rank_jump_series,
    segment_reference,
    sphere_distance,
    subspace_shift_series,
    transported_increment_matrix,
    turning_angle_series,
)
from .modeling import TrajectoryBatch


FEATURE_NAMES: tuple[str, ...] = (
    "mean_curvature",
    "std_curvature",
    "skew_curvature",
    "max_curvature",
    "mean_turning_angle",
    "mean_step_norm",
    "std_step_norm",
    "max_step_norm",
    "rank95",
    "mean_rank_jump",
    "mean_subspace_shift",
    "mean_boundary_score",
    "role_switch_rate",
    "switch_step_norm",
    "segment_turn_count",
)

LOG_FEATURES: frozenset[str] = frozenset(
    {
        "mean_curvature",
        "std_curvature",
        "max_curvature",
        "mean_rank_jump",
        "mean_subspace_shift",
        "mean_boundary_score",
    }
)


@dataclass(slots=True)
class AtlasSegmentRow:
    conversation_id: str
    family: str
    segment_index: int
    start_turn: int
    end_turn: int
    start_role: str
    end_role: str
    segment_turn_count: int
    mean_curvature: float
    std_curvature: float
    skew_curvature: float
    max_curvature: float
    mean_turning_angle: float
    mean_step_norm: float
    std_step_norm: float
    max_step_norm: float
    rank95: int
    mean_rank_jump: float
    mean_subspace_shift: float
    mean_boundary_score: float
    role_switch_rate: float
    switch_step_norm: float
    example_excerpt: str
    regime_id: int = -1
    regime_name: str = ""


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def _safe_skew(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    std = float(np.std(values))
    if std < EPS:
        return 0.0
    centered = (values - float(np.mean(values))) / std
    return float(np.mean(np.power(centered, 3)))


def _series_slice(values: np.ndarray, start_turn: int, end_turn: int) -> np.ndarray:
    left = max(start_turn - 1, 0)
    right = max(end_turn - 1, 0)
    if right <= left:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(values[left:right], dtype=np.float32)


def _segment_excerpt(conversation: ConversationRecord, start_turn: int, end_turn: int) -> str:
    selected = conversation.turns[start_turn : end_turn + 1]
    if not selected:
        return ""
    excerpt_turns = selected[:2]
    if len(selected) > 4:
        excerpt_turns = selected[:2] + selected[-2:]
    elif len(selected) > 2:
        excerpt_turns = selected
    parts = [
        f"{turn.role}: {turn.content[:120].replace(chr(10), ' | ')}"
        for turn in excerpt_turns
    ]
    return " || ".join(parts)


def extract_segment_rows(
    conversation: ConversationRecord,
    batch: TrajectoryBatch,
    *,
    rank_energy: float,
    max_segment_len: int,
    min_segment_len: int,
) -> list[AtlasSegmentRow]:
    if batch.states.shape[0] < 2:
        return []

    unit_states, _ = normalize_rows(np.asarray(batch.states, dtype=np.float32))
    curvatures = curvature_series(unit_states)
    turning_angles = turning_angle_series(unit_states)
    boundary_scores = boundary_score_series(unit_states)
    rank_jumps = rank_jump_series(unit_states, rank_energy=rank_energy)
    subspace_shifts = subspace_shift_series(unit_states, rank_energy=rank_energy)
    segments = choose_segments(
        unit_states=unit_states,
        curvatures=curvatures,
        boundary_scores=boundary_scores,
        turning_angles=turning_angles,
        rank_energy=rank_energy,
        max_segment_len=max_segment_len,
        min_segment_len=min_segment_len,
    )

    rows: list[AtlasSegmentRow] = []
    for segment_index, (start_turn, end_turn) in enumerate(segments):
        reference = segment_reference(unit_states, start_turn, end_turn)
        steps = transported_increment_matrix(unit_states, start_turn, end_turn, reference)
        _, _, singular_values = low_rank_project(steps, rank=max(1, min(steps.shape))) if steps.size else (
            steps.copy(),
            np.zeros((unit_states.shape[1], 0), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
        rank95 = max(effective_rank(singular_values, rank_energy), 1) if singular_values.size else 0

        curvature_slice = _series_slice(curvatures, start_turn, end_turn)
        turning_slice = _series_slice(turning_angles, start_turn, end_turn)
        rank_jump_slice = _series_slice(rank_jumps, start_turn, end_turn)
        subspace_slice = _series_slice(subspace_shifts, start_turn, end_turn)
        boundary_slice = _series_slice(boundary_scores, start_turn, end_turn)

        step_norms = np.asarray(
            [
                sphere_distance(unit_states[idx], unit_states[idx + 1])
                for idx in range(start_turn, end_turn)
            ],
            dtype=np.float32,
        )
        role_switch_values = np.asarray(
            [
                step_norms[idx - start_turn - 1]
                for idx in range(start_turn + 1, end_turn + 1)
                if batch.turn_roles[idx] != batch.turn_roles[idx - 1]
            ],
            dtype=np.float32,
        )
        switch_count = int(role_switch_values.size)
        step_count = max(end_turn - start_turn, 1)

        rows.append(
            AtlasSegmentRow(
                conversation_id=conversation.conversation_id,
                family=conversation.family,
                segment_index=segment_index,
                start_turn=start_turn,
                end_turn=end_turn,
                start_role=batch.turn_roles[start_turn],
                end_role=batch.turn_roles[end_turn],
                segment_turn_count=end_turn - start_turn + 1,
                mean_curvature=_safe_mean(curvature_slice),
                std_curvature=_safe_std(curvature_slice),
                skew_curvature=_safe_skew(curvature_slice),
                max_curvature=_safe_max(curvature_slice),
                mean_turning_angle=_safe_mean(turning_slice),
                mean_step_norm=_safe_mean(step_norms),
                std_step_norm=_safe_std(step_norms),
                max_step_norm=_safe_max(step_norms),
                rank95=rank95,
                mean_rank_jump=_safe_mean(rank_jump_slice),
                mean_subspace_shift=_safe_mean(subspace_slice),
                mean_boundary_score=_safe_mean(boundary_slice),
                role_switch_rate=float(switch_count / step_count),
                switch_step_norm=_safe_mean(role_switch_values),
                example_excerpt=_segment_excerpt(conversation, start_turn, end_turn),
            )
        )
    return rows


def _feature_matrix(rows: list[AtlasSegmentRow]) -> np.ndarray:
    matrix = np.asarray(
        [
            [
                float(np.log1p(max(float(getattr(row, name)), 0.0))) if name in LOG_FEATURES else float(getattr(row, name))
                for name in FEATURE_NAMES
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    if matrix.size == 0:
        return matrix
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    safe_std = np.where(std < 1e-6, 1.0, std)
    return (matrix - mean) / safe_std


def _kmeans_pp_init(matrix: np.ndarray, cluster_count: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.zeros((cluster_count, matrix.shape[1]), dtype=np.float32)
    first_idx = int(rng.integers(0, matrix.shape[0]))
    centers[0] = matrix[first_idx]
    min_dist2 = np.sum(np.square(matrix - centers[0]), axis=1)
    for center_idx in range(1, cluster_count):
        total = float(np.sum(min_dist2))
        if total < 1e-8:
            pick = int(rng.integers(0, matrix.shape[0]))
        else:
            probs = min_dist2 / total
            pick = int(rng.choice(matrix.shape[0], p=probs))
        centers[center_idx] = matrix[pick]
        dist2 = np.sum(np.square(matrix - centers[center_idx]), axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return centers


def cluster_segment_rows(
    rows: list[AtlasSegmentRow],
    *,
    cluster_count: int,
    seed: int = 0,
    max_iters: int = 50,
) -> tuple[np.ndarray, dict[int, dict[str, object]]]:
    if not rows:
        return np.zeros(0, dtype=np.int32), {}

    matrix = _feature_matrix(rows)
    use_clusters = max(1, min(cluster_count, matrix.shape[0]))
    rng = np.random.default_rng(seed)
    centers = _kmeans_pp_init(matrix, use_clusters, rng)
    assignments = np.zeros(matrix.shape[0], dtype=np.int32)

    for _ in range(max_iters):
        dist2 = np.sum(np.square(matrix[:, None, :] - centers[None, :, :]), axis=2)
        next_assignments = np.argmin(dist2, axis=1).astype(np.int32)
        if np.array_equal(assignments, next_assignments):
            break
        assignments = next_assignments
        for cluster_idx in range(use_clusters):
            mask = assignments == cluster_idx
            if not np.any(mask):
                centers[cluster_idx] = matrix[int(rng.integers(0, matrix.shape[0]))]
            else:
                centers[cluster_idx] = matrix[mask].mean(axis=0)

    summaries: dict[int, dict[str, object]] = {}
    for cluster_idx in range(use_clusters):
        cluster_rows = [row for row, assignment in zip(rows, assignments, strict=True) if assignment == cluster_idx]
        family_counts: dict[str, int] = {}
        for row in cluster_rows:
            family_counts[row.family] = family_counts.get(row.family, 0) + 1
        centroid = {
            feature_name: float(np.mean([float(getattr(row, feature_name)) for row in cluster_rows]))
            for feature_name in FEATURE_NAMES
        }
        summaries[cluster_idx] = {
            "cluster_id": cluster_idx,
            "count": len(cluster_rows),
            "family_counts": dict(sorted(family_counts.items(), key=lambda item: (-item[1], item[0]))),
            "centroid": centroid,
            "description": describe_cluster(centroid),
        }

    for row, assignment in zip(rows, assignments, strict=True):
        row.regime_id = int(assignment)
        row.regime_name = str(summaries[int(assignment)]["description"])
    return assignments, summaries


def describe_cluster(centroid: dict[str, float]) -> str:
    curvature = centroid.get("mean_curvature", 0.0)
    curvature_std = centroid.get("std_curvature", 0.0)
    rank_jump = centroid.get("mean_rank_jump", 0.0)
    subspace = centroid.get("mean_subspace_shift", 0.0)
    step = centroid.get("mean_step_norm", 0.0)
    role_switch = centroid.get("role_switch_rate", 0.0)
    boundary = centroid.get("mean_boundary_score", 0.0)

    if step < 0.05 and role_switch > 0.9:
        return "near_stationary_fact_memory"
    if curvature < 0.35 and curvature_std < 0.25 and subspace < 0.2:
        return "stable_low_curvature"
    if (boundary > 1.0 and step > 0.4) or (curvature > 0.75 and subspace > 0.35):
        return "curvature_spike_transition"
    if rank_jump > 0.35 or subspace > 0.6:
        return "structural_transition"
    if step > 0.45 and role_switch > 0.45:
        return "dialogue_exchange_flow"
    return "mixed_fact_flow"


def family_regime_counts(rows: list[AtlasSegmentRow]) -> list[dict[str, object]]:
    counts: dict[tuple[str, int, str], int] = {}
    family_totals: dict[str, int] = {}
    for row in rows:
        key = (row.family, row.regime_id, row.regime_name)
        counts[key] = counts.get(key, 0) + 1
        family_totals[row.family] = family_totals.get(row.family, 0) + 1
    output: list[dict[str, object]] = []
    for (family, regime_id, regime_name), count in sorted(counts.items()):
        total = max(family_totals.get(family, 1), 1)
        output.append(
            {
                "family": family,
                "regime_id": regime_id,
                "regime_name": regime_name,
                "count": count,
                "fraction": float(count / total),
            }
        )
    return output


def build_atlas_report(
    rows: list[AtlasSegmentRow],
    cluster_summaries: dict[int, dict[str, object]],
    family_rows: list[dict[str, object]],
    *,
    model_key: str,
    input_paths: list[str],
    cluster_count: int,
    max_segment_len: int,
    min_segment_len: int,
) -> str:
    lines = [
        "# Geometric Regime Atlas Report",
        "",
        f"- Model: `{model_key}`",
        f"- Input paths: {', '.join(input_paths)}",
        f"- Segments: {len(rows)}",
        f"- Requested clusters: {cluster_count}",
        f"- Segment length bounds: {min_segment_len}..{max_segment_len}",
        "",
        "## Regimes",
        "",
    ]
    for cluster_id, summary in sorted(cluster_summaries.items()):
        lines.extend(
            [
                f"### Regime {cluster_id}: {summary['description']}",
                "",
                f"- Segments: {summary['count']}",
                f"- Family counts: {summary['family_counts']}",
                "- Centroid stats:",
                f"  - mean_curvature={summary['centroid']['mean_curvature']:.3f}",
                f"  - std_curvature={summary['centroid']['std_curvature']:.3f}",
                f"  - max_curvature={summary['centroid']['max_curvature']:.3f}",
                f"  - rank95={summary['centroid']['rank95']:.3f}",
                f"  - mean_rank_jump={summary['centroid']['mean_rank_jump']:.3f}",
                f"  - mean_subspace_shift={summary['centroid']['mean_subspace_shift']:.3f}",
                f"  - role_switch_rate={summary['centroid']['role_switch_rate']:.3f}",
                "",
                "- Example segments:",
            ]
        )
        examples = [row for row in rows if row.regime_id == cluster_id][:3]
        for row in examples:
            lines.append(
                f"  - `{row.family}` `{row.conversation_id}` turns {row.start_turn}-{row.end_turn}: {row.example_excerpt}"
            )
        lines.append("")

    lines.extend(
        [
            "## Family Regime Distribution",
            "",
            "| Family | Regime | Count | Fraction |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in family_rows:
        lines.append(
            f"| {row['family']} | {row['regime_id']} {row['regime_name']} | {row['count']} | {row['fraction']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def write_atlas_outputs(
    output_dir: Path,
    rows: list[AtlasSegmentRow],
    cluster_summaries: dict[int, dict[str, object]],
    family_rows: list[dict[str, object]],
    report_text: str,
    *,
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "segment_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "conversation_id",
                "family",
                "segment_index",
                "start_turn",
                "end_turn",
                "start_role",
                "end_role",
                "segment_turn_count",
                *FEATURE_NAMES,
                "regime_id",
                "regime_name",
                "example_excerpt",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.conversation_id,
                    row.family,
                    row.segment_index,
                    row.start_turn,
                    row.end_turn,
                    row.start_role,
                    row.end_role,
                    row.segment_turn_count,
                    *[getattr(row, name) for name in FEATURE_NAMES],
                    row.regime_id,
                    row.regime_name,
                    row.example_excerpt,
                ]
            )

    with (output_dir / "family_regime_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["family", "regime_id", "regime_name", "count", "fraction"],
        )
        writer.writeheader()
        writer.writerows(family_rows)

    summary_payload = {
        "metadata": metadata,
        "cluster_summaries": cluster_summaries,
        "family_regime_summary": family_rows,
        "num_segments": len(rows),
    }
    (output_dir / "atlas_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(report_text, encoding="utf-8")
