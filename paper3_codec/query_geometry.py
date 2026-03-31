from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paper1_geometry.geometry import (
    EPS,
    effective_rank,
    normalize_rows,
    segment_reference,
    sphere_log_map,
    sphere_parallel_transport,
    stabilized_curvature_series,
    low_rank_project,
    transported_increment_matrix,
)


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


@dataclass(frozen=True, slots=True)
class QueryConditionedGeometry:
    projected_curvature: np.ndarray
    projected_subspace_energy: np.ndarray
    query_alignment: np.ndarray
    risk: np.ndarray


@dataclass(frozen=True, slots=True)
class QueryConditionedGeometryV2:
    projected_curvature: np.ndarray
    projected_subspace_energy: np.ndarray
    query_alignment: np.ndarray
    segment_rank95: np.ndarray
    segment_mean_step_norm: np.ndarray
    segment_mean_stabilized_curvature: np.ndarray
    local_projection: np.ndarray
    risk: np.ndarray


def query_conditioned_turn_risk(
    states: np.ndarray,
    target_turn: int,
    *,
    window_radius: int = 2,
    ambient_geometry: np.ndarray | None = None,
) -> QueryConditionedGeometry:
    if states.size == 0 or target_turn <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return QueryConditionedGeometry(
            projected_curvature=empty,
            projected_subspace_energy=empty,
            query_alignment=empty,
            risk=empty,
        )

    prefix_states = np.asarray(states[:target_turn + 1], dtype=np.float32)
    unit_states, _ = normalize_rows(prefix_states)
    prefix_turn_count = target_turn
    query_state = unit_states[target_turn]

    projected_curvature = np.zeros(prefix_turn_count, dtype=np.float32)
    projected_subspace_energy = np.zeros(prefix_turn_count, dtype=np.float32)
    query_alignment = np.zeros(prefix_turn_count, dtype=np.float32)

    for idx in range(prefix_turn_count):
        current = unit_states[idx]
        query_tangent = sphere_log_map(current, query_state)
        query_norm = float(np.linalg.norm(query_tangent))
        query_alignment[idx] = query_norm
        if query_norm < EPS:
            continue
        query_dir = query_tangent / max(query_norm, EPS)

        if 0 < idx < prefix_turn_count - 1:
            prev_step = sphere_log_map(unit_states[idx - 1], current)
            prev_step_current = sphere_parallel_transport(unit_states[idx - 1], current, prev_step)
            next_step = sphere_log_map(current, unit_states[idx + 1])
            prev_norm = float(np.linalg.norm(prev_step_current))
            next_norm = float(np.linalg.norm(next_step))
            local_scale = max(0.5 * (prev_norm + next_norm), EPS)
            step_delta = next_step - prev_step_current
            projected_curvature[idx] = float(abs(np.dot(step_delta, query_dir)) / local_scale)

        start = max(0, idx - window_radius)
        end = min(prefix_turn_count - 1, idx + window_radius)
        local_steps = transported_increment_matrix(unit_states, start, end, current)
        if local_steps.size == 0:
            continue
        coeffs = np.abs(local_steps @ query_dir)
        step_norms = np.linalg.norm(local_steps, axis=1)
        scale = max(float(np.mean(step_norms)), EPS)
        projected_subspace_energy[idx] = float(np.mean(coeffs) / scale)

    query_geom = (
        0.55 * _normalize(projected_curvature)
        + 0.45 * _normalize(projected_subspace_energy)
    )
    if ambient_geometry is not None and ambient_geometry.size:
        query_geom = (
            0.70 * query_geom
            + 0.30 * _normalize(np.asarray(ambient_geometry[:prefix_turn_count], dtype=np.float32))
        )

    return QueryConditionedGeometry(
        projected_curvature=_normalize(projected_curvature),
        projected_subspace_energy=_normalize(projected_subspace_energy),
        query_alignment=_normalize(query_alignment),
        risk=_normalize(query_geom),
    )


def query_conditioned_turn_risk_v2(
    states: np.ndarray,
    target_turn: int,
    *,
    segment_span: int = 3,
    ambient_geometry: np.ndarray | None = None,
) -> QueryConditionedGeometryV2:
    if states.size == 0 or target_turn <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return QueryConditionedGeometryV2(
            projected_curvature=empty,
            projected_subspace_energy=empty,
            query_alignment=empty,
            segment_rank95=empty,
            segment_mean_step_norm=empty,
            segment_mean_stabilized_curvature=empty,
            local_projection=empty,
            risk=empty,
        )

    prefix_states = np.asarray(states[:target_turn + 1], dtype=np.float32)
    unit_states, _ = normalize_rows(prefix_states)
    prefix_turn_count = target_turn
    query_state = unit_states[target_turn]
    span = max(int(segment_span), 2)

    projected_curvature = np.zeros(prefix_turn_count, dtype=np.float32)
    projected_subspace_energy = np.zeros(prefix_turn_count, dtype=np.float32)
    query_alignment = np.zeros(prefix_turn_count, dtype=np.float32)
    segment_rank95 = np.zeros(prefix_turn_count, dtype=np.float32)
    segment_mean_step_norm = np.zeros(prefix_turn_count, dtype=np.float32)
    segment_mean_stabilized_curvature = np.zeros(prefix_turn_count, dtype=np.float32)
    local_projection = np.zeros(prefix_turn_count, dtype=np.float32)

    for start in range(0, prefix_turn_count, span):
        end = min(start + span - 1, prefix_turn_count - 1)
        if end < start:
            continue
        reference = segment_reference(unit_states, start, end)
        query_tangent = sphere_log_map(reference, query_state)
        query_norm = float(np.linalg.norm(query_tangent))
        if query_norm < EPS:
            continue
        query_dir = query_tangent / max(query_norm, EPS)

        steps = transported_increment_matrix(unit_states, start, end, reference)
        if steps.size == 0:
            indices = slice(start, end + 1)
            query_alignment[indices] = query_norm
            continue

        step_norms = np.linalg.norm(steps, axis=1)
        mean_step_norm = float(np.mean(step_norms)) if step_norms.size else 0.0

        _, _, singular_values = low_rank_project(steps, rank=max(1, min(steps.shape)))
        rank95 = float(max(effective_rank(singular_values, 0.95), 1))

        segment_states = unit_states[start : end + 1]
        segment_curvature = stabilized_curvature_series(segment_states)
        mean_stabilized_curvature = float(np.mean(segment_curvature)) if segment_curvature.size else 0.0

        if steps.shape[0] >= 2:
            step_deltas = steps[1:] - steps[:-1]
            local_scale = np.maximum(0.5 * (step_norms[1:] + step_norms[:-1]), EPS)
            curvature_value = float(np.mean(np.abs(step_deltas @ query_dir) / local_scale))
        else:
            curvature_value = 0.0
        subspace_energy = float(np.mean(np.abs(steps @ query_dir)) / max(mean_step_norm, EPS))

        local_projection_values = np.abs(steps @ query_dir) / np.maximum(step_norms, EPS)
        turn_projection_values = np.zeros(end - start + 1, dtype=np.float32)
        if local_projection_values.size:
            turn_projection_values[:-1] = local_projection_values.astype(np.float32)
            turn_projection_values[-1] = float(local_projection_values[-1])

        indices = slice(start, end + 1)
        projected_curvature[indices] = curvature_value
        projected_subspace_energy[indices] = subspace_energy
        query_alignment[indices] = query_norm
        segment_rank95[indices] = rank95
        segment_mean_step_norm[indices] = mean_step_norm
        segment_mean_stabilized_curvature[indices] = mean_stabilized_curvature
        local_projection[indices] = turn_projection_values

    risk = (
        0.45 * _normalize(projected_curvature)
        + 0.25 * _normalize(projected_subspace_energy)
        + 0.15 * _normalize(query_alignment)
        + 0.15 * _normalize(local_projection)
    )
    if ambient_geometry is not None and ambient_geometry.size:
        risk = 0.80 * risk + 0.20 * _normalize(np.asarray(ambient_geometry[:prefix_turn_count], dtype=np.float32))

    return QueryConditionedGeometryV2(
        projected_curvature=_normalize(projected_curvature),
        projected_subspace_energy=_normalize(projected_subspace_energy),
        query_alignment=_normalize(query_alignment),
        segment_rank95=_normalize(segment_rank95),
        segment_mean_step_norm=_normalize(segment_mean_step_norm),
        segment_mean_stabilized_curvature=_normalize(segment_mean_stabilized_curvature),
        local_projection=_normalize(local_projection),
        risk=_normalize(risk),
    )
