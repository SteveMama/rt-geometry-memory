from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paper1_geometry.geometry import (
    EPS,
    normalize_rows,
    sphere_log_map,
    sphere_parallel_transport,
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
