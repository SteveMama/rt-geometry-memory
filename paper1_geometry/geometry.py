from __future__ import annotations

import numpy as np


EPS = 1e-8
CURVATURE_ARCLENGTH_FLOOR = 0.05


def normalize_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe_norms = np.clip(norms, EPS, None)
    return x / safe_norms, norms[:, 0]


def sphere_distance(x: np.ndarray, y: np.ndarray) -> float:
    dot = float(np.clip(np.dot(x, y), -1.0, 1.0))
    return float(np.arccos(dot))


def sphere_log_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dot = float(np.clip(np.dot(x, y), -1.0, 1.0))
    theta = float(np.arccos(dot))
    if theta < 1e-7:
        delta = y - dot * x
        return delta - np.dot(delta, x) * x
    tangent = y - dot * x
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < EPS:
        return np.zeros_like(x)
    return tangent * (theta / tangent_norm)


def sphere_exp_map(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-7:
        return x.copy()
    out = np.cos(v_norm) * x + np.sin(v_norm) * (v / v_norm)
    out_norm = np.linalg.norm(out)
    return out / max(out_norm, EPS)


def sphere_parallel_transport(x: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
    denom = 1.0 + float(np.dot(x, y))
    if denom < 1e-6:
        # Antipodal transport is unstable; project into the target tangent space.
        projected = v - np.dot(v, y) * y
        proj_norm = np.linalg.norm(projected)
        return projected if proj_norm < EPS else projected / proj_norm * np.linalg.norm(v)
    return v - (np.dot(v, y) / denom) * (x + y)


def _unit_with_norm(v: np.ndarray) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(v))
    if norm < EPS:
        return np.zeros_like(v), 0.0
    return v / norm, norm


def turning_angle_series(unit_states: np.ndarray) -> np.ndarray:
    n_states = unit_states.shape[0]
    if n_states < 3:
        return np.zeros(0, dtype=np.float32)
    values: list[float] = []
    for idx in range(1, n_states - 1):
        prev_step = sphere_log_map(unit_states[idx - 1], unit_states[idx])
        prev_step_at_current = sphere_parallel_transport(unit_states[idx - 1], unit_states[idx], prev_step)
        next_step = sphere_log_map(unit_states[idx], unit_states[idx + 1])
        prev_dir, prev_norm = _unit_with_norm(prev_step_at_current)
        next_dir, next_norm = _unit_with_norm(next_step)
        if prev_norm < EPS or next_norm < EPS:
            values.append(0.0)
            continue
        dot = float(np.clip(np.dot(prev_dir, next_dir), -1.0, 1.0))
        values.append(float(np.arccos(dot)))
    return np.asarray(values, dtype=np.float32)


def step_norm_series(unit_states: np.ndarray) -> np.ndarray:
    n_states = unit_states.shape[0]
    if n_states < 2:
        return np.zeros(0, dtype=np.float32)
    values: list[float] = []
    for idx in range(0, n_states - 1):
        step = sphere_log_map(unit_states[idx], unit_states[idx + 1])
        values.append(float(np.linalg.norm(step)))
    return np.asarray(values, dtype=np.float32)


def curvature_series(
    unit_states: np.ndarray,
    *,
    min_arclength: float = 0.0,
) -> np.ndarray:
    n_states = unit_states.shape[0]
    if n_states < 3:
        return np.zeros(0, dtype=np.float32)
    values: list[float] = []
    for idx in range(1, n_states - 1):
        prev_step = sphere_log_map(unit_states[idx - 1], unit_states[idx])
        prev_step_at_current = sphere_parallel_transport(unit_states[idx - 1], unit_states[idx], prev_step)
        next_step = sphere_log_map(unit_states[idx], unit_states[idx + 1])
        prev_norm = float(np.linalg.norm(prev_step_at_current))
        next_norm = float(np.linalg.norm(next_step))
        if prev_norm < EPS or next_norm < EPS:
            values.append(0.0)
            continue
        prev_dir = prev_step_at_current / prev_norm
        next_dir = next_step / next_norm
        turning_angle = float(np.arccos(np.clip(np.dot(prev_dir, next_dir), -1.0, 1.0)))
        local_arclength = max(0.5 * (prev_norm + next_norm), min_arclength, EPS)
        values.append(float(turning_angle / local_arclength))
    return np.asarray(values, dtype=np.float32)


def stabilized_curvature_series(
    unit_states: np.ndarray,
    *,
    min_arclength: float = CURVATURE_ARCLENGTH_FLOOR,
) -> np.ndarray:
    return curvature_series(unit_states, min_arclength=min_arclength)


def boundary_score_series(unit_states: np.ndarray) -> np.ndarray:
    n_states = unit_states.shape[0]
    if n_states < 3:
        return np.zeros(0, dtype=np.float32)
    values: list[float] = []
    for idx in range(1, n_states - 1):
        prev_step = sphere_log_map(unit_states[idx - 1], unit_states[idx])
        prev_step_at_current = sphere_parallel_transport(unit_states[idx - 1], unit_states[idx], prev_step)
        next_step = sphere_log_map(unit_states[idx], unit_states[idx + 1])
        prev_norm = float(np.linalg.norm(prev_step_at_current))
        next_norm = float(np.linalg.norm(next_step))
        if prev_norm < EPS or next_norm < EPS:
            values.append(0.0)
            continue
        prev_dir = prev_step_at_current / prev_norm
        next_dir = next_step / next_norm
        turning_angle = float(np.arccos(np.clip(np.dot(prev_dir, next_dir), -1.0, 1.0)))
        local_scale = 0.5 * (prev_norm + next_norm)
        values.append(float(turning_angle * local_scale))
    return np.asarray(values, dtype=np.float32)


def boundary_prominence_series(boundary_scores: np.ndarray) -> np.ndarray:
    if boundary_scores.size == 0:
        return np.zeros(0, dtype=np.float32)
    prominences = np.zeros_like(boundary_scores, dtype=np.float32)
    for idx, value in enumerate(boundary_scores):
        if boundary_scores.size == 1:
            prominences[idx] = max(float(value), 0.0)
        elif idx == 0:
            prominences[idx] = max(float(value - boundary_scores[idx + 1]), 0.0)
        elif idx == boundary_scores.size - 1:
            prominences[idx] = max(float(value - boundary_scores[idx - 1]), 0.0)
        else:
            neighborhood_baseline = 0.5 * float(boundary_scores[idx - 1] + boundary_scores[idx + 1])
            prominences[idx] = max(float(value - neighborhood_baseline), 0.0)
    return prominences.astype(np.float32)


def normalize_positive_series(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    scale = float(np.quantile(values, 0.75))
    if scale < EPS:
        scale = float(values.max())
    if scale < EPS:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / scale, 0.0, None).astype(np.float32)


def segment_low_rank_residual(
    unit_states: np.ndarray,
    start: int,
    end: int,
    rank_energy: float,
) -> float:
    if end <= start:
        return 0.0
    reference = segment_reference(unit_states, start, end)
    steps = transported_increment_matrix(unit_states, start, end, reference)
    if steps.size == 0:
        return 0.0
    _, _, singular_values = low_rank_project(steps, rank=max(1, min(steps.shape)))
    rank95 = max(effective_rank(singular_values, rank_energy), 1)
    approx_steps, _, _ = low_rank_project(steps, rank=rank95)
    residual = steps - approx_steps
    return float(np.mean(np.square(residual)))


def rank_jump_series(unit_states: np.ndarray, rank_energy: float, window_radius: int = 2) -> np.ndarray:
    n_states = unit_states.shape[0]
    if n_states < 3:
        return np.zeros(0, dtype=np.float32)
    values: list[float] = []
    for boundary_idx in range(1, n_states - 1):
        start = max(0, boundary_idx - window_radius)
        end = min(n_states - 1, boundary_idx + window_radius)
        left_start = start
        left_end = boundary_idx
        right_start = boundary_idx
        right_end = end
        combined = segment_low_rank_residual(unit_states, start, end, rank_energy)
        left = segment_low_rank_residual(unit_states, left_start, left_end, rank_energy)
        right = segment_low_rank_residual(unit_states, right_start, right_end, rank_energy)
        split_gain = max(combined - 0.5 * (left + right), 0.0)
        values.append(float(split_gain))
    return np.asarray(values, dtype=np.float32)


def _local_projector(
    unit_states: np.ndarray,
    start: int,
    end: int,
    reference: np.ndarray,
    rank_energy: float,
) -> np.ndarray:
    steps = transported_increment_matrix(unit_states, start, end, reference)
    dim = unit_states.shape[1]
    if steps.size == 0:
        return np.zeros((dim, dim), dtype=np.float32)
    _, _, singular_values = low_rank_project(steps, rank=max(1, min(steps.shape)))
    rank95 = max(effective_rank(singular_values, rank_energy), 1)
    _, basis, _ = low_rank_project(steps, rank=rank95)
    if basis.size == 0:
        return np.zeros((dim, dim), dtype=np.float32)
    projector = basis @ basis.T
    return projector.astype(np.float32)


def subspace_shift_series(unit_states: np.ndarray, rank_energy: float, window_radius: int = 2) -> np.ndarray:
    n_states = unit_states.shape[0]
    if n_states < 3:
        return np.zeros(0, dtype=np.float32)
    values: list[float] = []
    for boundary_idx in range(1, n_states - 1):
        reference = unit_states[boundary_idx]
        left_start = max(0, boundary_idx - window_radius)
        left_end = boundary_idx
        right_start = boundary_idx
        right_end = min(n_states - 1, boundary_idx + window_radius)
        left_projector = _local_projector(unit_states, left_start, left_end, reference, rank_energy)
        right_projector = _local_projector(unit_states, right_start, right_end, reference, rank_energy)
        values.append(float(np.linalg.norm(left_projector - right_projector, ord="fro")))
    return np.asarray(values, dtype=np.float32)


def hybrid_boundary_score_series(
    turning_angles: np.ndarray,
    structural_shifts: np.ndarray,
    lexical_scores: np.ndarray | None = None,
    angle_weight: float = 1.0,
    structure_weight: float = 1.0,
    lexical_weight: float = 0.75,
) -> np.ndarray:
    if turning_angles.size == 0:
        return np.zeros(0, dtype=np.float32)
    angle_term = normalize_positive_series(turning_angles)
    structure_term = normalize_positive_series(structural_shifts)
    hybrid = angle_weight * angle_term + structure_weight * structure_term
    if lexical_scores is not None and lexical_scores.size:
        lexical_term = normalize_positive_series(lexical_scores[: turning_angles.size])
        hybrid = hybrid + lexical_weight * lexical_term
    return hybrid.astype(np.float32)


def changepoint_segments(
    unit_states: np.ndarray,
    turning_angles: np.ndarray,
    boundary_scores: np.ndarray,
    rank_energy: float,
    max_segment_len: int,
    min_segment_len: int,
    curvature_weight: float = 0.15,
    segment_penalty: float = 0.2,
    boundary_reward: float = 0.35,
) -> list[tuple[int, int]]:
    n_states = unit_states.shape[0]
    if n_states <= 1:
        return [(0, n_states - 1)] if n_states else []

    segment_cost_cache: dict[tuple[int, int], float] = {}

    def segment_cost(start: int, end: int) -> float:
        key = (start, end)
        if key in segment_cost_cache:
            return segment_cost_cache[key]
        length = end - start + 1
        if length <= 1:
            value = 0.0
        else:
            residual = segment_low_rank_residual(unit_states, start, end, rank_energy)
            local_turning = turning_angles[start : end - 1]
            curvature_cost = float(np.mean(np.square(local_turning))) if local_turning.size else 0.0
            value = float(length * (residual + curvature_weight * curvature_cost))
        segment_cost_cache[key] = value
        return value

    inf = float("inf")
    dp = [inf] * n_states
    prev = [-1] * n_states

    for end in range(n_states):
        for start in range(0, end + 1):
            length = end - start + 1
            if start > 0 and length < min_segment_len:
                continue
            if length > max_segment_len:
                continue
            cost = segment_cost(start, end) + segment_penalty
            if start > 0 and end < n_states - 1 and boundary_scores.size >= end:
                cost -= boundary_reward * float(boundary_scores[end - 1])
            prev_cost = 0.0 if start == 0 else dp[start]
            total = prev_cost + cost
            if total < dp[end]:
                dp[end] = total
                prev[end] = start

    if prev[-1] < 0:
        return [(0, n_states - 1)]

    segments_rev: list[tuple[int, int]] = []
    end = n_states - 1
    while end >= 0:
        start = prev[end]
        if start < 0:
            break
        segments_rev.append((start, end))
        if start == 0:
            break
        end = start
    segments = list(reversed(segments_rev))
    if not segments:
        return [(0, n_states - 1)]
    return segments


def choose_segments_from_boundary_scores(
    n_states: int,
    boundary_scores: np.ndarray,
    max_segment_len: int,
    min_segment_len: int,
) -> list[tuple[int, int]]:
    if n_states <= 1:
        return [(0, n_states - 1)] if n_states else []

    if boundary_scores.size == 0:
        return [(0, n_states - 1)]

    prominences = boundary_prominence_series(boundary_scores)
    median = float(np.median(prominences))
    mad = float(np.median(np.abs(prominences - median)))
    robust_threshold = median + 1.5 * max(mad, 0.05)
    quantile_threshold = float(np.quantile(prominences, 0.75)) if prominences.size >= 4 else float(prominences.max())
    minimum_threshold = 0.15
    threshold = max(robust_threshold, quantile_threshold, minimum_threshold)

    segments: list[tuple[int, int]] = []
    start = 0
    for idx, value in enumerate(boundary_scores, start=1):
        prominence = float(prominences[idx - 1])
        segment_len = idx - start + 1
        remaining_len = n_states - idx
        force_split = segment_len >= max_segment_len
        left_value = float(boundary_scores[idx - 2]) if idx - 2 >= 0 else float("-inf")
        right_value = float(boundary_scores[idx]) if idx < boundary_scores.size else float("-inf")
        local_peak = value >= left_value and value >= right_value
        standard_spike = prominence >= threshold and segment_len >= min_segment_len
        early_strong_spike = prominence >= 1.25 * threshold and segment_len >= 2 and remaining_len >= 2
        spike_split = local_peak and (standard_spike or early_strong_spike)
        if force_split or spike_split:
            segments.append((start, idx))
            start = idx
    if start < n_states - 1:
        segments.append((start, n_states - 1))
    elif not segments:
        segments.append((0, n_states - 1))
    return segments


def choose_segments(
    unit_states: np.ndarray,
    curvatures: np.ndarray,
    boundary_scores: np.ndarray,
    turning_angles: np.ndarray,
    rank_energy: float,
    max_segment_len: int,
    min_segment_len: int,
) -> list[tuple[int, int]]:
    if curvatures.size == 0 or boundary_scores.size == 0:
        n_states = unit_states.shape[0]
        return [(0, n_states - 1)] if n_states else []
    return changepoint_segments(
        unit_states=unit_states,
        turning_angles=turning_angles,
        boundary_scores=boundary_scores,
        rank_energy=rank_energy,
        max_segment_len=max_segment_len,
        min_segment_len=min_segment_len,
    )


def segment_reference(unit_states: np.ndarray, start: int, end: int) -> np.ndarray:
    mean_state = unit_states[start : end + 1].mean(axis=0)
    norm = np.linalg.norm(mean_state)
    if norm < EPS:
        return unit_states[start].copy()
    return mean_state / norm


def transported_increment_matrix(
    unit_states: np.ndarray,
    start: int,
    end: int,
    reference: np.ndarray,
) -> np.ndarray:
    steps: list[np.ndarray] = []
    for idx in range(start, end):
        local_step = sphere_log_map(unit_states[idx], unit_states[idx + 1])
        steps.append(sphere_parallel_transport(unit_states[idx], reference, local_step))
    if not steps:
        return np.zeros((0, unit_states.shape[1]), dtype=np.float32)
    return np.stack(steps, axis=0).astype(np.float32)


def effective_rank(singular_values: np.ndarray, energy_threshold: float) -> int:
    if singular_values.size == 0:
        return 0
    energy = np.square(singular_values)
    cumulative = np.cumsum(energy) / max(energy.sum(), EPS)
    return int(np.searchsorted(cumulative, energy_threshold, side="left") + 1)


def low_rank_project(steps: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if steps.size == 0:
        d = steps.shape[1] if steps.ndim == 2 else 0
        return steps.copy(), np.zeros((d, 0), dtype=np.float32), np.zeros(0, dtype=np.float32)
    _, singular_values, vh = np.linalg.svd(steps, full_matrices=False)
    use_rank = min(rank, vh.shape[0])
    basis = vh[:use_rank].T.astype(np.float32)
    coefficients = steps @ basis
    reconstructed = coefficients @ basis.T
    return reconstructed.astype(np.float32), basis, singular_values.astype(np.float32)


def reconstruct_segment_path(
    unit_states: np.ndarray,
    start: int,
    end: int,
    reference: np.ndarray,
    approx_steps: np.ndarray,
) -> np.ndarray:
    reconstructed = unit_states[start : end + 1].copy()
    reconstructed[0] = unit_states[start]
    for local_idx in range(approx_steps.shape[0]):
        current = reconstructed[local_idx]
        tangent = sphere_parallel_transport(reference, current, approx_steps[local_idx])
        reconstructed[local_idx + 1] = sphere_exp_map(current, tangent)
    return reconstructed
