from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .geometry import (
    boundary_prominence_series,
    boundary_score_series,
    choose_segments,
    curvature_series,
    effective_rank,
    hybrid_boundary_score_series,
    low_rank_project,
    normalize_rows,
    rank_jump_series,
    reconstruct_segment_path,
    segment_reference,
    sphere_distance,
    subspace_shift_series,
    turning_angle_series,
    transported_increment_matrix,
)


@dataclass(slots=True)
class SegmentResult:
    start_turn: int
    end_turn: int
    steps: int
    rank95: int
    mean_curvature: float
    max_curvature: float
    mean_reconstruction_geodesic: float
    max_reconstruction_geodesic: float
    singular_values: list[float]
    cumulative_energy: list[float]


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _kl_divergence(logits_p: np.ndarray, logits_q: np.ndarray) -> np.ndarray:
    p_log = logits_p - np.logaddexp.reduce(logits_p, axis=1, keepdims=True)
    q_log = logits_q - np.logaddexp.reduce(logits_q, axis=1, keepdims=True)
    p = np.exp(p_log)
    return np.maximum(np.sum(p * (p_log - q_log), axis=1), 0.0)


def _match_boundaries(predicted: list[int], gold: list[int], tolerance: int) -> dict[str, float]:
    predicted_sorted = sorted(predicted)
    gold_sorted = sorted(gold)
    used_gold: set[int] = set()
    true_positive = 0

    for pred in predicted_sorted:
        best_idx = None
        best_distance = None
        for gold_idx, gold_value in enumerate(gold_sorted):
            if gold_idx in used_gold:
                continue
            distance = abs(pred - gold_value)
            if distance > tolerance:
                continue
            if best_distance is None or distance < best_distance:
                best_idx = gold_idx
                best_distance = distance
        if best_idx is not None:
            used_gold.add(best_idx)
            true_positive += 1

    false_positive = len(predicted_sorted) - true_positive
    false_negative = len(gold_sorted) - true_positive
    precision = true_positive / len(predicted_sorted) if predicted_sorted else (1.0 if not gold_sorted else 0.0)
    recall = true_positive / len(gold_sorted) if gold_sorted else (1.0 if not predicted_sorted else 0.0)
    if precision + recall < 1e-8:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "tp": float(true_positive),
        "fp": float(false_positive),
        "fn": float(false_negative),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _ordered_boundary_mae(predicted: list[int], gold: list[int]) -> float:
    predicted_sorted = sorted(predicted)
    gold_sorted = sorted(gold)
    pairs = min(len(predicted_sorted), len(gold_sorted))
    if pairs == 0:
        return 0.0
    distances = [abs(predicted_sorted[idx] - gold_sorted[idx]) for idx in range(pairs)]
    return float(np.mean(distances))


def _mean_nearest_boundary_distance(predicted: list[int], gold: list[int]) -> float:
    predicted_sorted = sorted(predicted)
    gold_sorted = sorted(gold)
    if not predicted_sorted and not gold_sorted:
        return 0.0
    distances: list[float] = []
    for pred in predicted_sorted:
        distances.append(float(min(abs(pred - gold_value) for gold_value in gold_sorted)) if gold_sorted else 0.0)
    for gold_value in gold_sorted:
        distances.append(float(min(abs(gold_value - pred) for pred in predicted_sorted)) if predicted_sorted else 0.0)
    return float(np.mean(distances)) if distances else 0.0


def _segment_ids(boundaries: list[int], num_turns: int) -> np.ndarray:
    segment_ids = np.zeros(num_turns, dtype=np.int32)
    boundary_set = set(boundaries)
    current = 0
    for idx in range(num_turns):
        if idx in boundary_set and idx > 0:
            current += 1
        segment_ids[idx] = current
    return segment_ids


def _same_segment(segment_ids: np.ndarray, left: int, right: int) -> bool:
    return bool(segment_ids[left] == segment_ids[right])


def _pk_metric(predicted: list[int], gold: list[int], num_turns: int) -> float:
    if num_turns < 3:
        return 0.0
    gold_segments = len(gold) + 1
    mean_segment_len = num_turns / max(gold_segments, 1)
    window = max(int(round(mean_segment_len / 2.0)), 1)
    if num_turns - window <= 0:
        return 0.0
    pred_ids = _segment_ids(predicted, num_turns)
    gold_ids = _segment_ids(gold, num_turns)
    errors = [
        1.0 if _same_segment(pred_ids, idx, idx + window) != _same_segment(gold_ids, idx, idx + window) else 0.0
        for idx in range(num_turns - window)
    ]
    return float(np.mean(errors)) if errors else 0.0


def _windowdiff_metric(predicted: list[int], gold: list[int], num_turns: int) -> float:
    if num_turns < 3:
        return 0.0
    gold_segments = len(gold) + 1
    mean_segment_len = num_turns / max(gold_segments, 1)
    window = max(int(round(mean_segment_len)), 1)
    if num_turns - window <= 0:
        return 0.0
    pred_boundaries = set(predicted)
    gold_boundaries = set(gold)
    errors = []
    for start in range(0, num_turns - window):
        end = start + window
        pred_count = sum(1 for idx in range(start + 1, end + 1) if idx in pred_boundaries)
        gold_count = sum(1 for idx in range(start + 1, end + 1) if idx in gold_boundaries)
        errors.append(1.0 if pred_count != gold_count else 0.0)
    return float(np.mean(errors)) if errors else 0.0


def _boundary_auprc(boundary_scores: np.ndarray | None, gold: list[int]) -> float:
    if boundary_scores is None or boundary_scores.size == 0:
        return 0.0
    labels = np.zeros(boundary_scores.size, dtype=np.float32)
    for gold_idx in gold:
        candidate_idx = gold_idx - 1
        if 0 <= candidate_idx < labels.size:
            labels[candidate_idx] = 1.0
    positive = int(labels.sum())
    if positive == 0:
        return 1.0
    order = np.argsort(-boundary_scores)
    sorted_labels = labels[order]
    tp = 0.0
    fp = 0.0
    precisions = [1.0]
    recalls = [0.0]
    for label in sorted_labels:
        if label > 0.5:
            tp += 1.0
        else:
            fp += 1.0
        precisions.append(tp / max(tp + fp, 1.0))
        recalls.append(tp / positive)
    area = 0.0
    for idx in range(1, len(precisions)):
        area += (recalls[idx] - recalls[idx - 1]) * precisions[idx]
    return float(area)


def summarize_boundary_detection(
    predicted: list[int],
    gold: list[int],
    num_candidate_positions: int | None = None,
    boundary_scores: np.ndarray | None = None,
) -> dict[str, object]:
    gold_sorted = sorted(gold)
    predicted_sorted = sorted(predicted)
    exact = _match_boundaries(predicted_sorted, gold_sorted, tolerance=0)
    tol1 = _match_boundaries(predicted_sorted, gold_sorted, tolerance=1)
    tol2 = _match_boundaries(predicted_sorted, gold_sorted, tolerance=2)
    tol3 = _match_boundaries(predicted_sorted, gold_sorted, tolerance=3)
    gold_count = len(gold_sorted)
    overseg_rate = exact["fp"] / max(gold_count, 1)
    miss_rate = exact["fn"] / max(gold_count, 1) if gold_count else 0.0
    num_turns = (num_candidate_positions + 2) if num_candidate_positions is not None else (max(gold_sorted + predicted_sorted, default=0) + 2)
    return {
        "gold_boundaries": gold_sorted,
        "predicted_boundaries": predicted_sorted,
        "exact": exact,
        "tolerance_1": tol1,
        "tolerance_2": tol2,
        "tolerance_3": tol3,
        "ordered_boundary_mae": _ordered_boundary_mae(predicted_sorted, gold_sorted),
        "mean_nearest_boundary_distance": _mean_nearest_boundary_distance(predicted_sorted, gold_sorted),
        "oversegmentation_rate": float(overseg_rate),
        "miss_rate": float(miss_rate),
        "pk": _pk_metric(predicted_sorted, gold_sorted, num_turns),
        "windowdiff": _windowdiff_metric(predicted_sorted, gold_sorted, num_turns),
        "boundary_auprc": _boundary_auprc(boundary_scores, gold_sorted),
    }


def aggregate_boundary_summary_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {
            "macro_boundary_f1_exact": 0.0,
            "macro_boundary_f1_tol1": 0.0,
            "micro_boundary_precision_exact": 0.0,
            "micro_boundary_recall_exact": 0.0,
            "micro_boundary_f1_exact": 0.0,
            "micro_boundary_precision_tol1": 0.0,
            "micro_boundary_recall_tol1": 0.0,
            "micro_boundary_f1_tol1": 0.0,
            "macro_boundary_f1_tol2": 0.0,
            "macro_boundary_f1_tol3": 0.0,
            "micro_boundary_precision_tol2": 0.0,
            "micro_boundary_recall_tol2": 0.0,
            "micro_boundary_f1_tol2": 0.0,
            "micro_boundary_precision_tol3": 0.0,
            "micro_boundary_recall_tol3": 0.0,
            "micro_boundary_f1_tol3": 0.0,
            "mean_boundary_nearest_distance": 0.0,
            "mean_boundary_windowdiff": 0.0,
            "mean_boundary_pk": 0.0,
            "mean_boundary_auprc": 0.0,
            "mean_num_candidate_boundaries": 0.0,
            "mean_num_gold_boundaries": 0.0,
            "mean_gold_boundary_density": 0.0,
            "zero_gold_boundary_fraction": 0.0,
        }

    exact_tp = sum(float(row.get("boundary_tp_exact", 0.0)) for row in rows)
    exact_fp = sum(float(row.get("boundary_fp_exact", 0.0)) for row in rows)
    exact_fn = sum(float(row.get("boundary_fn_exact", 0.0)) for row in rows)
    tol1_tp = sum(float(row.get("boundary_tp_tol1", 0.0)) for row in rows)
    tol1_fp = sum(float(row.get("boundary_fp_tol1", 0.0)) for row in rows)
    tol1_fn = sum(float(row.get("boundary_fn_tol1", 0.0)) for row in rows)
    tol2_tp = sum(float(row.get("boundary_tp_tol2", 0.0)) for row in rows)
    tol2_fp = sum(float(row.get("boundary_fp_tol2", 0.0)) for row in rows)
    tol2_fn = sum(float(row.get("boundary_fn_tol2", 0.0)) for row in rows)
    tol3_tp = sum(float(row.get("boundary_tp_tol3", 0.0)) for row in rows)
    tol3_fp = sum(float(row.get("boundary_fp_tol3", 0.0)) for row in rows)
    tol3_fn = sum(float(row.get("boundary_fn_tol3", 0.0)) for row in rows)

    def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
        precision = tp / (tp + fp) if tp + fp > 0.0 else (1.0 if fn == 0.0 else 0.0)
        recall = tp / (tp + fn) if tp + fn > 0.0 else (1.0 if fp == 0.0 else 0.0)
        if precision + recall < 1e-8:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        return float(precision), float(recall), float(f1)

    exact_precision, exact_recall, exact_f1 = _prf(exact_tp, exact_fp, exact_fn)
    tol1_precision, tol1_recall, tol1_f1 = _prf(tol1_tp, tol1_fp, tol1_fn)
    tol2_precision, tol2_recall, tol2_f1 = _prf(tol2_tp, tol2_fp, tol2_fn)
    tol3_precision, tol3_recall, tol3_f1 = _prf(tol3_tp, tol3_fp, tol3_fn)
    return {
        "macro_boundary_f1_exact": float(np.mean([float(row.get("boundary_f1_exact", 0.0)) for row in rows])),
        "macro_boundary_f1_tol1": float(np.mean([float(row.get("boundary_f1_tol1", 0.0)) for row in rows])),
        "macro_boundary_f1_tol2": float(np.mean([float(row.get("boundary_f1_tol2", 0.0)) for row in rows])),
        "macro_boundary_f1_tol3": float(np.mean([float(row.get("boundary_f1_tol3", 0.0)) for row in rows])),
        "micro_boundary_precision_exact": exact_precision,
        "micro_boundary_recall_exact": exact_recall,
        "micro_boundary_f1_exact": exact_f1,
        "micro_boundary_precision_tol1": tol1_precision,
        "micro_boundary_recall_tol1": tol1_recall,
        "micro_boundary_f1_tol1": tol1_f1,
        "micro_boundary_precision_tol2": tol2_precision,
        "micro_boundary_recall_tol2": tol2_recall,
        "micro_boundary_f1_tol2": tol2_f1,
        "micro_boundary_precision_tol3": tol3_precision,
        "micro_boundary_recall_tol3": tol3_recall,
        "micro_boundary_f1_tol3": tol3_f1,
        "mean_boundary_nearest_distance": float(
            np.mean([float(row.get("boundary_nearest_distance", 0.0)) for row in rows])
        ),
        "mean_boundary_windowdiff": float(np.mean([float(row.get("boundary_windowdiff", 0.0)) for row in rows])),
        "mean_boundary_pk": float(np.mean([float(row.get("boundary_pk", 0.0)) for row in rows])),
        "mean_boundary_auprc": float(np.mean([float(row.get("boundary_auprc", 0.0)) for row in rows])),
        "mean_num_candidate_boundaries": float(np.mean([float(row.get("num_candidate_boundaries", 0.0)) for row in rows])),
        "mean_num_gold_boundaries": float(np.mean([float(row.get("num_gold_boundaries", 0.0)) for row in rows])),
        "mean_gold_boundary_density": float(np.mean([float(row.get("gold_boundary_density", 0.0)) for row in rows])),
        "zero_gold_boundary_fraction": float(
            np.mean([1.0 if float(row.get("num_gold_boundaries", 0.0)) == 0.0 else 0.0 for row in rows])
        ),
    }


def analyze_trajectory(
    states: np.ndarray,
    logits: np.ndarray,
    reconstructed_logits: np.ndarray,
    gold_boundaries: list[int] | None = None,
    lexical_boundary_scores: np.ndarray | None = None,
    rank_energy: float = 0.95,
    max_segment_len: int = 6,
    min_segment_len: int = 3,
) -> dict:
    unit_states, state_norms = normalize_rows(states)
    curvatures = curvature_series(unit_states)
    turning_angles = turning_angle_series(unit_states)
    geometry_boundary_scores = boundary_score_series(unit_states)
    rank_jumps = rank_jump_series(unit_states, rank_energy=rank_energy)
    subspace_shifts = subspace_shift_series(unit_states, rank_energy=rank_energy)
    lexical_only_scores = (
        np.asarray(lexical_boundary_scores[: turning_angles.size], dtype=np.float32)
        if lexical_boundary_scores is not None and lexical_boundary_scores.size
        else np.zeros_like(turning_angles, dtype=np.float32)
    )
    geometry_only_scores = hybrid_boundary_score_series(
        turning_angles=turning_angles,
        structural_shifts=subspace_shifts,
        lexical_scores=None,
    )
    hybrid_boundary_scores = hybrid_boundary_score_series(
        turning_angles=turning_angles,
        structural_shifts=subspace_shifts,
        lexical_scores=lexical_boundary_scores,
    )
    boundary_prominences = boundary_prominence_series(hybrid_boundary_scores)
    consecutive_angles = np.asarray(
        [sphere_distance(unit_states[idx], unit_states[idx + 1]) for idx in range(unit_states.shape[0] - 1)],
        dtype=np.float32,
    )
    segments = choose_segments(
        unit_states=unit_states,
        curvatures=curvatures,
        boundary_scores=hybrid_boundary_scores,
        turning_angles=turning_angles,
        rank_energy=rank_energy,
        max_segment_len=max_segment_len,
        min_segment_len=min_segment_len,
    )

    reconstructed_states = unit_states.copy()
    segment_rows: list[SegmentResult] = []

    for start, end in segments:
        reference = segment_reference(unit_states, start, end)
        steps = transported_increment_matrix(unit_states, start, end, reference)
        _, _, singular_values = low_rank_project(steps, rank=max(1, min(steps.shape)))
        rank95 = effective_rank(singular_values, rank_energy)
        approx_steps, _, _ = low_rank_project(steps, rank=rank95)
        reconstructed_segment = reconstruct_segment_path(unit_states, start, end, reference, approx_steps)
        reconstructed_states[start : end + 1] = reconstructed_segment
        singular_energy = np.square(singular_values)
        cumulative_energy = (
            np.cumsum(singular_energy) / max(float(singular_energy.sum()), 1e-8)
            if singular_energy.size
            else np.zeros(0, dtype=np.float32)
        )

        geodesic_errors = np.asarray(
            [sphere_distance(unit_states[idx], reconstructed_states[idx]) for idx in range(start, end + 1)],
            dtype=np.float32,
        )
        local_curvatures = curvatures[max(start - 1, 0) : max(end - 1, 0)]
        segment_rows.append(
            SegmentResult(
                start_turn=start,
                end_turn=end,
                steps=max(end - start, 0),
                rank95=rank95,
                mean_curvature=float(local_curvatures.mean()) if local_curvatures.size else 0.0,
                max_curvature=float(local_curvatures.max()) if local_curvatures.size else 0.0,
                mean_reconstruction_geodesic=float(geodesic_errors.mean()) if geodesic_errors.size else 0.0,
                max_reconstruction_geodesic=float(geodesic_errors.max()) if geodesic_errors.size else 0.0,
                singular_values=singular_values.astype(np.float32).tolist(),
                cumulative_energy=cumulative_energy.astype(np.float32).tolist(),
            )
        )

    predicted_boundaries = [row.start_turn for row in segment_rows[1:]]
    gold_boundary_list = sorted(gold_boundaries or [])
    num_candidate_boundaries = max(unit_states.shape[0] - 2, 0)
    boundary_eval = summarize_boundary_detection(
        predicted_boundaries,
        gold_boundary_list,
        num_candidate_positions=num_candidate_boundaries,
        boundary_scores=hybrid_boundary_scores,
    )
    gold_boundary_density = len(gold_boundary_list) / max(num_candidate_boundaries, 1)

    boundary_variants: dict[str, dict[str, object]] = {}
    for variant_name, scores in {
        "geometry_only": geometry_only_scores,
        "lexical_only": lexical_only_scores,
        "geometry_lexical": hybrid_boundary_scores,
    }.items():
        variant_segments = choose_segments(
            unit_states=unit_states,
            curvatures=curvatures,
            boundary_scores=scores,
            turning_angles=turning_angles,
            rank_energy=rank_energy,
            max_segment_len=max_segment_len,
            min_segment_len=min_segment_len,
        )
        variant_predicted = [segment_start for segment_start, _ in variant_segments[1:]]
        variant_eval = summarize_boundary_detection(
            variant_predicted,
            gold_boundary_list,
            num_candidate_positions=num_candidate_boundaries,
            boundary_scores=scores,
        )
        boundary_variants[variant_name] = {
            "predicted_boundaries": variant_predicted,
            "metrics": variant_eval,
        }

    scaled_reconstruction = reconstructed_states * state_norms[:, None]
    state_geodesic_errors = np.asarray(
        [sphere_distance(unit_states[idx], reconstructed_states[idx]) for idx in range(unit_states.shape[0])],
        dtype=np.float32,
    )
    logit_l2 = np.linalg.norm(logits - reconstructed_logits, axis=1).astype(np.float32)
    kl = _kl_divergence(logits, reconstructed_logits).astype(np.float32)
    top1_match = (np.argmax(logits, axis=1) == np.argmax(reconstructed_logits, axis=1)).astype(np.float32)

    return {
        "summary": {
            "num_turns": int(states.shape[0]),
            "state_dim": int(states.shape[1]),
            "mean_consecutive_angle": float(consecutive_angles.mean()) if consecutive_angles.size else 0.0,
            "max_consecutive_angle": float(consecutive_angles.max()) if consecutive_angles.size else 0.0,
            "mean_curvature": float(curvatures.mean()) if curvatures.size else 0.0,
            "max_curvature": float(curvatures.max()) if curvatures.size else 0.0,
            "mean_turning_angle": float(turning_angles.mean()) if turning_angles.size else 0.0,
            "max_turning_angle": float(turning_angles.max()) if turning_angles.size else 0.0,
            "segmentation_method": "changepoint_hybrid",
            "mean_geometry_boundary_score": float(geometry_boundary_scores.mean()) if geometry_boundary_scores.size else 0.0,
            "max_geometry_boundary_score": float(geometry_boundary_scores.max()) if geometry_boundary_scores.size else 0.0,
            "mean_rank_jump": float(rank_jumps.mean()) if rank_jumps.size else 0.0,
            "max_rank_jump": float(rank_jumps.max()) if rank_jumps.size else 0.0,
            "mean_subspace_shift": float(subspace_shifts.mean()) if subspace_shifts.size else 0.0,
            "max_subspace_shift": float(subspace_shifts.max()) if subspace_shifts.size else 0.0,
            "mean_boundary_score": float(hybrid_boundary_scores.mean()) if hybrid_boundary_scores.size else 0.0,
            "max_boundary_score": float(hybrid_boundary_scores.max()) if hybrid_boundary_scores.size else 0.0,
            "mean_boundary_prominence": float(boundary_prominences.mean()) if boundary_prominences.size else 0.0,
            "max_boundary_prominence": float(boundary_prominences.max()) if boundary_prominences.size else 0.0,
            "num_segments": len(segments),
            "num_candidate_boundaries": num_candidate_boundaries,
            "num_gold_boundaries": len(gold_boundary_list),
            "num_predicted_boundaries": len(predicted_boundaries),
            "gold_boundary_density": float(gold_boundary_density),
            "boundary_tp_exact": boundary_eval["exact"]["tp"],
            "boundary_fp_exact": boundary_eval["exact"]["fp"],
            "boundary_fn_exact": boundary_eval["exact"]["fn"],
            "boundary_precision_exact": boundary_eval["exact"]["precision"],
            "boundary_recall_exact": boundary_eval["exact"]["recall"],
            "boundary_f1_exact": boundary_eval["exact"]["f1"],
            "boundary_tp_tol1": boundary_eval["tolerance_1"]["tp"],
            "boundary_fp_tol1": boundary_eval["tolerance_1"]["fp"],
            "boundary_fn_tol1": boundary_eval["tolerance_1"]["fn"],
            "boundary_precision_tol1": boundary_eval["tolerance_1"]["precision"],
            "boundary_recall_tol1": boundary_eval["tolerance_1"]["recall"],
            "boundary_f1_tol1": boundary_eval["tolerance_1"]["f1"],
            "boundary_tp_tol2": boundary_eval["tolerance_2"]["tp"],
            "boundary_fp_tol2": boundary_eval["tolerance_2"]["fp"],
            "boundary_fn_tol2": boundary_eval["tolerance_2"]["fn"],
            "boundary_precision_tol2": boundary_eval["tolerance_2"]["precision"],
            "boundary_recall_tol2": boundary_eval["tolerance_2"]["recall"],
            "boundary_f1_tol2": boundary_eval["tolerance_2"]["f1"],
            "boundary_tp_tol3": boundary_eval["tolerance_3"]["tp"],
            "boundary_fp_tol3": boundary_eval["tolerance_3"]["fp"],
            "boundary_fn_tol3": boundary_eval["tolerance_3"]["fn"],
            "boundary_precision_tol3": boundary_eval["tolerance_3"]["precision"],
            "boundary_recall_tol3": boundary_eval["tolerance_3"]["recall"],
            "boundary_f1_tol3": boundary_eval["tolerance_3"]["f1"],
            "boundary_ordered_mae": boundary_eval["ordered_boundary_mae"],
            "boundary_nearest_distance": boundary_eval["mean_nearest_boundary_distance"],
            "boundary_oversegmentation_rate": boundary_eval["oversegmentation_rate"],
            "boundary_miss_rate": boundary_eval["miss_rate"],
            "boundary_windowdiff": boundary_eval["windowdiff"],
            "boundary_pk": boundary_eval["pk"],
            "boundary_auprc": boundary_eval["boundary_auprc"],
            "geometry_only_boundary_f1_exact": boundary_variants["geometry_only"]["metrics"]["exact"]["f1"],
            "geometry_only_boundary_f1_tol2": boundary_variants["geometry_only"]["metrics"]["tolerance_2"]["f1"],
            "geometry_only_boundary_auprc": boundary_variants["geometry_only"]["metrics"]["boundary_auprc"],
            "lexical_only_boundary_f1_exact": boundary_variants["lexical_only"]["metrics"]["exact"]["f1"],
            "lexical_only_boundary_f1_tol2": boundary_variants["lexical_only"]["metrics"]["tolerance_2"]["f1"],
            "lexical_only_boundary_auprc": boundary_variants["lexical_only"]["metrics"]["boundary_auprc"],
            "geometry_lexical_boundary_f1_exact": boundary_variants["geometry_lexical"]["metrics"]["exact"]["f1"],
            "geometry_lexical_boundary_f1_tol2": boundary_variants["geometry_lexical"]["metrics"]["tolerance_2"]["f1"],
            "geometry_lexical_boundary_auprc": boundary_variants["geometry_lexical"]["metrics"]["boundary_auprc"],
            "mean_rank95": float(np.mean([row.rank95 for row in segment_rows])) if segment_rows else 0.0,
            "max_rank95": int(max((row.rank95 for row in segment_rows), default=0)),
            "mean_state_geodesic_error": float(state_geodesic_errors.mean()) if state_geodesic_errors.size else 0.0,
            "mean_logit_l2": float(logit_l2.mean()) if logit_l2.size else 0.0,
            "mean_kl": float(kl.mean()) if kl.size else 0.0,
            "top1_agreement": float(top1_match.mean()) if top1_match.size else 0.0,
            "corr_geodesic_vs_logit_l2": _safe_correlation(state_geodesic_errors, logit_l2),
            "corr_geodesic_vs_kl": _safe_correlation(state_geodesic_errors, kl),
        },
        "segments": [asdict(row) for row in segment_rows],
        "boundary_eval": boundary_eval,
        "boundary_variants": boundary_variants,
        "series": {
            "state_norms": state_norms.tolist(),
            "consecutive_angles": consecutive_angles.tolist(),
            "curvatures": curvatures.tolist(),
            "turning_angles": turning_angles.tolist(),
            "geometry_boundary_scores": geometry_boundary_scores.tolist(),
            "rank_jumps": rank_jumps.tolist(),
            "subspace_shifts": subspace_shifts.tolist(),
            "lexical_boundary_scores": lexical_boundary_scores.tolist() if lexical_boundary_scores is not None else [],
            "boundary_scores": hybrid_boundary_scores.tolist(),
            "boundary_prominences": boundary_prominences.tolist(),
            "state_geodesic_errors": state_geodesic_errors.tolist(),
            "logit_l2": logit_l2.tolist(),
            "kl": kl.tolist(),
            "top1_match": top1_match.tolist(),
        },
        "artifacts": {
            "reconstructed_states": scaled_reconstruction.astype(np.float32),
            "reconstructed_unit_states": reconstructed_states.astype(np.float32),
        },
    }


def save_analysis_json(analysis: dict, output_path: str | Path) -> None:
    serializable = dict(analysis)
    serializable["artifacts"] = {
        "reconstructed_states_shape": list(analysis["artifacts"]["reconstructed_states"].shape),
        "reconstructed_unit_states_shape": list(analysis["artifacts"]["reconstructed_unit_states"].shape),
    }
    Path(output_path).write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def save_analysis_npz(
    analysis: dict,
    original_states: np.ndarray,
    original_logits: np.ndarray,
    output_path: str | Path,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        original_states=original_states.astype(np.float32),
        original_logits=original_logits.astype(np.float32),
        reconstructed_states=analysis["artifacts"]["reconstructed_states"].astype(np.float32),
        reconstructed_unit_states=analysis["artifacts"]["reconstructed_unit_states"].astype(np.float32),
        consecutive_angles=np.asarray(analysis["series"]["consecutive_angles"], dtype=np.float32),
        curvatures=np.asarray(analysis["series"]["curvatures"], dtype=np.float32),
        turning_angles=np.asarray(analysis["series"]["turning_angles"], dtype=np.float32),
        geometry_boundary_scores=np.asarray(analysis["series"]["geometry_boundary_scores"], dtype=np.float32),
        rank_jumps=np.asarray(analysis["series"]["rank_jumps"], dtype=np.float32),
        subspace_shifts=np.asarray(analysis["series"]["subspace_shifts"], dtype=np.float32),
        lexical_boundary_scores=np.asarray(analysis["series"]["lexical_boundary_scores"], dtype=np.float32),
        boundary_scores=np.asarray(analysis["series"]["boundary_scores"], dtype=np.float32),
        boundary_prominences=np.asarray(analysis["series"]["boundary_prominences"], dtype=np.float32),
        state_geodesic_errors=np.asarray(analysis["series"]["state_geodesic_errors"], dtype=np.float32),
        logit_l2=np.asarray(analysis["series"]["logit_l2"], dtype=np.float32),
        kl=np.asarray(analysis["series"]["kl"], dtype=np.float32),
        top1_match=np.asarray(analysis["series"]["top1_match"], dtype=np.float32),
    )
