from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

import numpy as np

from .analysis import summarize_boundary_detection
from .boundary_features import lexical_shift_scores, style_shift_scores
from .conversations import ConversationRecord
from .geometry import choose_segments_from_boundary_scores


@dataclass(frozen=True, slots=True)
class BaselineResult:
    baseline_name: str
    family: str
    conversation_id: str
    num_turns: int
    num_candidate_boundaries: int
    gold_boundary_density: float
    gold_boundaries: list[int]
    predicted_boundaries: list[int]
    ordered_boundary_mae: float
    boundary_nearest_distance: float
    oversegmentation_rate: float
    miss_rate: float
    boundary_windowdiff: float
    boundary_pk: float
    boundary_auprc: float
    boundary_tp_exact: float
    boundary_fp_exact: float
    boundary_fn_exact: float
    boundary_precision_exact: float
    boundary_recall_exact: float
    boundary_f1_exact: float
    boundary_tp_tol1: float
    boundary_fp_tol1: float
    boundary_fn_tol1: float
    boundary_precision_tol1: float
    boundary_recall_tol1: float
    boundary_f1_tol1: float
    boundary_tp_tol2: float
    boundary_fp_tol2: float
    boundary_fn_tol2: float
    boundary_precision_tol2: float
    boundary_recall_tol2: float
    boundary_f1_tol2: float
    boundary_tp_tol3: float
    boundary_fp_tol3: float
    boundary_fn_tol3: float
    boundary_precision_tol3: float
    boundary_recall_tol3: float
    boundary_f1_tol3: float


def _valid_boundary_indices(conversation: ConversationRecord) -> list[int]:
    return list(range(1, max(len(conversation.turns) - 1, 1)))


def _score_boundaries(
    conversation: ConversationRecord,
    scores: np.ndarray,
    max_segment_len: int,
    min_segment_len: int,
) -> list[int]:
    segments = choose_segments_from_boundary_scores(
        n_states=len(conversation.turns),
        boundary_scores=scores.astype(np.float32),
        max_segment_len=max_segment_len,
        min_segment_len=min_segment_len,
    )
    return [start for start, _ in segments[1:]]


def fixed_window_boundaries(conversation: ConversationRecord, max_segment_len: int) -> list[int]:
    n_turns = len(conversation.turns)
    boundaries: list[int] = []
    start = 0
    while start + max_segment_len < n_turns:
        boundary = start + max_segment_len - 1
        boundaries.append(boundary)
        start = boundary
    return boundaries


def random_matched_count_boundaries(conversation: ConversationRecord, num_boundaries: int, trial: int = 0) -> list[int]:
    valid = _valid_boundary_indices(conversation)
    use_count = min(num_boundaries, len(valid))
    if use_count == 0:
        return []
    seed_input = f"{conversation.conversation_id}:{trial}"
    seed = int(hashlib.sha256(seed_input.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    return sorted(rng.sample(valid, use_count))


def _average_metric_dicts(metric_dicts: list[dict[str, object]]) -> dict[str, object]:
    if not metric_dicts:
        return summarize_boundary_detection([], [])
    mean_exact = {
        key: float(np.mean([metrics["exact"][key] for metrics in metric_dicts]))
        for key in ["tp", "fp", "fn", "precision", "recall", "f1"]
    }
    mean_tol1 = {
        key: float(np.mean([metrics["tolerance_1"][key] for metrics in metric_dicts]))
        for key in ["tp", "fp", "fn", "precision", "recall", "f1"]
    }
    mean_tol2 = {
        key: float(np.mean([metrics["tolerance_2"][key] for metrics in metric_dicts]))
        for key in ["tp", "fp", "fn", "precision", "recall", "f1"]
    }
    mean_tol3 = {
        key: float(np.mean([metrics["tolerance_3"][key] for metrics in metric_dicts]))
        for key in ["tp", "fp", "fn", "precision", "recall", "f1"]
    }
    return {
        "gold_boundaries": metric_dicts[0]["gold_boundaries"],
        "predicted_boundaries": [],
        "exact": mean_exact,
        "tolerance_1": mean_tol1,
        "tolerance_2": mean_tol2,
        "tolerance_3": mean_tol3,
        "ordered_boundary_mae": float(np.mean([metrics["ordered_boundary_mae"] for metrics in metric_dicts])),
        "mean_nearest_boundary_distance": float(np.mean([metrics["mean_nearest_boundary_distance"] for metrics in metric_dicts])),
        "oversegmentation_rate": float(np.mean([metrics["oversegmentation_rate"] for metrics in metric_dicts])),
        "miss_rate": float(np.mean([metrics["miss_rate"] for metrics in metric_dicts])),
        "windowdiff": float(np.mean([metrics["windowdiff"] for metrics in metric_dicts])),
        "pk": float(np.mean([metrics["pk"] for metrics in metric_dicts])),
        "boundary_auprc": float(np.mean([metrics["boundary_auprc"] for metrics in metric_dicts])),
    }


def evaluate_baselines(
    conversations: list[ConversationRecord],
    max_segment_len: int,
    min_segment_len: int,
) -> list[BaselineResult]:
    results: list[BaselineResult] = []
    for conversation in conversations:
        gold = sorted(conversation.boundary_indices or [])
        num_candidate_boundaries = max(len(conversation.turns) - 2, 0)
        gold_boundary_density = len(gold) / max(num_candidate_boundaries, 1)
        score_builders = {
            "lexical_shift": lexical_shift_scores(conversation),
            "style_shift": style_shift_scores(conversation),
        }
        predicted_sets: dict[str, list[int]] = {
            "fixed_window": fixed_window_boundaries(conversation, max_segment_len=max_segment_len),
        }
        for baseline_name, scores in score_builders.items():
            predicted_sets[baseline_name] = _score_boundaries(
                conversation=conversation,
                scores=scores,
                max_segment_len=max_segment_len,
                min_segment_len=min_segment_len,
            )

        for baseline_name, predicted in predicted_sets.items():
            scores = score_builders.get(baseline_name)
            metrics = summarize_boundary_detection(
                predicted,
                gold,
                num_candidate_positions=num_candidate_boundaries,
                boundary_scores=scores,
            )
            results.append(
                BaselineResult(
                    baseline_name=baseline_name,
                    family=conversation.family,
                    conversation_id=conversation.conversation_id,
                    num_turns=len(conversation.turns),
                    num_candidate_boundaries=num_candidate_boundaries,
                    gold_boundary_density=float(gold_boundary_density),
                    gold_boundaries=metrics["gold_boundaries"],
                    predicted_boundaries=metrics["predicted_boundaries"],
                    ordered_boundary_mae=metrics["ordered_boundary_mae"],
                    boundary_nearest_distance=metrics["mean_nearest_boundary_distance"],
                    oversegmentation_rate=metrics["oversegmentation_rate"],
                    miss_rate=metrics["miss_rate"],
                    boundary_windowdiff=metrics["windowdiff"],
                    boundary_pk=metrics["pk"],
                    boundary_auprc=metrics["boundary_auprc"],
                    boundary_tp_exact=metrics["exact"]["tp"],
                    boundary_fp_exact=metrics["exact"]["fp"],
                    boundary_fn_exact=metrics["exact"]["fn"],
                    boundary_precision_exact=metrics["exact"]["precision"],
                    boundary_recall_exact=metrics["exact"]["recall"],
                    boundary_f1_exact=metrics["exact"]["f1"],
                    boundary_tp_tol1=metrics["tolerance_1"]["tp"],
                    boundary_fp_tol1=metrics["tolerance_1"]["fp"],
                    boundary_fn_tol1=metrics["tolerance_1"]["fn"],
                    boundary_precision_tol1=metrics["tolerance_1"]["precision"],
                    boundary_recall_tol1=metrics["tolerance_1"]["recall"],
                    boundary_f1_tol1=metrics["tolerance_1"]["f1"],
                    boundary_tp_tol2=metrics["tolerance_2"]["tp"],
                    boundary_fp_tol2=metrics["tolerance_2"]["fp"],
                    boundary_fn_tol2=metrics["tolerance_2"]["fn"],
                    boundary_precision_tol2=metrics["tolerance_2"]["precision"],
                    boundary_recall_tol2=metrics["tolerance_2"]["recall"],
                    boundary_f1_tol2=metrics["tolerance_2"]["f1"],
                    boundary_tp_tol3=metrics["tolerance_3"]["tp"],
                    boundary_fp_tol3=metrics["tolerance_3"]["fp"],
                    boundary_fn_tol3=metrics["tolerance_3"]["fn"],
                    boundary_precision_tol3=metrics["tolerance_3"]["precision"],
                    boundary_recall_tol3=metrics["tolerance_3"]["recall"],
                    boundary_f1_tol3=metrics["tolerance_3"]["f1"],
                )
            )
        random_metrics = _average_metric_dicts(
            [
                summarize_boundary_detection(
                    random_matched_count_boundaries(conversation, num_boundaries=len(gold), trial=trial),
                    gold,
                    num_candidate_positions=num_candidate_boundaries,
                )
                for trial in range(64)
            ]
        )
        results.append(
            BaselineResult(
                baseline_name="oracle_random_matched_count",
                family=conversation.family,
                conversation_id=conversation.conversation_id,
                num_turns=len(conversation.turns),
                num_candidate_boundaries=num_candidate_boundaries,
                gold_boundary_density=float(gold_boundary_density),
                gold_boundaries=random_metrics["gold_boundaries"],
                predicted_boundaries=random_metrics["predicted_boundaries"],
                ordered_boundary_mae=random_metrics["ordered_boundary_mae"],
                boundary_nearest_distance=random_metrics["mean_nearest_boundary_distance"],
                oversegmentation_rate=random_metrics["oversegmentation_rate"],
                miss_rate=random_metrics["miss_rate"],
                boundary_windowdiff=random_metrics["windowdiff"],
                boundary_pk=random_metrics["pk"],
                boundary_auprc=random_metrics["boundary_auprc"],
                boundary_tp_exact=random_metrics["exact"]["tp"],
                boundary_fp_exact=random_metrics["exact"]["fp"],
                boundary_fn_exact=random_metrics["exact"]["fn"],
                boundary_precision_exact=random_metrics["exact"]["precision"],
                boundary_recall_exact=random_metrics["exact"]["recall"],
                boundary_f1_exact=random_metrics["exact"]["f1"],
                boundary_tp_tol1=random_metrics["tolerance_1"]["tp"],
                boundary_fp_tol1=random_metrics["tolerance_1"]["fp"],
                boundary_fn_tol1=random_metrics["tolerance_1"]["fn"],
                boundary_precision_tol1=random_metrics["tolerance_1"]["precision"],
                boundary_recall_tol1=random_metrics["tolerance_1"]["recall"],
                boundary_f1_tol1=random_metrics["tolerance_1"]["f1"],
                boundary_tp_tol2=random_metrics["tolerance_2"]["tp"],
                boundary_fp_tol2=random_metrics["tolerance_2"]["fp"],
                boundary_fn_tol2=random_metrics["tolerance_2"]["fn"],
                boundary_precision_tol2=random_metrics["tolerance_2"]["precision"],
                boundary_recall_tol2=random_metrics["tolerance_2"]["recall"],
                boundary_f1_tol2=random_metrics["tolerance_2"]["f1"],
                boundary_tp_tol3=random_metrics["tolerance_3"]["tp"],
                boundary_fp_tol3=random_metrics["tolerance_3"]["fp"],
                boundary_fn_tol3=random_metrics["tolerance_3"]["fn"],
                boundary_precision_tol3=random_metrics["tolerance_3"]["precision"],
                boundary_recall_tol3=random_metrics["tolerance_3"]["recall"],
                boundary_f1_tol3=random_metrics["tolerance_3"]["f1"],
            )
        )
    return results
