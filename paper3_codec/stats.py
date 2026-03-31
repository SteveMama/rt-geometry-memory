from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def bootstrap_mean_ci(
    values: list[float],
    *,
    rng: np.random.Generator,
    num_bootstrap: int = 2000,
) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        value = float(array[0])
        return {"mean": value, "std": 0.0, "ci_low": value, "ci_high": value}
    samples = rng.choice(array, size=(num_bootstrap, array.size), replace=True)
    means = samples.mean(axis=1)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def paired_signflip_test(
    deltas: np.ndarray,
    *,
    rng: np.random.Generator,
    num_samples: int = 4000,
) -> float:
    if deltas.size == 0:
        return 1.0
    observed = float(abs(np.mean(deltas)))
    if observed < 1e-12:
        return 1.0
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=(num_samples, deltas.size), replace=True)
    null_means = np.abs((signs * deltas[None, :]).mean(axis=1))
    return float(np.mean(null_means >= observed))


def collapse_rows_by_keys(
    rows: list[dict[str, Any]],
    *,
    metric_keys: Iterable[str],
    group_keys: Iterable[str],
) -> list[dict[str, Any]]:
    group_key_list = list(group_keys)
    metric_key_list = list(metric_keys)
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_key_list)
        bucket = grouped.setdefault(key, {metric_key: [] for metric_key in metric_key_list})
        for metric_key in metric_key_list:
            value = row.get(metric_key)
            if value is None or value == "":
                continue
            bucket[metric_key].append(float(value))

    collapsed: list[dict[str, Any]] = []
    for key, metrics in grouped.items():
        item = {group_key: key[idx] for idx, group_key in enumerate(group_key_list)}
        for metric_key, values in metrics.items():
            item[metric_key] = float(np.mean(values)) if values else 0.0
        collapsed.append(item)
    return collapsed


def zscore(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def average_rank(values: np.ndarray, *, descending: bool = False) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    array = -values if descending else values
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and float(array[order[end]]) == float(array[order[index]]):
            end += 1
        rank = 0.5 * (index + end - 1) + 1.0
        ranks[order[index:end]] = rank
        index = end
    return ranks


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if values_a.size < 2 or values_b.size < 2:
        return 0.0
    rank_a = average_rank(values_a)
    rank_b = average_rank(values_b)
    rank_a = rank_a - float(np.mean(rank_a))
    rank_b = rank_b - float(np.mean(rank_b))
    denom = float(np.linalg.norm(rank_a) * np.linalg.norm(rank_b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(rank_a, rank_b) / denom)


def kendall_tau(values_a: np.ndarray, values_b: np.ndarray) -> float:
    n = int(values_a.size)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0
    for left in range(n - 1):
        for right in range(left + 1, n):
            diff_a = float(values_a[left] - values_a[right])
            diff_b = float(values_b[left] - values_b[right])
            if abs(diff_a) < 1e-12 and abs(diff_b) < 1e-12:
                continue
            if abs(diff_a) < 1e-12:
                ties_a += 1
                continue
            if abs(diff_b) < 1e-12:
                ties_b += 1
                continue
            if diff_a * diff_b > 0:
                concordant += 1
            elif diff_a * diff_b < 0:
                discordant += 1
    denom = float(np.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b)))
    if denom < 1e-8:
        return 0.0
    return float((concordant - discordant) / denom)


def topk_recall(pred_scores: np.ndarray, oracle_scores: np.ndarray, *, k: int = 5) -> float:
    if pred_scores.size == 0 or oracle_scores.size == 0:
        return 0.0
    use_k = max(1, min(int(k), int(pred_scores.size), int(oracle_scores.size)))
    pred_top = set(np.argsort(-pred_scores, kind="stable")[:use_k].tolist())
    oracle_top = set(np.argsort(-oracle_scores, kind="stable")[:use_k].tolist())
    if not oracle_top:
        return 0.0
    return float(len(pred_top & oracle_top) / len(oracle_top))


def ndcg_score(pred_scores: np.ndarray, oracle_scores: np.ndarray, *, k: int = 5) -> float:
    if pred_scores.size == 0 or oracle_scores.size == 0:
        return 0.0
    use_k = max(1, min(int(k), int(pred_scores.size), int(oracle_scores.size)))
    pred_order = np.argsort(-pred_scores, kind="stable")[:use_k]
    ideal_order = np.argsort(-oracle_scores, kind="stable")[:use_k]
    gains = np.maximum(oracle_scores, 0.0)
    discounts = 1.0 / np.log2(np.arange(2, use_k + 2, dtype=np.float64))
    dcg = float(np.sum(gains[pred_order] * discounts))
    idcg = float(np.sum(gains[ideal_order] * discounts))
    if idcg < 1e-8:
        return 0.0
    return float(dcg / idcg)


def metric_summary_by_unit(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    group_keys: Iterable[str],
    unit_keys: Iterable[str] | None = None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    if unit_keys is None:
        for row in rows:
            key = tuple(row[group_key] for group_key in group_keys)
            grouped[key].append(float(row[value_key]))
    else:
        collapsed = collapse_rows_by_keys(rows, metric_keys=[value_key], group_keys=[*group_keys, *unit_keys])
        for row in collapsed:
            key = tuple(row[group_key] for group_key in group_keys)
            grouped[key].append(float(row[value_key]))
    summary: dict[str, Any] = {}
    group_key_list = list(group_keys)
    for key, values in grouped.items():
        cursor = summary
        for idx, group_key in enumerate(group_key_list[:-1]):
            cursor = cursor.setdefault(str(key[idx]), {})
        cursor[str(key[-1])] = bootstrap_mean_ci(values, rng=rng)
    return summary
