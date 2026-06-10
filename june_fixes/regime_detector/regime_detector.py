"""Supervised memory-regime detector from oracle candidate features.

Review fix #7: the title promises *signal-conditioned* compression, but regime
selection in the paper is done by the experimenter, and the geometry-only
regime atlas (unsupervised) is a documented negative result. This module asks
the cheapest honest version of the missing question: can a small *supervised*
classifier, given per-conversation aggregates of the features the system
already computes, predict which benchmark regime a conversation comes from?

If yes, the paper gains a working (if modest) selector and the title is
earned. If no, the limitation section gets a sharper sentence. Either result
is publishable.

Implementation is numpy-only (multinomial logistic regression, full-batch
gradient descent, L2 regularization) with conversation-level k-fold cross
validation, so there is no sklearn dependency and no GPU requirement.

    python -m june_fixes.regime_detector.regime_detector \
      --labeled-csv hardset=results/paper3/harm_oracle/<hardset>/candidate_rows.csv \
      --labeled-csv msc=paper3_gate1_scaleup_multigpu_merged_results/paper3_gate1_scaleup_multigpu_oracle_msc_valid_32conv/candidate_rows.csv \
      --labeled-csv longmemeval=paper3_gate1_scaleup_multigpu_merged_results/paper3_gate1_scaleup_multigpu_oracle_longmemeval_s_cleaned_12conv/candidate_rows.csv \
      --output-dir results/june_fixes/regime_detector
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

BASE_FEATURES = [
    "semantic_score",
    "geometry_score",
    "support_score",
    "constraint_score",
    "query_geom_v2_risk",
    "query_geom_v2_curvature",
    "query_geom_v2_energy",
    "segment_rank95",
    "segment_mean_step_norm",
    "segment_mean_stabilized_curvature",
    "token_cost",
    "recency",
]


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def conversation_features(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate candidate rows of one conversation into a feature vector."""
    features: dict[str, float] = {}
    for column in BASE_FEATURES:
        values = np.asarray([_to_float(row.get(column)) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            features[f"{column}__mean"] = 0.0
            features[f"{column}__std"] = 0.0
            features[f"{column}__max"] = 0.0
            continue
        features[f"{column}__mean"] = float(np.mean(values))
        features[f"{column}__std"] = float(np.std(values))
        features[f"{column}__max"] = float(np.max(values))
    features["num_candidates"] = float(len(rows))
    user_flags = np.asarray(
        [1.0 if str(row.get("role_user")).strip() in {"1", "1.0", "true", "True"} else 0.0 for row in rows]
    )
    features["user_turn_fraction"] = float(np.mean(user_flags)) if user_flags.size else 0.0
    return features


def load_labeled(labeled_specs: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Return X, y, feature_names, class_names, conversation_ids."""
    per_conversation: dict[tuple[str, str], list[dict[str, Any]]] = {}
    labels: dict[tuple[str, str], str] = {}
    for spec in labeled_specs:
        label, _, path_text = spec.partition("=")
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"missing candidate rows for {label}: {path}")
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (label, str(row["conversation_id"]))
                per_conversation.setdefault(key, []).append(row)
                labels[key] = label

    class_names = sorted({label for label, _ in per_conversation})
    feature_names: list[str] = []
    vectors: list[list[float]] = []
    targets: list[int] = []
    conversation_ids: list[str] = []
    for key in sorted(per_conversation):
        features = conversation_features(per_conversation[key])
        if not feature_names:
            feature_names = sorted(features)
        vectors.append([features.get(name, 0.0) for name in feature_names])
        targets.append(class_names.index(labels[key]))
        conversation_ids.append(f"{key[0]}::{key[1]}")
    return (
        np.asarray(vectors, dtype=np.float64),
        np.asarray(targets, dtype=np.int64),
        feature_names,
        class_names,
        conversation_ids,
    )


def train_softmax(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    *,
    l2: float = 1e-2,
    learning_rate: float = 0.1,
    epochs: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_samples, num_features = x.shape
    weights = rng.normal(scale=0.01, size=(num_features, num_classes))
    bias = np.zeros(num_classes)
    one_hot = np.eye(num_classes)[y]
    for _ in range(epochs):
        logits = x @ weights + bias
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        gradient = x.T @ (probabilities - one_hot) / num_samples + l2 * weights
        bias_gradient = (probabilities - one_hot).mean(axis=0)
        weights -= learning_rate * gradient
        bias -= learning_rate * bias_gradient
    return weights, bias


def predict(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.argmax(x @ weights + bias, axis=1)


def k_fold_cv(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    *,
    k: int = 5,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray]:
    """Conversation-level k-fold CV; returns (predictions, fold_ids)."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(x.shape[0])
    folds = np.array_split(indices, k)
    predictions = np.full(x.shape[0], -1, dtype=np.int64)
    fold_ids = np.full(x.shape[0], -1, dtype=np.int64)
    for fold_index, test_indices in enumerate(folds):
        train_mask = np.ones(x.shape[0], dtype=bool)
        train_mask[test_indices] = False
        mean = x[train_mask].mean(axis=0)
        std = x[train_mask].std(axis=0)
        std[std < 1e-8] = 1.0
        x_train = (x[train_mask] - mean) / std
        x_test = (x[test_indices] - mean) / std
        weights, bias = train_softmax(x_train, y[train_mask], num_classes, seed=fold_index)
        predictions[test_indices] = predict(x_test, weights, bias)
        fold_ids[test_indices] = fold_index
    return predictions, fold_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labeled-csv",
        action="append",
        required=True,
        help="label=path/to/candidate_rows.csv (repeatable)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    x, y, feature_names, class_names, conversation_ids = load_labeled(args.labeled_csv)
    if len(class_names) < 2:
        raise SystemExit("need at least two labeled benchmarks to train a detector")
    predictions, fold_ids = k_fold_cv(
        x, y, len(class_names), k=min(args.folds, x.shape[0]), seed=args.seed
    )

    accuracy = float(np.mean(predictions == y))
    confusion = np.zeros((len(class_names), len(class_names)), dtype=int)
    for true_label, predicted_label in zip(y, predictions):
        confusion[true_label, predicted_label] += 1
    per_class_f1: dict[str, float] = {}
    for class_index, class_name in enumerate(class_names):
        tp = confusion[class_index, class_index]
        fp = confusion[:, class_index].sum() - tp
        fn = confusion[class_index, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        per_class_f1[class_name] = float(
            2 * precision * recall / max(precision + recall, 1e-8)
        )
    majority_baseline = float(np.max(np.bincount(y)) / y.size)

    # Final model on all data for feature inspection.
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    weights, _ = train_softmax((x - mean) / std, y, len(class_names), seed=args.seed)
    top_features = {
        class_name: [
            {"feature": feature_names[i], "weight": float(weights[i, class_index])}
            for i in np.argsort(-np.abs(weights[:, class_index]))[:8]
        ]
        for class_index, class_name in enumerate(class_names)
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_conversations": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "class_names": class_names,
        "class_counts": {
            name: int(np.sum(y == idx)) for idx, name in enumerate(class_names)
        },
        "cv_accuracy": accuracy,
        "majority_baseline": majority_baseline,
        "per_class_f1": per_class_f1,
        "macro_f1": float(np.mean(list(per_class_f1.values()))),
        "confusion_matrix": confusion.tolist(),
        "top_features_per_class": top_features,
        "predictions": [
            {
                "conversation": conversation_ids[i],
                "true": class_names[y[i]],
                "predicted": class_names[predictions[i]],
                "fold": int(fold_ids[i]),
            }
            for i in range(x.shape[0])
        ],
    }
    (args.output_dir / "regime_detector_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# Supervised Regime Detector",
        "",
        f"Conversations: {x.shape[0]} across {len(class_names)} regimes "
        f"({', '.join(class_names)}).",
        "",
        f"- {args.folds}-fold CV accuracy: **{accuracy:.3f}** "
        f"(majority baseline {majority_baseline:.3f})",
        f"- macro F1: **{payload['macro_f1']:.3f}**",
        "",
        "## Confusion matrix (rows = true, cols = predicted)",
        "",
        "| | " + " | ".join(class_names) + " |",
        "|---|" + "---|" * len(class_names),
    ]
    for class_index, class_name in enumerate(class_names):
        lines.append(
            f"| {class_name} | "
            + " | ".join(str(value) for value in confusion[class_index])
            + " |"
        )
    lines += ["", "## Strongest features per regime", ""]
    for class_name, items in top_features.items():
        lines.append(f"### {class_name}")
        for item in items:
            lines.append(f"- `{item['feature']}`: {item['weight']:+.3f}")
        lines.append("")
    lines += [
        "## Reading",
        "",
        "If CV accuracy clearly beats the majority baseline, the paper can claim a",
        "working supervised regime selector (replacing the failed unsupervised",
        "atlas) and the 'signal-conditioned' title is earned. If it does not, add",
        "this as an explicit limitation and consider retitling.",
    ]
    (args.output_dir / "regime_detector_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        f"[regime_detector] accuracy={accuracy:.3f} (majority {majority_baseline:.3f}) "
        f"-> {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
