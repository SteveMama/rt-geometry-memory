from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _group_mean(rows: list[dict[str, str]], metric: str, *, model_key: str, policy_name: str, budget_fraction: str) -> float:
    values = [
        float(row[metric])
        for row in rows
        if row["model_key"] == model_key
        and row["policy_name"] == policy_name
        and row["budget_fraction"] == budget_fraction
    ]
    return float(np.mean(values)) if values else 0.0


def plot_budget_curves(rows: list[dict[str, str]], output_path: Path, metric: str, ylabel: str, title: str) -> None:
    model_keys = sorted({row["model_key"] for row in rows})
    budget_keys = sorted({row["budget_fraction"] for row in rows}, key=float)
    preferred_order = [
        "uniform",
        "uniform_segment_actions",
        "lexical",
        "geometry",
        "geometry_lexical",
        "geometry_segment_actions",
    ]
    present = {row["policy_name"] for row in rows}
    policy_names = [name for name in preferred_order if name in present]
    if not model_keys or not budget_keys:
        return

    fig, axes = plt.subplots(1, len(model_keys), figsize=(5.2 * len(model_keys), 4.4), squeeze=False)
    color_map = {
        "uniform": "#5f5f5f",
        "uniform_segment_actions": "#8a8a8a",
        "lexical": "#1f77b4",
        "geometry": "#d95f02",
        "geometry_lexical": "#2ca02c",
        "geometry_segment_actions": "#7b3294",
    }
    for axis, model_key in zip(axes[0], model_keys):
        x = np.asarray([float(key) for key in budget_keys], dtype=np.float32)
        for policy_name in policy_names:
            y = np.asarray(
                [_group_mean(rows, metric, model_key=model_key, policy_name=policy_name, budget_fraction=budget_key) for budget_key in budget_keys],
                dtype=np.float32,
            )
            axis.plot(x, y, marker="o", linewidth=2.0, label=policy_name, color=color_map[policy_name])
        axis.set_title(model_key)
        axis.set_xlabel("Budget fraction")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, -1].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_family_heatmap(rows: list[dict[str, str]], output_path: Path, metric: str, budget_fraction: str, title: str) -> None:
    families = sorted({row["family"] for row in rows})
    model_policy_keys = sorted({(row["model_key"], row["policy_name"]) for row in rows})
    if not families or not model_policy_keys:
        return

    matrix = np.zeros((len(model_policy_keys), len(families)), dtype=np.float32)
    for row_idx, (model_key, policy_name) in enumerate(model_policy_keys):
        for col_idx, family in enumerate(families):
            values = [
                float(row[metric])
                for row in rows
                if row["model_key"] == model_key
                and row["policy_name"] == policy_name
                and row["family"] == family
                and row["budget_fraction"] == budget_fraction
            ]
            matrix[row_idx, col_idx] = float(np.mean(values)) if values else 0.0

    fig, axis = plt.subplots(figsize=(10.5, max(4.6, 0.6 * len(model_policy_keys))))
    image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd")
    axis.set_xticks(np.arange(len(families)))
    axis.set_xticklabels(families, rotation=20, ha="right")
    axis.set_yticks(np.arange(len(model_policy_keys)))
    axis.set_yticklabels([f"{model_key}:{policy_name}" for model_key, policy_name in model_policy_keys])
    axis.set_title(f"{title} @ budget {budget_fraction}")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            axis.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_study_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
