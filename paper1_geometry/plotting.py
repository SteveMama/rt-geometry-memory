from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_single_heatmap(
    matrix: np.ndarray,
    *,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    output_path: Path,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 5.5))
    image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(np.arange(len(xlabels)))
    axis.set_xticklabels(xlabels, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(ylabels)))
    axis.set_yticklabels(ylabels)
    axis.set_title(title)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            axis.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_single_scatter(
    x: list[float],
    y: list[float],
    families: list[str],
    family_to_points: dict[str, tuple[list[float], list[float]]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(6.6, 4.8))
    color_map = plt.get_cmap("tab10")
    for family_idx, family in enumerate(families):
        points = family_to_points.get(family)
        if not points:
            continue
        axis.scatter(points[0], points[1], s=18, alpha=0.7, label=family, color=color_map(family_idx % 10))
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_family_curvature_traces(conversation_records: list[dict], output_path: Path) -> None:
    families = sorted({record["family"] for record in conversation_records})
    if not families:
        return

    fig, axes = plt.subplots(len(families), 1, figsize=(10, max(3.5, 2.8 * len(families))), squeeze=False)
    for axis, family in zip(axes[:, 0], families):
        family_records = [record for record in conversation_records if record["family"] == family]
        for record in family_records:
            payload = _load_json(record["output_json"])
            curvatures = np.asarray(payload["series"]["curvatures"], dtype=np.float32)
            if curvatures.size == 0:
                continue
            axis.plot(np.arange(1, curvatures.size + 1), curvatures, marker="o", linewidth=1.5, alpha=0.8)
        axis.set_title(f"Curvature Traces: {family}")
        axis.set_xlabel("Interior turn index")
        axis.set_ylabel("Curvature")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_family_boundary_traces(conversation_records: list[dict], output_path: Path) -> None:
    families = sorted({record["family"] for record in conversation_records})
    if not families:
        return

    fig, axes = plt.subplots(len(families), 1, figsize=(10, max(3.5, 2.8 * len(families))), squeeze=False)
    for axis, family in zip(axes[:, 0], families):
        family_records = [record for record in conversation_records if record["family"] == family]
        for record in family_records:
            payload = _load_json(record["output_json"])
            boundary_scores = np.asarray(payload["series"].get("boundary_scores", []), dtype=np.float32)
            if boundary_scores.size == 0:
                continue
            axis.plot(np.arange(1, boundary_scores.size + 1), boundary_scores, marker="o", linewidth=1.5, alpha=0.8)
        axis.set_title(f"Hybrid Boundary Score Traces: {family}")
        axis.set_xlabel("Interior turn index")
        axis.set_ylabel("Hybrid score")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_family_boundary_prominence_traces(conversation_records: list[dict], output_path: Path) -> None:
    families = sorted({record["family"] for record in conversation_records})
    if not families:
        return

    fig, axes = plt.subplots(len(families), 1, figsize=(10, max(3.5, 2.8 * len(families))), squeeze=False)
    for axis, family in zip(axes[:, 0], families):
        family_records = [record for record in conversation_records if record["family"] == family]
        for record in family_records:
            payload = _load_json(record["output_json"])
            values = np.asarray(payload["series"].get("boundary_prominences", []), dtype=np.float32)
            if values.size == 0:
                continue
            axis.plot(np.arange(1, values.size + 1), values, marker="o", linewidth=1.5, alpha=0.8)
        axis.set_title(f"Boundary Prominence Traces: {family}")
        axis.set_xlabel("Interior turn index")
        axis.set_ylabel("Boundary prominence")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rank95_by_family(conversation_records: list[dict], output_path: Path) -> None:
    grouped: dict[tuple[str, str], list[float]] = {}
    for record in conversation_records:
        key = (record["model_key"], record["family"])
        grouped.setdefault(key, []).append(float(record["mean_rank95"]))

    model_keys = sorted({record["model_key"] for record in conversation_records})
    families = sorted({record["family"] for record in conversation_records})
    if not model_keys or not families:
        return

    x = np.arange(len(families), dtype=np.float32)
    width = min(0.75 / max(len(model_keys), 1), 0.28)

    fig, axis = plt.subplots(figsize=(10, 4.8))
    for idx, model_key in enumerate(model_keys):
        offsets = x + (idx - (len(model_keys) - 1) / 2.0) * width
        values = [_mean(grouped.get((model_key, family), [])) for family in families]
        axis.bar(offsets, values, width=width, label=model_key)

    axis.set_xticks(x)
    axis.set_xticklabels(families, rotation=20, ha="right")
    axis.set_ylabel("Mean rank95")
    axis.set_title("Segment Rank95 by Family")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_family_correlation_heatmap(conversation_records: list[dict], output_path: Path) -> None:
    model_keys = sorted({record["model_key"] for record in conversation_records})
    families = sorted({record["family"] for record in conversation_records})
    if not model_keys or not families:
        return

    logit_matrix = np.zeros((len(model_keys), len(families)), dtype=np.float32)
    kl_matrix = np.zeros((len(model_keys), len(families)), dtype=np.float32)
    for model_idx, model_key in enumerate(model_keys):
        for family_idx, family in enumerate(families):
            rows = [
                record for record in conversation_records
                if record["model_key"] == model_key and record["family"] == family
            ]
            if not rows:
                continue
            logit_matrix[model_idx, family_idx] = _mean([float(row["corr_geodesic_vs_logit_l2"]) for row in rows])
            kl_matrix[model_idx, family_idx] = _mean([float(row["corr_geodesic_vs_kl"]) for row in rows])

    _save_single_heatmap(
        logit_matrix,
        xlabels=families,
        ylabels=model_keys,
        title="Mean corr(geodesic, logit L2)",
        output_path=output_path.with_name("family_correlation_geodesic_logitL2.png"),
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    _save_single_heatmap(
        kl_matrix,
        xlabels=families,
        ylabels=model_keys,
        title="Mean corr(geodesic, KL)",
        output_path=output_path.with_name("family_correlation_geodesic_KL.png"),
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )


def plot_boundary_eval_heatmap(conversation_records: list[dict], output_path: Path) -> None:
    model_keys = sorted({record["model_key"] for record in conversation_records})
    families = sorted({record["family"] for record in conversation_records})
    if not model_keys or not families:
        return

    exact_matrix = np.zeros((len(model_keys), len(families)), dtype=np.float32)
    tol1_matrix = np.zeros((len(model_keys), len(families)), dtype=np.float32)
    tol2_matrix = np.zeros((len(model_keys), len(families)), dtype=np.float32)
    tol3_matrix = np.zeros((len(model_keys), len(families)), dtype=np.float32)
    for model_idx, model_key in enumerate(model_keys):
        for family_idx, family in enumerate(families):
            rows = [
                record for record in conversation_records
                if record["model_key"] == model_key and record["family"] == family
            ]
            if not rows:
                continue
            exact_matrix[model_idx, family_idx] = _mean([float(row.get("boundary_f1_exact", 0.0)) for row in rows])
            tol1_matrix[model_idx, family_idx] = _mean([float(row.get("boundary_f1_tol1", 0.0)) for row in rows])
            tol2_matrix[model_idx, family_idx] = _mean([float(row.get("boundary_f1_tol2", 0.0)) for row in rows])
            tol3_matrix[model_idx, family_idx] = _mean([float(row.get("boundary_f1_tol3", 0.0)) for row in rows])

    for matrix, suffix, title in [
        (exact_matrix, "boundary_f1_exact.png", "Mean boundary F1 exact"),
        (tol1_matrix, "boundary_f1_tol1.png", "Mean boundary F1 tol1"),
        (tol2_matrix, "boundary_f1_tol2.png", "Mean boundary F1 tol2"),
        (tol3_matrix, "boundary_f1_tol3.png", "Mean boundary F1 tol3"),
    ]:
        _save_single_heatmap(
            matrix,
            xlabels=families,
            ylabels=model_keys,
            title=title,
            output_path=output_path.with_name(suffix),
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
        )


def plot_geometry_vs_decoder_scatter(conversation_records: list[dict], output_path: Path) -> None:
    families = sorted({record["family"] for record in conversation_records})
    logit_points: dict[str, tuple[list[float], list[float]]] = {}
    kl_points: dict[str, tuple[list[float], list[float]]] = {}

    for family_idx, family in enumerate(families):
        geodesic_values: list[float] = []
        logit_values: list[float] = []
        kl_values: list[float] = []
        for record in conversation_records:
            if record["family"] != family:
                continue
            payload = _load_json(record["output_json"])
            geodesic_values.extend(payload["series"]["state_geodesic_errors"])
            logit_values.extend(payload["series"]["logit_l2"])
            kl_values.extend(payload["series"]["kl"])
        if not geodesic_values:
            continue
        logit_points[family] = (geodesic_values, logit_values)
        kl_points[family] = (geodesic_values, kl_values)

    _save_single_scatter(
        [],
        [],
        families,
        logit_points,
        title="Geodesic Error vs Logit Drift",
        xlabel="State geodesic error",
        ylabel="Logit L2",
        output_path=output_path.with_name("geodesic_vs_logitL2.png"),
    )
    _save_single_scatter(
        [],
        [],
        families,
        kl_points,
        title="Geodesic Error vs KL",
        xlabel="State geodesic error",
        ylabel="KL divergence",
        output_path=output_path.with_name("geodesic_vs_KL.png"),
    )


def plot_rank_energy_curves(conversation_records: list[dict], output_path: Path) -> None:
    grouped: dict[tuple[str, str], list[list[float]]] = {}
    for record in conversation_records:
        payload = _load_json(record["output_json"])
        for segment in payload["segments"]:
            key = (record["model_key"], record["family"])
            grouped.setdefault(key, []).append(segment.get("cumulative_energy", []))

    keys = sorted(grouped)
    if not keys:
        return

    fig, axis = plt.subplots(figsize=(10, 5.2))
    color_map = plt.get_cmap("tab20")
    for idx, key in enumerate(keys):
        curves = [np.asarray(curve, dtype=np.float32) for curve in grouped[key] if curve]
        if not curves:
            continue
        max_len = max(curve.size for curve in curves)
        padded = np.ones((len(curves), max_len), dtype=np.float32)
        for row_idx, curve in enumerate(curves):
            padded[row_idx, : curve.size] = curve
            if curve.size < max_len:
                padded[row_idx, curve.size :] = curve[-1]
        mean_curve = padded.mean(axis=0)
        axis.plot(np.arange(1, max_len + 1), mean_curve, marker="o", linewidth=1.5, alpha=0.85, color=color_map(idx % 20), label=f"{key[0]} / {key[1]}")

    axis.set_title("Average Rank-Energy Curves")
    axis.set_xlabel("Rank")
    axis.set_ylabel("Cumulative energy")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_study_plots(conversation_records: list[dict], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("curvature_traces.png", plot_family_curvature_traces, ["curvature_traces.png"]),
        ("boundary_score_traces.png", plot_family_boundary_traces, ["boundary_score_traces.png"]),
        ("boundary_prominence_traces.png", plot_family_boundary_prominence_traces, ["boundary_prominence_traces.png"]),
        ("rank95_by_family.png", plot_rank95_by_family, ["rank95_by_family.png"]),
        (
            "family_correlation_heatmap.png",
            plot_family_correlation_heatmap,
            ["family_correlation_geodesic_logitL2.png", "family_correlation_geodesic_KL.png"],
        ),
        (
            "boundary_eval_heatmap.png",
            plot_boundary_eval_heatmap,
            ["boundary_f1_exact.png", "boundary_f1_tol1.png", "boundary_f1_tol2.png", "boundary_f1_tol3.png"],
        ),
        (
            "geometry_vs_decoder.png",
            plot_geometry_vs_decoder_scatter,
            ["geodesic_vs_logitL2.png", "geodesic_vs_KL.png"],
        ),
        ("rank_energy_curves.png", plot_rank_energy_curves, ["rank_energy_curves.png"]),
    ]
    written: list[str] = []
    for file_name, fn, expected_outputs in plot_specs:
        output_path = output_dir / file_name
        fn(conversation_records, output_path)
        for expected_name in expected_outputs:
            expected_path = output_dir / expected_name
            if expected_path.exists():
                written.append(str(expected_path))
    return written


def plot_baseline_eval_heatmap(baseline_rows: list[dict], output_path: Path) -> None:
    baseline_names = sorted({row["baseline_name"] for row in baseline_rows})
    families = sorted({row["family"] for row in baseline_rows})
    if not baseline_names or not families:
        return

    exact_matrix = np.zeros((len(baseline_names), len(families)), dtype=np.float32)
    tol1_matrix = np.zeros((len(baseline_names), len(families)), dtype=np.float32)
    tol2_matrix = np.zeros((len(baseline_names), len(families)), dtype=np.float32)
    tol3_matrix = np.zeros((len(baseline_names), len(families)), dtype=np.float32)
    for base_idx, baseline_name in enumerate(baseline_names):
        for fam_idx, family in enumerate(families):
            rows = [row for row in baseline_rows if row["baseline_name"] == baseline_name and row["family"] == family]
            if not rows:
                continue
            exact_matrix[base_idx, fam_idx] = _mean([float(row["boundary_f1_exact"]) for row in rows])
            tol1_matrix[base_idx, fam_idx] = _mean([float(row["boundary_f1_tol1"]) for row in rows])
            tol2_matrix[base_idx, fam_idx] = _mean([float(row.get("boundary_f1_tol2", 0.0)) for row in rows])
            tol3_matrix[base_idx, fam_idx] = _mean([float(row.get("boundary_f1_tol3", 0.0)) for row in rows])

    for matrix, suffix, title in [
        (exact_matrix, "baseline_f1_exact.png", "Baseline boundary F1 exact"),
        (tol1_matrix, "baseline_f1_tol1.png", "Baseline boundary F1 tol1"),
        (tol2_matrix, "baseline_f1_tol2.png", "Baseline boundary F1 tol2"),
        (tol3_matrix, "baseline_f1_tol3.png", "Baseline boundary F1 tol3"),
    ]:
        _save_single_heatmap(
            matrix,
            xlabels=families,
            ylabels=baseline_names,
            title=title,
            output_path=output_path.with_name(suffix),
            cmap="YlOrBr",
            vmin=0.0,
            vmax=1.0,
        )


def generate_baseline_plots(baseline_rows: list[dict], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_eval_heatmap.png"
    plot_baseline_eval_heatmap(baseline_rows, output_path)
    outputs = [
        output_dir / "baseline_f1_exact.png",
        output_dir / "baseline_f1_tol1.png",
        output_dir / "baseline_f1_tol2.png",
        output_dir / "baseline_f1_tol3.png",
    ]
    return [str(path) for path in outputs if path.exists()]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
