from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .analysis import aggregate_boundary_summary_rows, analyze_trajectory
from .baselines import evaluate_baselines
from .conversations import load_conversations_from_paths
from .modeling import list_default_models, resolve_model_spec, transformers_version_ok
from .plotting import generate_baseline_plots, generate_study_plots
from .reporting import render_markdown, summarize_run_directory
from .run_paper1 import DEFAULT_INPUT, run_model_experiment


DEFAULT_STUDY_ROOT = Path(__file__).resolve().parents[1] / "results" / "paper1" / "studies"


def _parse_model_keys(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return ["qwen25_05b"]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_input_paths(primary: Path, extra_raw: str | None) -> list[Path]:
    paths = [primary]
    if extra_raw is not None and extra_raw.strip():
        paths.extend(Path(item.strip()) for item in extra_raw.split(",") if item.strip())
    return paths


def _supported_model_keys(requested_keys: list[str]) -> tuple[list[str], list[str]]:
    import transformers

    installed = transformers.__version__
    runnable: list[str] = []
    skipped: list[str] = []
    for key in requested_keys:
        spec = resolve_model_spec(key)
        if spec is None:
            raise ValueError(f"Unknown model key: {key}")
        if transformers_version_ok(installed, spec.min_transformers_version):
            runnable.append(key)
        else:
            skipped.append(key)
    return runnable, skipped


def _conversation_rows(model_key: str, run_summary: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in run_summary["conversations"]:
        summary = record["summary"]
        rows.append(
            {
                "model_key": model_key,
                "model_name": run_summary["model_name"],
                "family": record["family"],
                "conversation_id": record["conversation_id"],
                "num_turns": record["num_turns"],
                "num_candidate_boundaries": summary.get("num_candidate_boundaries", 0),
                "num_gold_boundaries": summary.get("num_gold_boundaries", 0),
                "gold_boundary_density": summary.get("gold_boundary_density", 0.0),
                "mean_rank95": summary["mean_rank95"],
                "mean_curvature": summary["mean_curvature"],
                "mean_turning_angle": summary.get("mean_turning_angle", 0.0),
                "mean_rank_jump": summary.get("mean_rank_jump", 0.0),
                "mean_subspace_shift": summary.get("mean_subspace_shift", 0.0),
                "mean_boundary_score": summary.get("mean_boundary_score", 0.0),
                "mean_boundary_prominence": summary.get("mean_boundary_prominence", 0.0),
                "boundary_tp_exact": summary.get("boundary_tp_exact", 0.0),
                "boundary_fp_exact": summary.get("boundary_fp_exact", 0.0),
                "boundary_fn_exact": summary.get("boundary_fn_exact", 0.0),
                "boundary_f1_exact": summary.get("boundary_f1_exact", 0.0),
                "boundary_tp_tol1": summary.get("boundary_tp_tol1", 0.0),
                "boundary_fp_tol1": summary.get("boundary_fp_tol1", 0.0),
                "boundary_fn_tol1": summary.get("boundary_fn_tol1", 0.0),
                "boundary_f1_tol1": summary.get("boundary_f1_tol1", 0.0),
                "boundary_tp_tol2": summary.get("boundary_tp_tol2", 0.0),
                "boundary_fp_tol2": summary.get("boundary_fp_tol2", 0.0),
                "boundary_fn_tol2": summary.get("boundary_fn_tol2", 0.0),
                "boundary_f1_tol2": summary.get("boundary_f1_tol2", 0.0),
                "boundary_tp_tol3": summary.get("boundary_tp_tol3", 0.0),
                "boundary_fp_tol3": summary.get("boundary_fp_tol3", 0.0),
                "boundary_fn_tol3": summary.get("boundary_fn_tol3", 0.0),
                "boundary_f1_tol3": summary.get("boundary_f1_tol3", 0.0),
                "boundary_nearest_distance": summary.get("boundary_nearest_distance", 0.0),
                "boundary_windowdiff": summary.get("boundary_windowdiff", 0.0),
                "boundary_pk": summary.get("boundary_pk", 0.0),
                "boundary_auprc": summary.get("boundary_auprc", 0.0),
                "mean_state_geodesic_error": summary["mean_state_geodesic_error"],
                "mean_logit_l2": summary["mean_logit_l2"],
                "mean_kl": summary["mean_kl"],
                "top1_agreement": summary["top1_agreement"],
                "corr_geodesic_vs_logit_l2": summary["corr_geodesic_vs_logit_l2"],
                "corr_geodesic_vs_kl": summary["corr_geodesic_vs_kl"],
                "output_json": record["output_json"],
                "output_npz": record["output_npz"],
                "summary": summary,
            }
        )
    return rows


def _write_conversation_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_key",
        "model_name",
        "family",
        "conversation_id",
        "num_turns",
        "num_candidate_boundaries",
        "num_gold_boundaries",
        "gold_boundary_density",
        "mean_rank95",
        "mean_curvature",
        "mean_turning_angle",
        "mean_rank_jump",
        "mean_subspace_shift",
        "mean_boundary_score",
        "mean_boundary_prominence",
        "boundary_tp_exact",
        "boundary_fp_exact",
        "boundary_fn_exact",
        "boundary_f1_exact",
        "boundary_tp_tol1",
        "boundary_fp_tol1",
        "boundary_fn_tol1",
        "boundary_f1_tol1",
        "boundary_tp_tol2",
        "boundary_fp_tol2",
        "boundary_fn_tol2",
        "boundary_f1_tol2",
        "boundary_tp_tol3",
        "boundary_fp_tol3",
        "boundary_fn_tol3",
        "boundary_f1_tol3",
        "boundary_nearest_distance",
        "boundary_windowdiff",
        "boundary_pk",
        "boundary_auprc",
        "mean_state_geodesic_error",
        "mean_logit_l2",
        "mean_kl",
        "top1_agreement",
        "corr_geodesic_vs_logit_l2",
        "corr_geodesic_vs_kl",
        "output_json",
        "output_npz",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def _summarize_baselines(baseline_rows: list[dict[str, object]]) -> dict[str, object]:
    summaries: dict[str, object] = {}
    baseline_names = sorted({row["baseline_name"] for row in baseline_rows})
    for baseline_name in baseline_names:
        rows = [row for row in baseline_rows if row["baseline_name"] == baseline_name]
        boundary_aggregate = aggregate_boundary_summary_rows(rows)
        families = sorted({row["family"] for row in rows})
        family_summary = {
            family: {
                "num_conversations": len([row for row in rows if row["family"] == family]),
                **aggregate_boundary_summary_rows([row for row in rows if row["family"] == family]),
                "mean_ordered_boundary_mae": sum(float(row["ordered_boundary_mae"]) for row in rows if row["family"] == family)
                / max(len([row for row in rows if row["family"] == family]), 1),
                "mean_oversegmentation_rate": sum(float(row["oversegmentation_rate"]) for row in rows if row["family"] == family)
                / max(len([row for row in rows if row["family"] == family]), 1),
                "mean_miss_rate": sum(float(row["miss_rate"]) for row in rows if row["family"] == family)
                / max(len([row for row in rows if row["family"] == family]), 1),
            }
            for family in families
        }
        summaries[baseline_name] = {
            "aggregate": {
                "num_conversations": len(rows),
                "mean_ordered_boundary_mae": sum(float(row["ordered_boundary_mae"]) for row in rows) / max(len(rows), 1),
                "mean_oversegmentation_rate": sum(float(row["oversegmentation_rate"]) for row in rows) / max(len(rows), 1),
                "mean_miss_rate": sum(float(row["miss_rate"]) for row in rows) / max(len(rows), 1),
                **boundary_aggregate,
            },
            "family_summary": family_summary,
        }
    return summaries


def _summarize_boundary_variants(run_summary: dict[str, object]) -> dict[str, object]:
    variant_rows: dict[str, list[dict[str, object]]] = {}
    for record in run_summary.get("conversations", []):
        variants = record.get("boundary_variants", {})
        for variant_name, payload in variants.items():
            variant_rows.setdefault(variant_name, []).append(payload["metrics"])

    def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
        precision = tp / (tp + fp) if tp + fp > 0.0 else (1.0 if fn == 0.0 else 0.0)
        recall = tp / (tp + fn) if tp + fn > 0.0 else (1.0 if fp == 0.0 else 0.0)
        if precision + recall < 1e-8:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        return float(precision), float(recall), float(f1)

    summary: dict[str, object] = {}
    for variant_name, rows in sorted(variant_rows.items()):
        exact_tp = sum(float(row["exact"]["tp"]) for row in rows)
        exact_fp = sum(float(row["exact"]["fp"]) for row in rows)
        exact_fn = sum(float(row["exact"]["fn"]) for row in rows)
        tol2_tp = sum(float(row["tolerance_2"]["tp"]) for row in rows)
        tol2_fp = sum(float(row["tolerance_2"]["fp"]) for row in rows)
        tol2_fn = sum(float(row["tolerance_2"]["fn"]) for row in rows)
        _, _, exact_f1 = _prf(exact_tp, exact_fp, exact_fn)
        _, _, tol2_f1 = _prf(tol2_tp, tol2_fp, tol2_fn)
        summary[variant_name] = {
            "num_conversations": len(rows),
            "macro_boundary_f1_exact": float(np.mean([float(row["exact"]["f1"]) for row in rows])),
            "micro_boundary_f1_exact": exact_f1,
            "macro_boundary_f1_tol2": float(np.mean([float(row["tolerance_2"]["f1"]) for row in rows])),
            "micro_boundary_f1_tol2": tol2_f1,
            "mean_boundary_auprc": float(np.mean([float(row["boundary_auprc"]) for row in rows])),
            "mean_boundary_nearest_distance": float(np.mean([float(row["mean_nearest_boundary_distance"]) for row in rows])),
        }
    return summary


def _stable_seed(*parts: str) -> int:
    joined = "::".join(parts)
    return int(hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16], 16)


def _bootstrap_statistic(
    rows: list[dict[str, object]],
    stat_fn,
    num_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    if not rows:
        return {"estimate": 0.0, "bootstrap_std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    estimate = float(stat_fn(rows))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(num_bootstrap):
        sample = [rows[int(idx)] for idx in rng.integers(0, len(rows), size=len(rows))]
        values.append(float(stat_fn(sample)))
    samples = np.asarray(values, dtype=np.float32)
    return {
        "estimate": estimate,
        "bootstrap_std": float(samples.std(ddof=1)) if samples.size > 1 else 0.0,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def _model_uncertainty_summary(rows: list[dict[str, object]], seed: int) -> dict[str, object]:
    return {
        "mean_rank95": _bootstrap_statistic(
            rows,
            lambda sample: np.mean([float(row["mean_rank95"]) for row in sample]),
            seed=seed + 11,
        ),
        "mean_corr_geodesic_vs_logit_l2": _bootstrap_statistic(
            rows,
            lambda sample: np.mean([float(row["corr_geodesic_vs_logit_l2"]) for row in sample]),
            seed=seed + 13,
        ),
        "micro_boundary_f1_exact": _bootstrap_statistic(
            rows,
            lambda sample: aggregate_boundary_summary_rows(sample)["micro_boundary_f1_exact"],
            seed=seed + 17,
        ),
        "micro_boundary_f1_tol2": _bootstrap_statistic(
            rows,
            lambda sample: aggregate_boundary_summary_rows(sample)["micro_boundary_f1_tol2"],
            seed=seed + 19,
        ),
        "mean_boundary_auprc": _bootstrap_statistic(
            rows,
            lambda sample: np.mean([float(row["boundary_auprc"]) for row in sample]),
            seed=seed + 23,
        ),
    }


def _paired_permutation_test(
    values_a: list[float],
    values_b: list[float],
    seed: int,
    num_permutations: int = 10000,
) -> dict[str, float]:
    if not values_a or not values_b or len(values_a) != len(values_b):
        return {"mean_difference": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
    diffs = np.asarray(values_a, dtype=np.float32) - np.asarray(values_b, dtype=np.float32)
    observed = float(diffs.mean())
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(1000):
        sample = diffs[rng.integers(0, diffs.size, size=diffs.size)]
        boot.append(float(sample.mean()))
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(num_permutations, diffs.size))
    permuted = (signs * diffs[None, :]).mean(axis=1)
    p_value = float(np.mean(np.abs(permuted) >= abs(observed)))
    return {
        "mean_difference": observed,
        "ci_low": float(np.quantile(np.asarray(boot, dtype=np.float32), 0.025)),
        "ci_high": float(np.quantile(np.asarray(boot, dtype=np.float32), 0.975)),
        "p_value": p_value,
    }


def _variant_significance(run_summary: dict[str, object], seed: int) -> dict[str, object]:
    metric_extractors = {
        "exact_f1": lambda payload: float(payload["metrics"]["exact"]["f1"]),
        "tol2_f1": lambda payload: float(payload["metrics"]["tolerance_2"]["f1"]),
        "auprc": lambda payload: float(payload["metrics"]["boundary_auprc"]),
    }
    comparisons = [
        ("geometry_lexical", "lexical_only"),
        ("geometry_lexical", "geometry_only"),
    ]
    records = run_summary.get("conversations", [])
    summary: dict[str, object] = {}
    for left_name, right_name in comparisons:
        comparison_key = f"{left_name}_vs_{right_name}"
        summary[comparison_key] = {}
        for metric_name, extractor in metric_extractors.items():
            left_values = []
            right_values = []
            for record in records:
                variants = record.get("boundary_variants", {})
                if left_name not in variants or right_name not in variants:
                    continue
                left_values.append(extractor(variants[left_name]))
                right_values.append(extractor(variants[right_name]))
            summary[comparison_key][metric_name] = _paired_permutation_test(
                left_values,
                right_values,
                seed=seed + _stable_seed(comparison_key, metric_name),
            )
    return summary


def _permutation_null_correlation(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    num_permutations: int = 256,
) -> tuple[float, float]:
    if x.size < 2 or y.size < 2:
        return 0.0, 1.0
    real = float(np.corrcoef(x, y)[0, 1]) if float(np.std(x)) > 1e-8 and float(np.std(y)) > 1e-8 else 0.0
    rng = np.random.default_rng(seed)
    permuted = []
    for _ in range(num_permutations):
        shuffled = y[rng.permutation(y.size)]
        corr = float(np.corrcoef(x, shuffled)[0, 1]) if float(np.std(shuffled)) > 1e-8 else 0.0
        permuted.append(corr)
    permuted_array = np.asarray(permuted, dtype=np.float32)
    p_value = float(np.mean(np.abs(permuted_array) >= abs(real)))
    return float(permuted_array.mean()), p_value


def _model_null_controls(
    rows: list[dict[str, object]],
    rank_energy: float,
    max_segment_len: int,
    min_segment_len: int,
    seed: int,
) -> dict[str, object]:
    real_rank95 = np.asarray([float(row["mean_rank95"]) for row in rows], dtype=np.float32)
    shuffled_rank95: list[float] = []
    real_corr = np.asarray([float(row["corr_geodesic_vs_logit_l2"]) for row in rows], dtype=np.float32)
    permuted_corr_mean: list[float] = []
    permutation_p_values: list[float] = []

    for row in rows:
        npz = np.load(row["output_npz"])
        states = npz["original_states"].astype(np.float32)
        shuffle_rng = np.random.default_rng(_stable_seed(str(row["model_key"]), str(row["conversation_id"])))
        shuffled_states = states[shuffle_rng.permutation(states.shape[0])]
        dummy_logits = np.zeros((shuffled_states.shape[0], 2), dtype=np.float32)
        shuffled_analysis = analyze_trajectory(
            states=shuffled_states,
            logits=dummy_logits,
            reconstructed_logits=dummy_logits,
            gold_boundaries=[],
            lexical_boundary_scores=None,
            rank_energy=rank_energy,
            max_segment_len=max_segment_len,
            min_segment_len=min_segment_len,
        )
        shuffled_rank95.append(float(shuffled_analysis["summary"]["mean_rank95"]))

        payload = json.loads(Path(row["output_json"]).read_text(encoding="utf-8"))
        geodesic = np.asarray(payload["series"]["state_geodesic_errors"], dtype=np.float32)
        logit_l2 = np.asarray(payload["series"]["logit_l2"], dtype=np.float32)
        perm_mean, perm_p = _permutation_null_correlation(
            geodesic,
            logit_l2,
            seed=_stable_seed(str(row["model_key"]), str(row["conversation_id"]), "corr"),
        )
        permuted_corr_mean.append(perm_mean)
        permutation_p_values.append(perm_p)

    shuffled_rank95_array = np.asarray(shuffled_rank95, dtype=np.float32)
    real_corr_gap = real_corr - np.asarray(permuted_corr_mean, dtype=np.float32)
    rank95_gap = shuffled_rank95_array - real_rank95
    h1_test = _paired_permutation_test(shuffled_rank95, real_rank95.tolist(), seed=seed + 31)
    h3_test = _paired_permutation_test(real_corr.tolist(), permuted_corr_mean, seed=seed + 37)
    return {
        "h1_shuffled_turn_order": {
            "real_mean_rank95": float(real_rank95.mean()) if real_rank95.size else 0.0,
            "shuffled_mean_rank95": float(shuffled_rank95_array.mean()) if shuffled_rank95_array.size else 0.0,
            "mean_gap": float(rank95_gap.mean()) if rank95_gap.size else 0.0,
            "ci_low": h1_test["ci_low"],
            "ci_high": h1_test["ci_high"],
            "p_value": h1_test["p_value"],
        },
        "h3_permuted_alignment": {
            "real_mean_corr": float(real_corr.mean()) if real_corr.size else 0.0,
            "permuted_mean_corr": float(np.mean(permuted_corr_mean)) if permuted_corr_mean else 0.0,
            "mean_gap": float(real_corr_gap.mean()) if real_corr_gap.size else 0.0,
            "ci_low": h3_test["ci_low"],
            "ci_high": h3_test["ci_high"],
            "p_value": h3_test["p_value"],
            "mean_turnwise_permutation_p_value": float(np.mean(permutation_p_values)) if permutation_p_values else 1.0,
        },
    }


def _write_baseline_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "baseline_name",
        "family",
        "conversation_id",
        "num_turns",
        "num_candidate_boundaries",
        "gold_boundary_density",
        "gold_boundaries",
        "predicted_boundaries",
        "ordered_boundary_mae",
        "boundary_nearest_distance",
        "oversegmentation_rate",
        "miss_rate",
        "boundary_windowdiff",
        "boundary_pk",
        "boundary_auprc",
        "boundary_tp_exact",
        "boundary_fp_exact",
        "boundary_fn_exact",
        "boundary_precision_exact",
        "boundary_recall_exact",
        "boundary_f1_exact",
        "boundary_tp_tol1",
        "boundary_fp_tol1",
        "boundary_fn_tol1",
        "boundary_precision_tol1",
        "boundary_recall_tol1",
        "boundary_f1_tol1",
        "boundary_tp_tol2",
        "boundary_fp_tol2",
        "boundary_fn_tol2",
        "boundary_precision_tol2",
        "boundary_recall_tol2",
        "boundary_f1_tol2",
        "boundary_tp_tol3",
        "boundary_fp_tol3",
        "boundary_fn_tol3",
        "boundary_precision_tol3",
        "boundary_recall_tol3",
        "boundary_f1_tol3",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def _benchmark_audit(conversations: list, baseline_summary: dict[str, object]) -> dict[str, object]:
    candidate_counts = [max(len(conversation.turns) - 2, 0) for conversation in conversations]
    gold_counts = [len(conversation.boundary_indices or []) for conversation in conversations]
    turn_counts = [len(conversation.turns) for conversation in conversations]
    densities = [gold / max(candidates, 1) for gold, candidates in zip(gold_counts, candidate_counts)]
    zero_gold_fraction = float(np.mean([1.0 if gold == 0 else 0.0 for gold in gold_counts])) if gold_counts else 0.0

    oracle_random_expected = []
    oracle_random_expected_nonempty = []
    for gold, candidates in zip(gold_counts, candidate_counts):
        if gold == 0:
            oracle_random_expected.append(1.0)
            continue
        if candidates == 0:
            oracle_random_expected.append(0.0)
            continue
        expected = gold / candidates
        oracle_random_expected.append(float(expected))
        oracle_random_expected_nonempty.append(float(expected))

    family_rows: dict[str, list[tuple[int, int, int]]] = {}
    for conversation, turns, candidates, gold in zip(conversations, turn_counts, candidate_counts, gold_counts):
        family_rows.setdefault(conversation.family, []).append((turns, candidates, gold))

    family_summary: dict[str, dict[str, float]] = {}
    for family, rows in sorted(family_rows.items()):
        turns = [row[0] for row in rows]
        candidates = [row[1] for row in rows]
        gold = [row[2] for row in rows]
        family_summary[family] = {
            "num_conversations": len(rows),
            "mean_num_turns": float(np.mean(turns)) if turns else 0.0,
            "mean_num_candidate_boundaries": float(np.mean(candidates)) if candidates else 0.0,
            "mean_num_gold_boundaries": float(np.mean(gold)) if gold else 0.0,
            "mean_gold_boundary_density": float(
                np.mean([g / max(c, 1) for g, c in zip(gold, candidates)])
            )
            if rows
            else 0.0,
        }

    oracle_baseline = baseline_summary.get("oracle_random_matched_count", {}).get("aggregate", {})
    return {
        "num_conversations": len(conversations),
        "mean_num_turns": float(np.mean(turn_counts)) if turn_counts else 0.0,
        "median_num_turns": float(np.median(turn_counts)) if turn_counts else 0.0,
        "mean_num_candidate_boundaries": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "median_num_candidate_boundaries": float(np.median(candidate_counts)) if candidate_counts else 0.0,
        "mean_num_gold_boundaries": float(np.mean(gold_counts)) if gold_counts else 0.0,
        "median_num_gold_boundaries": float(np.median(gold_counts)) if gold_counts else 0.0,
        "mean_gold_boundary_density": float(np.mean(densities)) if densities else 0.0,
        "median_gold_boundary_density": float(np.median(densities)) if densities else 0.0,
        "zero_gold_boundary_fraction": zero_gold_fraction,
        "oracle_random_expected_macro_f1_exact": float(np.mean(oracle_random_expected)) if oracle_random_expected else 0.0,
        "oracle_random_expected_macro_f1_exact_nonempty": (
            float(np.mean(oracle_random_expected_nonempty)) if oracle_random_expected_nonempty else 0.0
        ),
        "oracle_random_empirical_macro_f1_exact": float(oracle_baseline.get("macro_boundary_f1_exact", 0.0)),
        "oracle_random_empirical_micro_f1_exact": float(oracle_baseline.get("micro_boundary_f1_exact", 0.0)),
        "family_summary": family_summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a multi-model Paper 1 study with aggregation and plots.")
    parser.add_argument("--study-name", default="bootstrap")
    parser.add_argument("--model-keys", default="qwen25_05b")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--extra-input-paths", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--state-layer", type=int, default=-1)
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--rank-energy", type=float, default=0.95)
    parser.add_argument("--max-segment-len", type=int, default=6)
    parser.add_argument("--min-segment-len", type=int, default=3)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-incompatible-models", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.list_models:
        for spec in list_default_models():
            print(f"{spec.key:12s}  {spec.model_name:36s}  min_tf={spec.min_transformers_version}")
        return

    requested_keys = _parse_model_keys(args.model_keys)
    input_paths = _parse_input_paths(args.input_path, args.extra_input_paths)
    runnable_keys, skipped_keys = _supported_model_keys(requested_keys)
    if skipped_keys and not args.skip_incompatible_models:
        skipped = ", ".join(skipped_keys)
        raise RuntimeError(
            f"These model keys are incompatible with the current transformers install: {skipped}. "
            "Use --skip-incompatible-models or upgrade the environment."
        )
    if not runnable_keys:
        raise RuntimeError("No compatible models selected for the study.")

    study_dir = args.output_root / args.study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    model_runs: dict[str, dict[str, object]] = {}
    variant_runs: dict[str, dict[str, object]] = {}
    uncertainty_runs: dict[str, dict[str, object]] = {}
    significance_runs: dict[str, dict[str, object]] = {}
    conversation_rows: list[dict[str, object]] = []
    conversations = load_conversations_from_paths(input_paths)
    if args.limit_conversations is not None:
        conversations = conversations[: args.limit_conversations]
    for model_key in runnable_keys:
        spec = resolve_model_spec(model_key)
        assert spec is not None
        model_dir = study_dir / model_key
        run_summary = run_model_experiment(
            model_name=spec.model_name,
            device=args.device,
            dtype=args.dtype,
            state_layer=args.state_layer,
            input_path=args.input_path,
            input_paths=input_paths,
            output_dir=model_dir,
            limit_conversations=args.limit_conversations,
            max_turns=args.max_turns,
            max_input_tokens=args.max_input_tokens,
            rank_energy=args.rank_energy,
            max_segment_len=args.max_segment_len,
            min_segment_len=args.min_segment_len,
            model_key=model_key,
        )
        model_runs[model_key] = summarize_run_directory(model_dir)
        variant_runs[model_key] = _summarize_boundary_variants(run_summary)
        conversation_rows.extend(_conversation_rows(model_key, run_summary))

    for model_key in runnable_keys:
        model_rows = [row for row in conversation_rows if row["model_key"] == model_key]
        uncertainty_runs[model_key] = _model_uncertainty_summary(
            model_rows,
            seed=_stable_seed(args.study_name, model_key, "uncertainty"),
        )
        run_summary_payload = json.loads((study_dir / model_key / "run_summary.json").read_text(encoding="utf-8"))
        significance_runs[model_key] = _variant_significance(
            run_summary_payload,
            seed=_stable_seed(args.study_name, model_key, "significance"),
        )

    _write_conversation_csv(conversation_rows, study_dir / "conversation_summary.csv")
    baseline_rows = [asdict(row) for row in evaluate_baselines(conversations, args.max_segment_len, args.min_segment_len)]
    _write_baseline_csv(baseline_rows, study_dir / "baseline_conversation_summary.csv")
    baseline_summary = _summarize_baselines(baseline_rows)
    (study_dir / "baseline_summary.json").write_text(json.dumps(baseline_summary, indent=2), encoding="utf-8")
    (study_dir / "variant_summary.json").write_text(json.dumps(variant_runs, indent=2), encoding="utf-8")
    (study_dir / "confidence_summary.json").write_text(json.dumps(uncertainty_runs, indent=2), encoding="utf-8")
    (study_dir / "significance_summary.json").write_text(json.dumps(significance_runs, indent=2), encoding="utf-8")
    benchmark_audit = _benchmark_audit(conversations, baseline_summary)
    (study_dir / "benchmark_audit.json").write_text(json.dumps(benchmark_audit, indent=2), encoding="utf-8")
    null_control_runs = {
        model_key: _model_null_controls(
            [row for row in conversation_rows if row["model_key"] == model_key],
            rank_energy=args.rank_energy,
            max_segment_len=args.max_segment_len,
            min_segment_len=args.min_segment_len,
            seed=_stable_seed(args.study_name, model_key, "null"),
        )
        for model_key in runnable_keys
    }
    (study_dir / "null_controls.json").write_text(json.dumps(null_control_runs, indent=2), encoding="utf-8")

    plot_files: list[str] = []
    if not args.skip_plots:
        plot_files = generate_study_plots(conversation_rows, study_dir / "plots")
        plot_files.extend(generate_baseline_plots(baseline_rows, study_dir / "plots"))

    study_summary = {
        "study_name": args.study_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(args.input_path),
        "input_paths": [str(path) for path in input_paths],
        "requested_model_keys": requested_keys,
        "runnable_model_keys": runnable_keys,
        "skipped_model_keys": skipped_keys,
        "models": model_runs,
        "boundary_variants": variant_runs,
        "uncertainty": uncertainty_runs,
        "significance": significance_runs,
        "baselines": baseline_summary,
        "benchmark_audit": benchmark_audit,
        "null_controls": null_control_runs,
        "plot_files": plot_files,
    }
    (study_dir / "study_summary.json").write_text(json.dumps(study_summary, indent=2), encoding="utf-8")

    md_lines = [
        f"# Paper 1 Study: {args.study_name}",
        "",
        f"- Input: `{args.input_path}`",
        f"- Input files: {', '.join(str(path) for path in input_paths)}",
        f"- Models run: {', '.join(runnable_keys)}",
    ]
    if skipped_keys:
        md_lines.append(f"- Models skipped: {', '.join(skipped_keys)}")
    md_lines.append("")
    md_lines.extend(
        [
            "## Benchmark Audit",
            "",
            f"- Mean turns / conversation: {benchmark_audit['mean_num_turns']:.3f}",
            f"- Median turns / conversation: {benchmark_audit['median_num_turns']:.3f}",
            f"- Mean candidate boundaries / conversation: {benchmark_audit['mean_num_candidate_boundaries']:.3f}",
            f"- Median candidate boundaries / conversation: {benchmark_audit['median_num_candidate_boundaries']:.3f}",
            f"- Mean gold boundaries / conversation: {benchmark_audit['mean_num_gold_boundaries']:.3f}",
            f"- Mean gold boundary density: {benchmark_audit['mean_gold_boundary_density']:.3f}",
            f"- Zero-gold conversations: {benchmark_audit['zero_gold_boundary_fraction']:.3f}",
            f"- Oracle random expected macro exact F1: {benchmark_audit['oracle_random_expected_macro_f1_exact']:.3f}",
            f"- Oracle random expected macro exact F1 (nonempty only): {benchmark_audit['oracle_random_expected_macro_f1_exact_nonempty']:.3f}",
            f"- Oracle random empirical macro exact F1: {benchmark_audit['oracle_random_empirical_macro_f1_exact']:.3f}",
            f"- Oracle random empirical micro exact F1: {benchmark_audit['oracle_random_empirical_micro_f1_exact']:.3f}",
            "",
            "### Caveats",
            "",
            "- Boundary metrics are high-variance here because most conversations are short and expose only a few candidate inter-turn boundaries.",
            "- `oracle_random_matched_count` is a chance-reference oracle that uses the gold boundary count; it is not a practical baseline.",
            "- Macro exact F1 is inflated by no-boundary conversations because predicting no boundaries receives a perfect score on those examples.",
            "- The strongest supported Paper 1 claims remain low-rank structure and geometry-to-decoder relevance. Boundary recovery is secondary, mixed, and formulation-sensitive.",
            "",
        ]
    )
    for model_key in runnable_keys:
        md_lines.append(f"## {model_key}")
        md_lines.append("")
        md_lines.append(render_markdown(model_runs[model_key]).rstrip())
        md_lines.append("")
        if uncertainty_runs.get(model_key):
            md_lines.append("### Uncertainty")
            md_lines.append("")
            for metric_name, payload in uncertainty_runs[model_key].items():
                md_lines.append(
                    f"- {metric_name}: estimate {payload['estimate']:.3f}, bootstrap std {payload['bootstrap_std']:.3f}, 95% CI [{payload['ci_low']:.3f}, {payload['ci_high']:.3f}]"
                )
            md_lines.append("")
        if null_control_runs.get(model_key):
            md_lines.append("### Null Controls")
            md_lines.append("")
            h1 = null_control_runs[model_key]["h1_shuffled_turn_order"]
            h3 = null_control_runs[model_key]["h3_permuted_alignment"]
            md_lines.append(
                f"- H1 shuffled turn order: real rank95 {h1['real_mean_rank95']:.3f}, shuffled {h1['shuffled_mean_rank95']:.3f}, gap {h1['mean_gap']:.3f}, 95% CI [{h1['ci_low']:.3f}, {h1['ci_high']:.3f}], p={h1['p_value']:.4f}"
            )
            md_lines.append(
                f"- H3 permuted alignment: real corr {h3['real_mean_corr']:.3f}, permuted {h3['permuted_mean_corr']:.3f}, gap {h3['mean_gap']:.3f}, 95% CI [{h3['ci_low']:.3f}, {h3['ci_high']:.3f}], p={h3['p_value']:.4f}"
            )
            md_lines.append("")
        if variant_runs.get(model_key):
            md_lines.append("### Boundary Variant Ablation")
            md_lines.append("")
            for variant_name, payload in variant_runs[model_key].items():
                md_lines.append(f"- {variant_name}: micro exact {payload['micro_boundary_f1_exact']:.3f}, micro tol2 {payload['micro_boundary_f1_tol2']:.3f}, mean AUPRC {payload['mean_boundary_auprc']:.3f}, mean nearest distance {payload['mean_boundary_nearest_distance']:.3f}")
            md_lines.append("")
        if significance_runs.get(model_key):
            md_lines.append("### Boundary Significance")
            md_lines.append("")
            for comparison_name, payload in significance_runs[model_key].items():
                md_lines.append(
                    f"- {comparison_name}: exact diff {payload['exact_f1']['mean_difference']:.3f} (p={payload['exact_f1']['p_value']:.4f}), tol2 diff {payload['tol2_f1']['mean_difference']:.3f} (p={payload['tol2_f1']['p_value']:.4f}), AUPRC diff {payload['auprc']['mean_difference']:.3f} (p={payload['auprc']['p_value']:.4f})"
                )
            md_lines.append("")
    if baseline_summary:
        md_lines.append("## Baselines")
        md_lines.append("")
        for baseline_name, payload in baseline_summary.items():
            md_lines.append(f"### {baseline_name}")
            md_lines.append("")
            aggregate = payload["aggregate"]
            md_lines.append(f"- Macro boundary F1 exact: {aggregate['macro_boundary_f1_exact']:.3f}")
            md_lines.append(f"- Micro boundary F1 exact: {aggregate['micro_boundary_f1_exact']:.3f}")
            md_lines.append(f"- Macro boundary F1 tol1: {aggregate['macro_boundary_f1_tol1']:.3f}")
            md_lines.append(f"- Micro boundary F1 tol1: {aggregate['micro_boundary_f1_tol1']:.3f}")
            md_lines.append(f"- Macro boundary F1 tol2: {aggregate['macro_boundary_f1_tol2']:.3f}")
            md_lines.append(f"- Micro boundary F1 tol2: {aggregate['micro_boundary_f1_tol2']:.3f}")
            md_lines.append(f"- Macro boundary F1 tol3: {aggregate['macro_boundary_f1_tol3']:.3f}")
            md_lines.append(f"- Micro boundary F1 tol3: {aggregate['micro_boundary_f1_tol3']:.3f}")
            md_lines.append(f"- Mean nearest boundary distance: {aggregate['mean_boundary_nearest_distance']:.3f}")
            md_lines.append(f"- Mean WindowDiff: {aggregate['mean_boundary_windowdiff']:.3f}")
            md_lines.append(f"- Mean Pk: {aggregate['mean_boundary_pk']:.3f}")
            md_lines.append(f"- Mean boundary AUPRC: {aggregate['mean_boundary_auprc']:.3f}")
            md_lines.append(f"- Mean candidate boundaries / conversation: {aggregate['mean_num_candidate_boundaries']:.3f}")
            md_lines.append(f"- Mean gold boundary density: {aggregate['mean_gold_boundary_density']:.3f}")
            md_lines.append(f"- Mean ordered boundary MAE: {aggregate['mean_ordered_boundary_mae']:.3f}")
            md_lines.append(f"- Mean oversegmentation rate: {aggregate['mean_oversegmentation_rate']:.3f}")
            md_lines.append(f"- Mean miss rate: {aggregate['mean_miss_rate']:.3f}")
            md_lines.append("")
    if plot_files:
        md_lines.append("## Plots")
        md_lines.append("")
        for plot_file in plot_files:
            md_lines.append(f"- `{plot_file}`")
    (study_dir / "study_report.md").write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")

    print(f"Wrote study outputs to {study_dir}")


if __name__ == "__main__":
    main()
