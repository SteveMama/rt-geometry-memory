"""Gate 1 reanalysis with answer-NLL-defined oracle harm.

Review fix #2 (the circularity concern): the Gate 1 oracle in the manuscript
ranks features against a harm scalar that mixes logit-L2 damage with answer
NLL damage. Because the paper's own characterization shows geometric
distortion correlates 0.989-0.994 with logit drift, a logit-defined harm
target structurally favors geometry features. This module recomputes every
Gate 1 ranking metric twice from the already-tracked ``candidate_rows.csv``:

  1. against logit harm only   (``delta_logit_l2``)
  2. against answer harm only  (``delta_answer_avg_neg_logprob_delta``,
     restricted to rows with a real behavior label)

and reports them side by side. If the geometry ranking gain survives under
answer-defined harm, the circularity objection is dead. If it does not, the
oracle section must be rewritten before submission.

It also runs the gate-threshold sensitivity sweep the runbook calls for
(``papers/paper3_gate1_real_runbook.md`` line 57).

CPU only — runs on already-tracked artifacts:

    python -m june_fixes.answer_harm_oracle.answer_harm_gate1 \
      --candidate-rows paper3_gate1_scaleup_multigpu_merged_results/paper3_gate1_scaleup_multigpu_oracle_msc_valid_32conv/candidate_rows.csv \
      --benchmark-name msc_valid \
      --output-dir results/june_fixes/answer_harm_oracle/msc_valid
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper3_codec.stats import (
    bootstrap_mean_ci,
    collapse_rows_by_keys,
    kendall_tau,
    paired_signflip_test,
    topk_recall,
)

BASELINE_FEATURE = "semantic_score"
CANDIDATE_FEATURES = [
    "semantic_score",
    "geometry_score",
    "query_geom_v2_risk",
    "combined_structural_score",
]
HARM_TARGETS = {
    "logit_harm": "delta_logit_l2",
    "answer_harm": "delta_answer_avg_neg_logprob_delta",
}
GATE_TAU_THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
GATE_RECALL_THRESHOLD = 0.05
MIN_POOL_SIZE = 3
TRUTHY = {"1", "1.0", "true", "True", "TRUE"}


def _is_true(value: Any) -> bool:
    return str(value).strip() in TRUTHY


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_candidate_rows(path: Path, benchmark_name: str | None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if benchmark_name:
        rows = [row for row in rows if row.get("benchmark") in (benchmark_name, None, "")]
    if not rows:
        raise ValueError(f"no candidate rows loaded from {path}")
    return rows


def _pool_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["model_key"]),
        str(row["budget_fraction"]),
        str(row["conversation_id"]),
        str(row["target_turn"]),
    )


def compute_pool_metrics(
    rows: list[dict[str, Any]],
    *,
    view: str,
) -> list[dict[str, Any]]:
    """One record per (pool, harm target): tau and top-5 recall per feature."""
    pools: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if view == "semantic_shortlist" and not _is_true(row.get("in_semantic_topk")):
            continue
        pools.setdefault(_pool_key(row), []).append(row)

    records: list[dict[str, Any]] = []
    for (model_key, budget, conversation_id, target_turn), pool in pools.items():
        for harm_name, harm_column in HARM_TARGETS.items():
            if harm_name == "answer_harm":
                scored = [
                    row
                    for row in pool
                    if _is_true(row.get("has_behavior_label"))
                    and np.isfinite(_to_float(row.get(harm_column)))
                ]
            else:
                scored = [row for row in pool if np.isfinite(_to_float(row.get(harm_column)))]
            if len(scored) < MIN_POOL_SIZE:
                continue
            oracle = np.asarray([_to_float(row[harm_column]) for row in scored], dtype=np.float64)
            if np.allclose(oracle, oracle[0]):
                continue
            record: dict[str, Any] = {
                "model_key": model_key,
                "budget_fraction": budget,
                "conversation_id": conversation_id,
                "target_turn": target_turn,
                "harm_target": harm_name,
                "view": view,
                "pool_size": len(scored),
            }
            for feature in CANDIDATE_FEATURES:
                predicted = np.asarray(
                    [_to_float(row.get(feature)) for row in scored], dtype=np.float64
                )
                if not np.all(np.isfinite(predicted)):
                    record[f"tau__{feature}"] = float("nan")
                    record[f"top5__{feature}"] = float("nan")
                    continue
                record[f"tau__{feature}"] = kendall_tau(predicted, oracle)
                record[f"top5__{feature}"] = topk_recall(predicted, oracle, k=5)
            records.append(record)
    return records


def aggregate_view(
    records: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Aggregate per (harm target, budget): per-feature means and deltas vs semantic."""
    out: dict[str, Any] = {}
    harm_targets = sorted({record["harm_target"] for record in records})
    for harm_name in harm_targets:
        out[harm_name] = {}
        harm_records = [record for record in records if record["harm_target"] == harm_name]
        budgets = sorted({record["budget_fraction"] for record in harm_records})
        for budget in budgets:
            budget_records = [
                record for record in harm_records if record["budget_fraction"] == budget
            ]
            budget_out: dict[str, Any] = {"num_pools": len(budget_records)}
            for feature in CANDIDATE_FEATURES:
                taus = [
                    record[f"tau__{feature}"]
                    for record in budget_records
                    if np.isfinite(record.get(f"tau__{feature}", float("nan")))
                ]
                top5s = [
                    record[f"top5__{feature}"]
                    for record in budget_records
                    if np.isfinite(record.get(f"top5__{feature}", float("nan")))
                ]
                feature_out: dict[str, Any] = {
                    "kendall_tau": bootstrap_mean_ci(taus, rng=rng) if taus else None,
                    "top5_recall": bootstrap_mean_ci(top5s, rng=rng) if top5s else None,
                }
                if feature != BASELINE_FEATURE:
                    paired = [
                        {
                            "conversation_id": record["conversation_id"],
                            "delta_tau": record[f"tau__{feature}"]
                            - record[f"tau__{BASELINE_FEATURE}"],
                            "delta_top5": record[f"top5__{feature}"]
                            - record[f"top5__{BASELINE_FEATURE}"],
                        }
                        for record in budget_records
                        if np.isfinite(record.get(f"tau__{feature}", float("nan")))
                        and np.isfinite(record.get(f"tau__{BASELINE_FEATURE}", float("nan")))
                    ]
                    if paired:
                        row_tau = np.asarray(
                            [item["delta_tau"] for item in paired], dtype=np.float64
                        )
                        row_top5 = np.asarray(
                            [item["delta_top5"] for item in paired], dtype=np.float64
                        )
                        conv_rows = collapse_rows_by_keys(
                            paired,
                            metric_keys=["delta_tau", "delta_top5"],
                            group_keys=["conversation_id"],
                        )
                        conv_tau = np.asarray(
                            [item["delta_tau"] for item in conv_rows], dtype=np.float64
                        )
                        feature_out["vs_semantic"] = {
                            "delta_tau": {
                                "row_level": {
                                    **bootstrap_mean_ci(row_tau.tolist(), rng=rng),
                                    "p_value": paired_signflip_test(row_tau, rng=rng),
                                },
                                "conversation_level": {
                                    **bootstrap_mean_ci(conv_tau.tolist(), rng=rng),
                                    "p_value": paired_signflip_test(conv_tau, rng=rng),
                                },
                            },
                            "delta_top5_recall": {
                                "row_level": {
                                    **bootstrap_mean_ci(row_top5.tolist(), rng=rng),
                                    "p_value": paired_signflip_test(row_top5, rng=rng),
                                },
                            },
                        }
                budget_out[feature] = feature_out
            out[harm_name][budget] = budget_out
    return out


def gate_sensitivity(aggregated: dict[str, Any]) -> dict[str, Any]:
    """For each harm target × threshold: does Gate 1 still pass anywhere?"""
    sensitivity: dict[str, Any] = {}
    for harm_name, budgets in aggregated.items():
        sensitivity[harm_name] = {}
        for threshold in GATE_TAU_THRESHOLDS:
            hits = []
            for budget, features in budgets.items():
                for feature, payload in features.items():
                    if feature in ("num_pools", BASELINE_FEATURE) or not isinstance(
                        payload, dict
                    ):
                        continue
                    vs = payload.get("vs_semantic", {})
                    delta_tau = vs.get("delta_tau", {}).get("row_level", {}).get("mean")
                    delta_top5 = (
                        vs.get("delta_top5_recall", {}).get("row_level", {}).get("mean")
                    )
                    if delta_tau is None:
                        continue
                    if delta_tau >= threshold or (
                        delta_top5 is not None and delta_top5 >= GATE_RECALL_THRESHOLD
                    ):
                        hits.append(
                            {
                                "budget_fraction": budget,
                                "feature": feature,
                                "delta_tau": delta_tau,
                                "delta_top5_recall": delta_top5,
                            }
                        )
            sensitivity[harm_name][f"tau>={threshold:.2f}"] = {
                "passed": bool(hits),
                "hits": hits,
            }
    return sensitivity


def write_report(
    output_dir: Path,
    *,
    benchmark_name: str,
    views: dict[str, dict[str, Any]],
    sensitivity: dict[str, dict[str, Any]],
) -> None:
    lines = [
        f"# Answer-Harm Gate 1 Reanalysis: {benchmark_name}",
        "",
        "Side-by-side Gate 1 ranking value under logit-defined vs answer-NLL-defined",
        "oracle harm. The circularity concern is resolved in favor of the paper only",
        "if the geometry/hybrid delta-tau survives under **answer_harm**.",
        "",
    ]
    for view_name, aggregated in views.items():
        lines.append(f"## View: {view_name}\n")
        for harm_name, budgets in aggregated.items():
            lines.append(f"### Harm target: {harm_name}\n")
            lines.append("| budget | feature | mean tau | Δtau vs semantic (row p / conv p) | Δtop5 |")
            lines.append("|---|---|---|---|---|")
            for budget in sorted(budgets):
                features = budgets[budget]
                for feature in CANDIDATE_FEATURES:
                    payload = features.get(feature)
                    if not isinstance(payload, dict):
                        continue
                    tau = (payload.get("kendall_tau") or {}).get("mean", float("nan"))
                    vs = payload.get("vs_semantic", {})
                    delta = vs.get("delta_tau", {})
                    row = delta.get("row_level", {})
                    conv = delta.get("conversation_level", {})
                    top5 = vs.get("delta_top5_recall", {}).get("row_level", {})
                    if feature == BASELINE_FEATURE:
                        delta_text = "baseline"
                        top5_text = "—"
                    elif row:
                        delta_text = (
                            f"{row.get('mean', float('nan')):+.4f} "
                            f"(p={row.get('p_value', float('nan')):.4f} / "
                            f"p={conv.get('p_value', float('nan')):.4f})"
                        )
                        top5_text = f"{top5.get('mean', float('nan')):+.4f}"
                    else:
                        delta_text = "—"
                        top5_text = "—"
                    lines.append(
                        f"| {budget} | {feature} | {tau:.4f} | {delta_text} | {top5_text} |"
                    )
            lines.append("")
    lines.append("## Gate threshold sensitivity (overall view)\n")
    lines.append(
        "Runbook requirement: the +0.03 tau cutoff is heuristic and must be shown "
        "stable under perturbation.\n"
    )
    for harm_name, thresholds in sensitivity.items():
        lines.append(f"### {harm_name}\n")
        lines.append("| threshold | passes | passing features |")
        lines.append("|---|---|---|")
        for threshold_name, payload in thresholds.items():
            features = sorted({hit["feature"] for hit in payload["hits"]})
            lines.append(
                f"| {threshold_name} | {payload['passed']} | {', '.join(features) or '—'} |"
            )
        lines.append("")
    (output_dir / "answer_harm_gate1_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-rows", type=Path, required=True)
    parser.add_argument("--benchmark-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = load_candidate_rows(args.candidate_rows, args.benchmark_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    views: dict[str, dict[str, Any]] = {}
    pool_counts: dict[str, int] = {}
    for view in ("overall", "semantic_shortlist"):
        records = compute_pool_metrics(rows, view=view)
        pool_counts[view] = len(records)
        views[view] = aggregate_view(records, rng=rng)
    sensitivity = gate_sensitivity(views["overall"])

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_rows": str(args.candidate_rows),
        "benchmark_name": args.benchmark_name,
        "num_candidate_rows": len(rows),
        "num_pool_records": pool_counts,
        "baseline_feature": BASELINE_FEATURE,
        "harm_targets": HARM_TARGETS,
        "views": views,
        "gate_sensitivity": sensitivity,
    }
    (args.output_dir / "answer_harm_gate1_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    write_report(
        args.output_dir,
        benchmark_name=args.benchmark_name or args.candidate_rows.parent.name,
        views=views,
        sensitivity=sensitivity,
    )
    print(
        f"[answer_harm_gate1] {len(rows)} rows -> {pool_counts} pool records -> "
        f"{args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
