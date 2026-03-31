#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PAPERS = ROOT / "papers"
PAPERS_GENERATED = PAPERS / "generated"
PAPER3_TRACKED = ROOT / "paper3"
RESULTS_TRACKED = ROOT / "results"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _parse_memory_critical_counts(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    patterns = {
        "cases_compared": r"Cases compared: (\d+)",
        "geometry_better": r"Geometry retains more support user turns than uniform: (\d+) cases",
        "uniform_better": r"Uniform retains more support user turns than geometry: (\d+) cases",
        "geometry_latest_support": r"Geometry keeps the latest support user turn while uniform drops it: (\d+) cases",
    }
    out: dict[str, int] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Missing {key} in {path}")
        out[key] = int(match.group(1))
    return out


def _experiment_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    p1_conf = _load_json(ARTIFACTS / "paper1" / "expanded_v8_final" / "confidence_summary.json")
    p1_study = _load_json(ARTIFACTS / "paper1" / "expanded_v8_final" / "study_summary.json")
    q15 = p1_conf["qwen25_15b"]
    rows.append(
        {
            "phase": "Phase 1",
            "paper": "Paper 1",
            "artifact": _path(ARTIFACTS / "paper1" / "expanded_v8_final"),
            "benchmark_or_family": "multi-family characterization study",
            "model": "qwen25_15b",
            "policy": "trajectory characterization",
            "budget": "NA",
            "primary_metric": "rank95 / corr(d_geo, delta_logit)",
            "key_result_summary": (
                f"rank95={_fmt(q15['mean_rank95']['estimate'])}; "
                f"corr={_fmt(q15['mean_corr_geodesic_vs_logit_l2']['estimate'])}; "
                f"shuffled_rank95={_fmt(p1_study['null_controls']['qwen25_15b']['h1_shuffled_turn_order']['shuffled_mean_rank95'])}"
            ),
            "status": "supporting",
        }
    )

    p2_sig = _load_json(ARTIFACTS / "paper2" / "behavior_stress_v1" / "significance_summary.json")
    p2_mech_counts = _parse_memory_critical_counts(
        ARTIFACTS / "paper2" / "behavior_stress_qwen_cases" / "memory_critical_qwen25_05b_b035.md"
    )
    rows.append(
        {
            "phase": "Phase 2",
            "paper": "Paper 2",
            "artifact": _path(ARTIFACTS / "paper2" / "behavior_stress_v1"),
            "benchmark_or_family": "hard stress set (all families)",
            "model": "qwen25_05b",
            "policy": "geometry",
            "budget": "0.35",
            "primary_metric": "delta_logit_l2 vs uniform",
            "key_result_summary": (
                f"delta_logit={_fmt(p2_sig['qwen25_05b']['0.35']['geometry']['delta_logit_l2']['mean'])}; "
                f"p={_fmt(p2_sig['qwen25_05b']['0.35']['geometry']['delta_logit_l2']['p_value'], 4)}"
            ),
            "status": "win",
        }
    )
    rows.append(
        {
            "phase": "Phase 2",
            "paper": "Paper 2",
            "artifact": _path(ARTIFACTS / "paper2" / "behavior_stress_qwen_cases"),
            "benchmark_or_family": "hard stress set mechanism readout",
            "model": "qwen25_05b",
            "policy": "geometry",
            "budget": "0.35",
            "primary_metric": "support-turn rescue",
            "key_result_summary": (
                f"geometry_better={p2_mech_counts['geometry_better']}/{p2_mech_counts['cases_compared']}; "
                f"uniform_better={p2_mech_counts['uniform_better']}/{p2_mech_counts['cases_compared']}; "
                f"latest_support_rescues={p2_mech_counts['geometry_latest_support']}"
            ),
            "status": "supporting",
        }
    )

    p3_pilot = _load_json(ARTIFACTS / "paper3" / "paper3_pilot_v3_full" / "significance_summary.json")
    rows.append(
        {
            "phase": "Phase 3",
            "paper": "Paper 3",
            "artifact": _path(ARTIFACTS / "paper3" / "paper3_pilot_v3_full"),
            "benchmark_or_family": "hard stress set pilot",
            "model": "qwen25_05b",
            "policy": "geometry_keep_compress_drop",
            "budget": "0.35",
            "primary_metric": "delta_logit_l2 vs uniform",
            "key_result_summary": (
                f"delta_logit={_fmt(p3_pilot['qwen25_05b']['0.35']['geometry_keep_compress_drop']['delta_logit_l2']['mean'])}; "
                f"p={_fmt(p3_pilot['qwen25_05b']['0.35']['geometry_keep_compress_drop']['delta_logit_l2']['p_value'], 4)}"
            ),
            "status": "win",
        }
    )

    p3_fair = _load_json(ARTIFACTS / "paper3" / "paper3_batch_v1_fairness" / "significance_summary.json")
    rows.append(
        {
            "phase": "Phase 4",
            "paper": "Paper 3",
            "artifact": _path(ARTIFACTS / "paper3" / "paper3_batch_v1_fairness"),
            "benchmark_or_family": "hard stress set fairness sweep",
            "model": "qwen25_15b",
            "policy": "geometry_keep_compress_drop",
            "budget": "0.32",
            "primary_metric": "delta_logit_l2 vs uniform",
            "key_result_summary": (
                f"delta_logit={_fmt(p3_fair['qwen25_15b']['0.32']['geometry_keep_compress_drop']['delta_logit_l2']['mean'])}; "
                f"p={_fmt(p3_fair['qwen25_15b']['0.32']['geometry_keep_compress_drop']['delta_logit_l2']['p_value'], 4)}"
            ),
            "status": "win",
        }
    )

    p3_3b_sig = _load_json(ARTIFACTS / "paper3" / "paper3_batch_v1_3b" / "significance_summary.json")
    p3_3b_pair = _load_json(ARTIFACTS / "paper3" / "paper3_batch_v1_3b" / "pairwise_summary.json")
    rows.append(
        {
            "phase": "Phase 4",
            "paper": "Paper 3",
            "artifact": _path(ARTIFACTS / "paper3" / "paper3_batch_v1_3b"),
            "benchmark_or_family": "hard stress set 3B probe",
            "model": "qwen25_3b",
            "policy": "geometry_keep_compress_drop vs geometry",
            "budget": "0.35 / 0.50",
            "primary_metric": "pairwise delta_logit_l2",
            "key_result_summary": (
                f"0.35 pairwise={_fmt(p3_3b_pair['logit_pairwise']['qwen25_3b']['0.35']['geometry_keep_compress_drop__vs__geometry']['delta_logit_l2']['mean'])}; "
                f"0.50 geometry={_fmt(p3_3b_sig['qwen25_3b']['0.50']['geometry']['delta_logit_l2']['mean'])}"
            ),
            "status": "mixed",
        }
    )

    p3_public = _load_json(PAPER3_TRACKED / "paper3_public_v1_public_benchmark" / "significance_summary.json")
    rows.append(
        {
            "phase": "Phase 6",
            "paper": "Paper 3",
            "artifact": _path(PAPER3_TRACKED / "paper3_public_v1_public_benchmark"),
            "benchmark_or_family": "LongMemEval public benchmark",
            "model": "qwen25_15b",
            "policy": "semantic / geometry_segment_actions / geometry_keep_compress_drop",
            "budget": "0.20 / 0.35 / 0.50",
            "primary_metric": "delta_logit_l2 vs uniform",
            "key_result_summary": (
                f"0.20 semantic={_fmt(p3_public['qwen25_15b']['0.20']['semantic']['delta_logit_l2']['mean'])}; "
                f"0.35 segment={_fmt(p3_public['qwen25_15b']['0.35']['geometry_segment_actions']['delta_logit_l2']['mean'])}; "
                f"0.50 KCD={_fmt(p3_public['qwen25_15b']['0.50']['geometry_keep_compress_drop']['delta_logit_l2']['mean'])}"
            ),
            "status": "mixed",
        }
    )

    rows.append(
        {
            "phase": "Phase 8",
            "paper": "Paper 3",
            "artifact": _path(PAPERS / "paper3_msc_semantic_codec_checkpoint.md"),
            "benchmark_or_family": "MSC semantic-memory benchmark",
            "model": "qwen25_15b",
            "policy": "semantic vs semantic_keep_compress_drop",
            "budget": "0.20 / 0.35 / 0.50",
            "primary_metric": "delta_logit_l2 + paired behavior",
            "key_result_summary": (
                "0.20 semantic-KCD competitive but not significant; "
                "0.35 and 0.50 plain semantic stronger, especially on behavior"
            ),
            "status": "mixed",
        }
    )

    low_msc = _load_json(RESULTS_TRACKED / "paper3" / "studies" / "paper3_low_budget_smoke_msc" / "significance_summary.json")
    rows.append(
        {
            "phase": "Phase 9",
            "paper": "Paper 3",
            "artifact": _path(RESULTS_TRACKED / "paper3" / "studies" / "paper3_low_budget_smoke_msc"),
            "benchmark_or_family": "MSC low-budget smoke",
            "model": "qwen25_05b",
            "policy": "support_aware_geometry_keep_compress_drop",
            "budget": "0.20",
            "primary_metric": "delta_logit_l2 vs uniform",
            "key_result_summary": (
                f"support_aware={_fmt(low_msc['qwen25_05b']['0.20']['support_aware_geometry_keep_compress_drop']['delta_logit_l2']['mean'])}; "
                f"old_KCD={_fmt(low_msc['qwen25_05b']['0.20']['geometry_keep_compress_drop']['delta_logit_l2']['mean'])}"
            ),
            "status": "supporting",
        }
    )

    semopt_msc = _load_json(RESULTS_TRACKED / "paper3" / "studies" / "paper3_semantic_kcd_opt_smoke_msc" / "significance_summary.json")
    rows.append(
        {
            "phase": "Phase 10",
            "paper": "Paper 3",
            "artifact": _path(RESULTS_TRACKED / "paper3" / "studies" / "paper3_semantic_kcd_opt_smoke_msc"),
            "benchmark_or_family": "MSC semantic-KCD optimization smoke",
            "model": "qwen25_05b",
            "policy": "semantic family variants",
            "budget": "0.35",
            "primary_metric": "delta_logit_l2 vs uniform",
            "key_result_summary": (
                f"semantic={_fmt(semopt_msc['qwen25_05b']['0.35']['semantic']['delta_logit_l2']['mean'])}; "
                f"semantic_KCD={_fmt(semopt_msc['qwen25_05b']['0.35']['semantic_keep_compress_drop']['delta_logit_l2']['mean'])}; "
                f"support_aware_semantic={_fmt(semopt_msc['qwen25_05b']['0.35']['support_aware_semantic_keep_compress_drop']['delta_logit_l2']['mean'])}"
            ),
            "status": "mixed",
        }
    )

    qgeom_msc = _load_json(RESULTS_TRACKED / "paper3" / "studies" / "paper3_query_geom_smoke_msc" / "significance_summary.json")
    rows.append(
        {
            "phase": "Phase 10",
            "paper": "Paper 3",
            "artifact": _path(RESULTS_TRACKED / "paper3" / "studies" / "paper3_query_geom_smoke_msc"),
            "benchmark_or_family": "MSC query-conditioned geometry smoke",
            "model": "qwen25_05b",
            "policy": "query_conditioned_geometry_keep_compress_drop",
            "budget": "0.20",
            "primary_metric": "delta_logit_l2 vs uniform",
            "key_result_summary": (
                f"query_geom_KCD={_fmt(qgeom_msc['qwen25_05b']['0.20']['query_conditioned_geometry_keep_compress_drop']['delta_logit_l2']['mean'])}; "
                f"geometry={_fmt(qgeom_msc['qwen25_05b']['0.20']['geometry']['delta_logit_l2']['mean'])}"
            ),
            "status": "supporting",
        }
    )

    atlas = _load_json(ARTIFACTS / "paper1" / "regime_atlas_smoke_v4" / "atlas_summary.json")
    retrieval_rows = [row for row in atlas["family_regime_summary"] if row["family"] == "retrieval_heavy"]
    retrieval_summary = "; ".join(f"regime {row['regime_id']}={row['count']}" for row in retrieval_rows)
    rows.append(
        {
            "phase": "Phase 11-12",
            "paper": "Paper 1 / Paper 3 bridge",
            "artifact": _path(ARTIFACTS / "paper1" / "regime_atlas_smoke_v4"),
            "benchmark_or_family": "geometry-only regime atlas",
            "model": "qwen25_05b",
            "policy": "segment curvature regime clustering",
            "budget": "NA",
            "primary_metric": "family / regime separation",
            "key_result_summary": f"num_segments={atlas['num_segments']}; retrieval_heavy split across {retrieval_summary}",
            "status": "falsified",
        }
    )

    persona = _load_json(ARTIFACTS / "paper3" / "msc_persona_curvature_v1" / "summary.json")
    persona_ag = persona["aggregate"]
    rows.append(
        {
            "phase": "Phase 13",
            "paper": "Paper 3",
            "artifact": _path(ARTIFACTS / "paper3" / "msc_persona_curvature_v1"),
            "benchmark_or_family": "MSC support/persona vs filler",
            "model": "qwen25_05b",
            "policy": "stabilized curvature separation",
            "budget": "NA",
            "primary_metric": "support-filler delta_kappa",
            "key_result_summary": (
                f"delta_kappa={_fmt(persona_ag['mean_delta_curvature'])}; "
                f"positive={persona_ag['positive_delta_count']}/5; negative={persona_ag['negative_delta_count']}/5"
            ),
            "status": "falsified",
        }
    )

    state_align = _load_json(ARTIFACTS / "paper3" / "state_update_alignment_smoke_qwen05b" / "summary.json")
    state_align_ag = state_align["aggregate"]
    rows.append(
        {
            "phase": "Phase 14",
            "paper": "Paper 3",
            "artifact": _path(ARTIFACTS / "paper3" / "state_update_alignment_smoke_qwen05b"),
            "benchmark_or_family": "synthetic state-update benchmark",
            "model": "qwen25_05b",
            "policy": "same-sign directional alignment",
            "budget": "NA",
            "primary_metric": "negative alignment count",
            "key_result_summary": (
                f"mean_alignment={_fmt(state_align_ag['mean_directional_alignment'])}; "
                f"negative={state_align_ag['negative_alignment_count']}/10; "
                f"semantic_gt_thresh={state_align_ag['semantic_gt_threshold_count']}/10"
            ),
            "status": "falsified",
        }
    )

    return rows


def _negative_result_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    persona = _load_json(ARTIFACTS / "paper3" / "msc_persona_curvature_v1" / "summary.json")
    persona_ag = persona["aggregate"]
    rows.append(
        {
            "hypothesis_name": "MSC support turns separate from filler by curvature",
            "signal_or_formula": "stabilized curvature delta between labeled support/persona and filler turns",
            "dataset": "MSC manual 5-conversation check",
            "result_summary": (
                f"mean_delta={_fmt(persona_ag['mean_delta_curvature'])}; "
                f"positive={persona_ag['positive_delta_count']}/5; negative={persona_ag['negative_delta_count']}/5"
            ),
            "failure_type": "failed data fit",
            "consequence": "Do not use curvature alone as a within-topic support detector on MSC.",
        }
    )

    atlas = _load_json(ARTIFACTS / "paper1" / "regime_atlas_smoke_v4" / "atlas_summary.json")
    retrieval_rows = [row for row in atlas["family_regime_summary"] if row["family"] == "retrieval_heavy"]
    retrieval_summary = "; ".join(f"regime {row['regime_id']}={row['count']}" for row in retrieval_rows)
    rows.append(
        {
            "hypothesis_name": "Geometry-only regime atlas separates benchmark memory types",
            "signal_or_formula": "segment-level curvature / step-norm / rank clustering after stabilization",
            "dataset": "MSC + LoCoMo + LongMemEval + hard stress smoke atlas",
            "result_summary": f"retrieval_heavy remained mixed across {retrieval_summary}",
            "failure_type": "failed benchmark transfer",
            "consequence": "Do not rely on geometry-only regime classification as a standalone policy selector.",
        }
    )

    state_align = _load_json(ARTIFACTS / "paper3" / "state_update_alignment_smoke_qwen05b" / "summary.json")
    state_align_ag = state_align["aggregate"]
    rows.append(
        {
            "hypothesis_name": "Same-sign increment alignment detects state supersession",
            "signal_or_formula": "A(s,t)=cos(u_s,u_t)",
            "dataset": "synthetic state-update benchmark",
            "result_summary": (
                f"mean_alignment={_fmt(state_align_ag['mean_directional_alignment'])}; "
                f"negative={state_align_ag['negative_alignment_count']}/10"
            ),
            "failure_type": "failed math",
            "consequence": "Retire the direct same-sign supersession rule in compact models.",
        }
    )

    state_cross = _load_json(ARTIFACTS / "paper3" / "state_update_cross_control_qwen05b" / "summary.json")
    cross_ag = state_cross["aggregate"]
    rows.append(
        {
            "hypothesis_name": "State/increment cross-term yields a negative update detector",
            "signal_or_formula": "cos(log_q(h_s), u_t^entry)",
            "dataset": "synthetic state-update benchmark",
            "result_summary": (
                f"update={_fmt(cross_ag['mean_state_update_entry_cross'])}; "
                f"control={_fmt(cross_ag['mean_control_entry_cross'])}; "
                f"negative_updates={cross_ag['negative_state_update_entry_cross_count']}/10"
            ),
            "failure_type": "failed math",
            "consequence": "Treat the cross-term as a weak ranking margin only, not a sign-based algorithm.",
        }
    )

    qgeom_locomo = _load_json(RESULTS_TRACKED / "paper3" / "studies" / "paper3_query_geom_smoke_locomo" / "significance_summary.json")
    rows.append(
        {
            "hypothesis_name": "Query-conditioned geometry is a stable standalone ranking signal",
            "signal_or_formula": "query-projected curvature + query-projected subspace energy",
            "dataset": "LoCoMo bounded smoke",
            "result_summary": (
                f"0.35 query_geom={_fmt(qgeom_locomo['qwen25_05b']['0.35']['query_conditioned_geometry']['delta_logit_l2']['mean'])}; "
                f"query_geom_KCD={_fmt(qgeom_locomo['qwen25_05b']['0.35']['query_conditioned_geometry_keep_compress_drop']['delta_logit_l2']['mean'])}"
            ),
            "failure_type": "failed benchmark transfer",
            "consequence": "Use query-conditioned geometry only as a refinement signal; do not promote it to a standalone policy.",
        }
    )

    return rows


def main() -> None:
    experiment_rows = _experiment_rows()
    negative_rows = _negative_result_rows()

    _write_csv(
        PAPERS_GENERATED / "rt_experiment_matrix.csv",
        experiment_rows,
        [
            "phase",
            "paper",
            "artifact",
            "benchmark_or_family",
            "model",
            "policy",
            "budget",
            "primary_metric",
            "key_result_summary",
            "status",
        ],
    )
    _write_csv(
        PAPERS_GENERATED / "rt_negative_result_matrix.csv",
        negative_rows,
        [
            "hypothesis_name",
            "signal_or_formula",
            "dataset",
            "result_summary",
            "failure_type",
            "consequence",
        ],
    )
    print(f"Wrote {PAPERS_GENERATED / 'rt_experiment_matrix.csv'}")
    print(f"Wrote {PAPERS_GENERATED / 'rt_negative_result_matrix.csv'}")


if __name__ == "__main__":
    main()
