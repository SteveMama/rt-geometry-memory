from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .run_paper3 import DEFAULT_INPUT, DEFAULT_OUTPUT, _parse_families, _parse_float_list, run_codec_pilot


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _study_summary(model_results: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for result in model_results:
        summary = result["summary"]
        payload[summary["model_key"]] = {
            "model_name": summary["model_name"],
            "num_conversations": summary["num_conversations"],
            "num_evaluations": summary["num_evaluations"],
            "segment_span": summary["segment_span"],
            "aggregate": summary["aggregate"],
            "improvement_vs_uniform": summary["improvement_vs_uniform"],
        }
    return payload


def _format_report(summary: dict[str, Any]) -> str:
    lines = [
        f"# Paper 3 Study: {summary['study_name']}",
        "",
        f"- Created: {summary['created_at']}",
        f"- Models: {', '.join(summary['model_keys'])}",
        f"- Families: {', '.join(summary['families']) if summary['families'] else 'all'}",
        f"- Budgets: {', '.join(f'{float(item):.2f}' for item in summary['budgets'])}",
        "",
    ]
    for model_key, payload in summary["models"].items():
        lines.extend(
            [
                f"## {model_key}",
                "",
                f"- Model name: `{payload['model_name']}`",
                f"- Conversations: {payload['num_conversations']}",
                f"- Evaluations: {payload['num_evaluations']}",
                f"- Segment span: {payload['segment_span']}",
                "",
            ]
        )
        for budget_key, budget_payload in payload["improvement_vs_uniform"].items():
            lines.append(f"- Improvement vs uniform @ {budget_key}:")
            for policy_name, metrics in budget_payload.items():
                lines.append(
                    f"  {policy_name}: delta logit L2 {metrics['delta_logit_l2']:.3f}, "
                    f"relative logit L2 {metrics['relative_logit_l2']:.3f}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Paper 3 pilot across multiple models.")
    parser.add_argument("--study-name", default="paper3_study_v1")
    parser.add_argument("--model-keys", default="qwen25_05b,qwen25_15b,smollm2_17b")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--extra-input-paths", default=None)
    parser.add_argument("--families", default="long_dependency,retrieval_heavy,code_conversation")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT / "studies")
    parser.add_argument("--budgets", default="0.20,0.35,0.50")
    parser.add_argument("--recent-window", type=int, default=2)
    parser.add_argument("--min-history", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--state-layer", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--segment-span", type=int, default=2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_paths = [args.input_path]
    if args.extra_input_paths:
        input_paths.extend(Path(item.strip()) for item in args.extra_input_paths.split(",") if item.strip())
    model_keys = [item.strip() for item in args.model_keys.split(",") if item.strip()]
    study_dir = args.output_root / args.study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    model_results: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for model_key in model_keys:
        result = run_codec_pilot(
            model_key=model_key,
            input_paths=input_paths,
            families=_parse_families(args.families),
            budgets=_parse_float_list(args.budgets),
            recent_window=args.recent_window,
            min_history=args.min_history,
            max_input_tokens=args.max_input_tokens,
            dtype=args.dtype,
            state_layer=args.state_layer,
            device=args.device,
            limit_conversations=args.limit_conversations,
            output_dir=study_dir / model_key,
            segment_span=args.segment_span,
        )
        model_results.append(result)
        combined_rows.extend(result["rows"])

    summary = {
        "study_name": args.study_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_keys": model_keys,
        "families": _parse_families(args.families),
        "budgets": _parse_float_list(args.budgets),
        "models": _study_summary(model_results),
    }

    _write_csv(study_dir / "evaluation_rows.csv", combined_rows)
    (study_dir / "study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (study_dir / "study_report.md").write_text(_format_report(summary), encoding="utf-8")
    print(f"Wrote Paper 3 study outputs to {study_dir}")


if __name__ == "__main__":
    main()
