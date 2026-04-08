from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .harm_oracle_study import (
    _apply_harm_scalar,
    _format_report,
    _gate_summary,
    _oracle_topk_summary,
    _ranking_summary,
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge sharded oracle harm study outputs.")
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-dirs", required=True, help="Comma-separated shard output directories.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    shard_dirs = [Path(item.strip()) for item in args.shard_dirs.split(",") if item.strip()]
    if not shard_dirs:
        raise RuntimeError("No shard directories provided.")

    candidate_rows: list[dict[str, Any]] = []
    model_payloads: dict[str, Any] = {}
    budgets: list[float] = []
    families: list[str] | None = None
    model_keys: list[str] = []
    conversation_ids: set[str] = set()

    for shard_dir in shard_dirs:
        candidate_csv = shard_dir / "candidate_rows.csv"
        summary_json = shard_dir / "summary.json"
        if not candidate_csv.exists():
            raise RuntimeError(f"Missing oracle shard CSV: {candidate_csv}")
        if not summary_json.exists():
            raise RuntimeError(f"Missing oracle shard summary: {summary_json}")
        shard_rows = _read_csv(candidate_csv)
        candidate_rows.extend(shard_rows)
        for row in shard_rows:
            conversation_ids.add(str(row["conversation_id"]))
        shard_summary = json.loads(summary_json.read_text(encoding="utf-8"))
        model_payloads.update(shard_summary.get("models", {}))
        if not budgets:
            budgets = [float(item) for item in shard_summary.get("budgets", [])]
        if families is None:
            families = shard_summary.get("families")
        if not model_keys:
            model_keys = [str(item) for item in shard_summary.get("model_keys", [])]

    _apply_harm_scalar(candidate_rows)
    ranking_summary = _ranking_summary(candidate_rows)
    oracle_topk_summary = _oracle_topk_summary(candidate_rows)
    gate_summary = _gate_summary(ranking_summary)

    output_dir = args.output_root / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "study_name": args.study_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_name": args.benchmark_name,
        "model_keys": model_keys,
        "budgets": budgets,
        "families": families,
        "num_conversations": len(conversation_ids),
        "num_candidate_rows": len(candidate_rows),
        "models": model_payloads,
        "ranking_summary": ranking_summary,
        "oracle_topk_summary": oracle_topk_summary,
        "gate_summary": gate_summary,
    }

    _write_csv(output_dir / "candidate_rows.csv", candidate_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_format_report(summary), encoding="utf-8")
    print(f"Wrote merged oracle study to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
