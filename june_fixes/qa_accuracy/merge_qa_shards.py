"""Merge sharded QA accuracy outputs and recompute summaries.

Mirrors paper3_codec.merge_study_shards: each shard directory must contain
qa_rows.csv (written by qa_accuracy_study.py with --conversation-ids-path).

    python -m june_fixes.qa_accuracy.merge_qa_shards \
      --study-name qa_msc_valid_merged \
      --output-root results/june_fixes/qa_accuracy \
      --shard-dirs dirA,dirB,dirC
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from june_fixes.qa_accuracy.qa_accuracy_study import (
    _write_csv,
    _write_json,
    summarize_qa_rows,
    write_qa_report,
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-dirs", type=str, required=True)
    args = parser.parse_args()

    shard_dirs = [Path(item) for item in args.shard_dirs.split(",") if item]
    combined: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        rows_path = shard_dir / "qa_rows.csv"
        if not rows_path.exists():
            print(f"[merge_qa_shards] WARNING: missing {rows_path}, skipping", flush=True)
            continue
        combined.extend(_read_rows(rows_path))
    if not combined:
        raise SystemExit("no qa_rows.csv found in any shard directory")

    output_dir = args.output_root / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, significance = summarize_qa_rows(combined)
    _write_csv(output_dir / "qa_rows.csv", combined)
    _write_json(
        output_dir / "qa_summary.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "merged_from": [str(item) for item in shard_dirs],
            "num_rows": len(combined),
            "num_conversations": len({row["conversation_id"] for row in combined}),
            "qa_summary": summary,
        },
    )
    _write_json(output_dir / "qa_significance_summary.json", significance)
    write_qa_report(
        output_dir,
        summary=summary,
        significance=significance,
        num_rows=len(combined),
        study_name=args.study_name,
    )
    _write_json(
        output_dir / "merge_progress.json",
        {"status": "complete", "updated_at": datetime.now(timezone.utc).isoformat()},
    )
    print(f"[merge_qa_shards] merged {len(combined)} rows into {output_dir}", flush=True)


if __name__ == "__main__":
    main()
