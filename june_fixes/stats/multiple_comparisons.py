"""Multiple-comparison correction sweep over all significance summaries.

Review fix #4: several load-bearing p-values sit at 0.045-0.079 across a grid
of 3 budgets x many policies x several metrics, with no correction. This
module walks the result trees, collects every p-value from
``significance_summary.json`` / ``behavior_significance_summary.json`` /
``qa_significance_summary.json`` files, and applies Benjamini-Hochberg (FDR)
and Holm (FWER) corrections within natural families
(source-study x metric x test level). It emits:

  - ``corrected_pvalues.csv``        every test with raw p, BH q, Holm p
  - ``corrections_summary.json``     machine-readable
  - ``corrections_report.md``        survives/dies table at q=0.05 and q=0.10

CPU only:

    python -m june_fixes.stats.multiple_comparisons \
      --search-roots results,artifacts,paper3_gate1_scaleup_multigpu_merged_results \
      --output-dir results/june_fixes/multiple_comparisons
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SUMMARY_FILENAMES = (
    "significance_summary.json",
    "behavior_significance_summary.json",
    "qa_significance_summary.json",
)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH adjusted q-values (monotone, step-up)."""
    n = p_values.size
    order = np.argsort(p_values)
    ranked = p_values[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    n = p_values.size
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * (n - np.arange(n))
    adjusted = np.maximum.accumulate(adjusted)
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _walk_tests(payload: Any, path: list[str], source: Path) -> list[dict[str, Any]]:
    """Recursively find dicts holding a p_value leaf; record their key path."""
    found: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        if "p_value" in payload and isinstance(payload["p_value"], (int, float)):
            found.append(
                {
                    "source_file": str(source),
                    "key_path": "/".join(path),
                    "p_value": float(payload["p_value"]),
                    "mean": float(payload.get("mean", float("nan"))),
                    "ci_low": float(payload.get("ci_low", float("nan"))),
                    "ci_high": float(payload.get("ci_high", float("nan"))),
                }
            )
        else:
            for key, value in payload.items():
                found.extend(_walk_tests(value, path + [str(key)], source))
    return found


def _family_key(record: dict[str, Any]) -> str:
    """Correction family: study directory + summary kind + test level."""
    source = Path(record["source_file"])
    level = "row_level" if record["key_path"].endswith("row_level") else (
        "conversation_level"
        if record["key_path"].endswith("conversation_level")
        else "other"
    )
    return f"{source.parent.name}|{source.name}|{level}"


def collect_records(search_roots: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            print(f"[multiple_comparisons] WARNING: missing root {root}", flush=True)
            continue
        for filename in SUMMARY_FILENAMES:
            for path in sorted(root.rglob(filename)):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    print(f"[multiple_comparisons] skipping {path}: {exc}", flush=True)
                    continue
                records.extend(_walk_tests(payload, [], path))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-roots",
        type=str,
        default="results,artifacts,paper3_gate1_scaleup_multigpu_merged_results,paper3",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    roots = [Path(item.strip()) for item in args.search_roots.split(",") if item.strip()]
    records = collect_records(roots)
    if not records:
        raise SystemExit("no significance summaries found under the given roots")

    families: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        families.setdefault(_family_key(record), []).append(index)

    for family, indices in families.items():
        p_values = np.asarray([records[i]["p_value"] for i in indices], dtype=np.float64)
        bh = benjamini_hochberg(p_values)
        holm = holm_bonferroni(p_values)
        for position, record_index in enumerate(indices):
            records[record_index]["family"] = family
            records[record_index]["family_size"] = len(indices)
            records[record_index]["bh_q"] = float(bh[position])
            records[record_index]["holm_p"] = float(holm[position])

    # Global correction across everything, as the most conservative view.
    all_p = np.asarray([record["p_value"] for record in records], dtype=np.float64)
    global_bh = benjamini_hochberg(all_p)
    for record, q in zip(records, global_bh):
        record["global_bh_q"] = float(q)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "key_path",
        "family",
        "family_size",
        "mean",
        "ci_low",
        "ci_high",
        "p_value",
        "bh_q",
        "holm_p",
        "global_bh_q",
    ]
    with (args.output_dir / "corrected_pvalues.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    raw_sig = [r for r in records if r["p_value"] < args.alpha]
    bh_sig = [r for r in records if r["bh_q"] < args.alpha]
    holm_sig = [r for r in records if r["holm_p"] < args.alpha]
    flipped = [r for r in raw_sig if r["bh_q"] >= args.alpha]

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "search_roots": [str(item) for item in roots],
        "alpha": args.alpha,
        "num_tests": len(records),
        "num_families": len(families),
        "raw_significant": len(raw_sig),
        "bh_significant": len(bh_sig),
        "holm_significant": len(holm_sig),
        "raw_significant_but_bh_killed": len(flipped),
    }
    (args.output_dir / "corrections_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Multiple-Comparison Correction Report",
        "",
        f"Tests collected: {len(records)} across {len(families)} families "
        f"(family = study x summary kind x test level).",
        "",
        f"| view | significant at alpha={args.alpha} |",
        "|---|---|",
        f"| raw p | {len(raw_sig)} |",
        f"| BH within family | {len(bh_sig)} |",
        f"| Holm within family | {len(holm_sig)} |",
        "",
        f"## Findings killed by BH correction ({len(flipped)})",
        "",
        "These are reported as significant in the manuscript pipeline but do not",
        "survive within-family FDR control. Each must be either re-run with more",
        "data, downgraded to 'directional', or dropped from the claims.",
        "",
        "| source | test | raw p | BH q | Holm p |",
        "|---|---|---|---|---|",
    ]
    for record in sorted(flipped, key=lambda item: item["p_value"]):
        lines.append(
            f"| {Path(record['source_file']).parent.name} | {record['key_path']} "
            f"| {record['p_value']:.4f} | {record['bh_q']:.4f} | {record['holm_p']:.4f} |"
        )
    lines += [
        "",
        "## Strongest surviving findings (BH q < 0.01)",
        "",
        "| source | test | raw p | BH q |",
        "|---|---|---|---|",
    ]
    for record in sorted(records, key=lambda item: item["bh_q"])[:40]:
        if record["bh_q"] >= 0.01:
            break
        lines.append(
            f"| {Path(record['source_file']).parent.name} | {record['key_path']} "
            f"| {record['p_value']:.4f} | {record['bh_q']:.4f} |"
        )
    (args.output_dir / "corrections_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"[multiple_comparisons] {len(records)} tests, raw {len(raw_sig)} -> "
        f"BH {len(bh_sig)} significant; {len(flipped)} killed. Output: {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
