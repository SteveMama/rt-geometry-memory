"""LLM judge evaluation for QA accuracy results.

Reads qa_rows.csv produced by june_fixes.qa_accuracy.qa_accuracy_study,
judges each (question, gold_answer, prediction) triple with two judges:
  1. Gemini 2.5 Flash  (primary — set GEMINI_API_KEY)
  2. Llama-3-70B-Instruct via Together.ai  (secondary — set TOGETHER_API_KEY)

Reports per-policy/budget accuracy from each judge and Cohen's kappa for
inter-judge agreement. Designed to run locally after downloading qa_rows.csv
from RunPod.

Usage:
    python scripts/run_llm_judge_eval.py \\
      --input results/reviewer_fixes/lme_qa/reviewer_fixes_lme_qa_merged/qa_rows.csv \\
      --output results/reviewer_fixes/lme_qa/llm_judge \\
      --judges gemini,llama

Required env vars:
    GEMINI_API_KEY     Google AI API key (for Gemini 2.5 Flash)
    TOGETHER_API_KEY   Together.ai API key (for Llama-3-70B-Instruct)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ── Judge prompt ───────────────────────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """\
You are a factual QA evaluator. Your task is to determine whether an AI assistant's response correctly answers a question, given a reference answer.

Rules:
- Answer "yes" if the AI response contains the key information from the reference answer, even if worded differently.
- Answer "no" if the AI response is missing critical information, is wrong, or says it doesn't know when the reference answer has a clear answer.
- Ignore stylistic differences; focus on factual correctness.
- Respond with ONLY "yes" or "no" on the first line, then a one-sentence explanation."""

JUDGE_USER_TEMPLATE = """\
Question: {question}
Reference Answer: {gold_answer}
AI Response: {prediction}

Does the AI response correctly answer the question? Reply "yes" or "no"."""


def _parse_judge_response(text: str) -> int | None:
    """Return 1 for yes, 0 for no, None if unparseable."""
    first_line = text.strip().split("\n")[0].strip().lower()
    if first_line.startswith("yes"):
        return 1
    if first_line.startswith("no"):
        return 0
    if "yes" in first_line[:10]:
        return 1
    if "no" in first_line[:10]:
        return 0
    return None


# ── Gemini judge ───────────────────────────────────────────────────────────────
def _call_gemini(question: str, gold: str, prediction: str, *, model: str, api_key: str) -> int | None:
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        raise SystemExit(
            "google-generativeai not installed. Run: pip install google-generativeai"
        )
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(
        model_name=model,
        system_instruction=JUDGE_SYSTEM_PROMPT,
    )
    prompt = JUDGE_USER_TEMPLATE.format(question=question, gold_answer=gold, prediction=prediction)
    response = client.generate_content(prompt)
    return _parse_judge_response(response.text)


# ── Together.ai / Llama judge ─────────────────────────────────────────────────
def _call_together(question: str, gold: str, prediction: str, *, model: str, api_key: str) -> int | None:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise SystemExit("openai not installed. Run: pip install openai")
    client = OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
    prompt = JUDGE_USER_TEMPLATE.format(question=question, gold_answer=gold, prediction=prediction)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=64,
        temperature=0.0,
    )
    return _parse_judge_response(response.choices[0].message.content or "")


# ── Cohen's kappa ─────────────────────────────────────────────────────────────
def cohen_kappa(a: list[int], b: list[int]) -> float:
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return float("nan")
    p_o = sum(x == y for x, y in zip(a, b)) / n
    p_yes_a = sum(a) / n
    p_yes_b = sum(b) / n
    p_e = p_yes_a * p_yes_b + (1 - p_yes_a) * (1 - p_yes_b)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="qa_rows.csv from qa_accuracy_study")
    parser.add_argument("--output", type=Path, required=True, help="output directory for judge results")
    parser.add_argument(
        "--judges",
        default="gemini,llama",
        help="comma-separated list of judges to use: gemini, llama (default: gemini,llama)",
    )
    parser.add_argument("--gemini-model", default="gemini-2.0-flash", help="Gemini model name")
    parser.add_argument(
        "--llama-model",
        default="meta-llama/Llama-3-70b-chat-hf",
        help="Together.ai model ID for the open-weight judge",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limit total rows evaluated (for testing)",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=1.0,
        help="seconds to sleep between retries on API errors",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="max retries per judge call on transient error",
    )
    args = parser.parse_args()

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    together_key = os.environ.get("TOGETHER_API_KEY", "")

    if "gemini" in judges and not gemini_key:
        raise SystemExit("Set GEMINI_API_KEY to use the Gemini judge")
    if "llama" in judges and not together_key:
        raise SystemExit("Set TOGETHER_API_KEY to use the Llama judge (Together.ai)")

    args.output.mkdir(parents=True, exist_ok=True)

    with args.input.open(encoding="utf-8") as fh:
        rows: list[dict[str, Any]] = list(csv.DictReader(fh))

    if args.limit:
        rows = rows[: args.limit]

    print(f"[judge_eval] Evaluating {len(rows)} rows with judges: {judges}", flush=True)

    judge_results: dict[str, list[dict[str, Any]]] = {judge: [] for judge in judges}

    for i, row in enumerate(rows):
        question = str(row.get("question", "")).strip()
        gold = str(row.get("gold_answer", "")).strip()
        prediction = str(row.get("prediction", "")).strip()

        if not question or not gold or not prediction:
            print(f"[judge_eval] row {i}: missing question/gold/prediction, skipping", flush=True)
            continue

        for judge in judges:
            for attempt in range(args.max_retries):
                try:
                    if judge == "gemini":
                        verdict = _call_gemini(
                            question, gold, prediction,
                            model=args.gemini_model, api_key=gemini_key,
                        )
                    elif judge == "llama":
                        verdict = _call_together(
                            question, gold, prediction,
                            model=args.llama_model, api_key=together_key,
                        )
                    else:
                        print(f"[judge_eval] unknown judge: {judge}", flush=True)
                        verdict = None
                    break
                except Exception as exc:
                    if attempt < args.max_retries - 1:
                        print(f"[judge_eval] {judge} attempt {attempt+1} error: {exc}, retrying...", flush=True)
                        time.sleep(args.retry_sleep * (2 ** attempt))
                    else:
                        print(f"[judge_eval] {judge} failed after {args.max_retries} attempts: {exc}", flush=True)
                        verdict = None

            judge_results[judge].append(
                {
                    "model_key": row.get("model_key", ""),
                    "conversation_id": row.get("conversation_id", ""),
                    "family": row.get("family", ""),
                    "target_turn": row.get("target_turn", ""),
                    "policy_name": row.get("policy_name", ""),
                    "budget_fraction": row.get("budget_fraction", ""),
                    "verdict": verdict,
                }
            )

        if (i + 1) % 50 == 0:
            print(f"[judge_eval] {i+1}/{len(rows)} rows done", flush=True)

    # ── Aggregate per-policy/budget accuracy ──────────────────────────────────
    report_lines = ["# LLM Judge Evaluation Report", "", f"Rows evaluated: {len(rows)}", ""]
    summary: dict[str, Any] = {}

    for judge in judges:
        results = judge_results[judge]
        report_lines.append(f"## Judge: {judge}")
        by_budget: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for r in results:
            if r["verdict"] is None:
                continue
            budget = str(r["budget_fraction"])
            policy = str(r["policy_name"])
            by_budget[budget][policy].append(int(r["verdict"]))
        summary[judge] = {}
        for budget in sorted(by_budget):
            report_lines.append(f"\n### Budget {budget}\n")
            report_lines.append("| policy | accuracy | n |")
            report_lines.append("|---|---|---|")
            summary[judge][budget] = {}
            for policy in sorted(by_budget[budget]):
                verdicts = by_budget[budget][policy]
                acc = sum(verdicts) / len(verdicts) if verdicts else float("nan")
                report_lines.append(f"| {policy} | {acc:.4f} | {len(verdicts)} |")
                summary[judge][budget][policy] = {"accuracy": acc, "n": len(verdicts)}
        report_lines.append("")

    # ── Inter-judge agreement ─────────────────────────────────────────────────
    if len(judges) == 2:
        j1, j2 = judges
        r1 = judge_results[j1]
        r2 = judge_results[j2]
        paired = [
            (a["verdict"], b["verdict"])
            for a, b in zip(r1, r2)
            if a["verdict"] is not None and b["verdict"] is not None
        ]
        if paired:
            a_verdicts = [p[0] for p in paired]
            b_verdicts = [p[1] for p in paired]
            kappa = cohen_kappa(a_verdicts, b_verdicts)
            agreement_pct = sum(x == y for x, y in paired) / len(paired)
            report_lines += [
                "## Inter-judge Agreement",
                "",
                f"- Judges: {j1} vs {j2}",
                f"- N pairs: {len(paired)}",
                f"- Agreement: {agreement_pct:.4f} ({agreement_pct*100:.1f}%)",
                f"- Cohen's κ: {kappa:.4f}",
                "",
            ]
            summary["inter_judge_agreement"] = {
                "judges": [j1, j2],
                "n_pairs": len(paired),
                "agreement": agreement_pct,
                "cohens_kappa": kappa,
            }

    report_text = "\n".join(report_lines)
    (args.output / "judge_report.md").write_text(report_text, encoding="utf-8")
    (args.output / "judge_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Save raw verdict rows per judge
    for judge in judges:
        out_csv = args.output / f"verdicts_{judge}.csv"
        rows_to_write = judge_results[judge]
        if rows_to_write:
            with out_csv.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows_to_write[0].keys()))
                writer.writeheader()
                writer.writerows(rows_to_write)

    print(f"[judge_eval] Report written to {args.output}/judge_report.md", flush=True)
    print(report_text)


if __name__ == "__main__":
    main()
