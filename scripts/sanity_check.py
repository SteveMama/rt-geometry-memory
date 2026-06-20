#!/usr/bin/env python3
"""
sanity_check.py

Quick health check on partial experiment results.
Run this mid-experiment to catch degenerate geometry, codec failures,
or LongLLMLingua identity (no compression).

Usage:
    python scripts/sanity_check.py --results-root results/reviewer_fixes
    python scripts/sanity_check.py --results-root results/reviewer_fixes --verbose

Exit codes:
    0  all checks passed
    1  warnings only (experiment can continue)
    2  critical failure (stop and inspect)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WARN  = "\033[33m[WARN]\033[0m"
FAIL  = "\033[31m[FAIL]\033[0m"
OK    = "\033[32m[ OK ]\033[0m"
INFO  = "\033[36m[INFO]\033[0m"

issues = []
criticals = []


def warn(msg):
    print(f"{WARN} {msg}")
    issues.append(msg)


def fail(msg):
    print(f"{FAIL} {msg}")
    criticals.append(msg)


def ok(msg):
    print(f"{OK}  {msg}")


def info(msg):
    print(f"{INFO} {msg}")


# ── loaders ────────────────────────────────────────────────────────────────────

def load_csvs(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for p in sorted(root.rglob(name)):
        try:
            df = pd.read_csv(p)
            df["_src"] = str(p.relative_to(root))
            frames.append(df)
        except Exception as e:
            warn(f"Could not read {p}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── checks ─────────────────────────────────────────────────────────────────────

def check_row_count(eval_df: pd.DataFrame):
    n = len(eval_df)
    info(f"evaluation_rows total: {n:,}")
    if n == 0:
        fail("No evaluation rows at all — nothing has completed yet or CSV is empty")
    elif n < 20:
        warn(f"Only {n} evaluation rows — experiment is early, check again in a few minutes")
    else:
        ok(f"{n:,} evaluation rows present")


def check_policies(eval_df: pd.DataFrame, expected_policies: list):
    if eval_df.empty:
        return
    found = set(eval_df["policy_name"].unique())
    missing = [p for p in expected_policies if p not in found]
    if missing:
        warn(f"Expected policies not yet seen: {missing}")
    else:
        ok(f"All expected policies present: {sorted(found)}")
    for pol in found:
        n = (eval_df["policy_name"] == pol).sum()
        info(f"  {pol}: {n} rows")


def check_budget_adherence(eval_df: pd.DataFrame):
    if eval_df.empty or "token_fraction" not in eval_df.columns:
        return
    df = eval_df.dropna(subset=["budget_fraction", "token_fraction"])
    df["err"] = (df["token_fraction"] - df["budget_fraction"]).abs()
    bad = df[df["err"] > 0.20]
    pct = len(bad) / max(len(df), 1) * 100
    if pct > 30:
        fail(f"Budget adherence bad: {pct:.0f}% of rows deviate >20% from target — "
             f"codec may be broken")
    elif pct > 10:
        warn(f"Budget adherence loose: {pct:.0f}% of rows deviate >20% from target")
    else:
        mean_err = df["err"].mean()
        ok(f"Budget adherence ok — mean |err|={mean_err:.3f}, {pct:.0f}% > 20% threshold")


def check_score_distribution(eval_df: pd.DataFrame):
    if eval_df.empty or "top1_match" not in eval_df.columns:
        return
    scores = eval_df["top1_match"].dropna()
    if len(scores) == 0:
        warn("top1_match column empty")
        return
    mean_s = scores.mean()
    if scores.std() < 0.01:
        fail(f"top1_match has zero variance (std={scores.std():.4f}) — "
             f"all rows scoring identically, likely a bug")
    elif mean_s > 0.98:
        warn(f"top1_match mean={mean_s:.3f} suspiciously high — "
             f"check that compression is actually happening")
    elif mean_s < 0.02:
        fail(f"top1_match mean={mean_s:.3f} suspiciously low — "
             f"check that compressed prompts reach the model")
    else:
        ok(f"top1_match distribution looks healthy: mean={mean_s:.3f}, "
           f"std={scores.std():.3f}, min={scores.min():.2f}, max={scores.max():.2f}")


def check_kcd_actions(eval_df: pd.DataFrame):
    if eval_df.empty:
        return
    need = {"kept_segment_count", "compressed_segment_count", "evicted_segment_count"}
    if not need.issubset(eval_df.columns):
        return

    df = eval_df.copy()
    df["total"] = df["kept_segment_count"] + df["compressed_segment_count"] + df["evicted_segment_count"]
    df = df[df["total"] > 0]
    if df.empty:
        warn("All segment counts are zero — KCD codec may not be running")
        return

    for pol_name, sub in df.groupby("policy_name"):
        if "keep_compress_drop" not in pol_name:
            continue
        all_keep = (sub["kept_segment_count"] == sub["total"]).mean()
        all_drop = (sub["evicted_segment_count"] == sub["total"]).mean()
        frac_k = sub["kept_segment_count"].sum() / sub["total"].sum()
        frac_c = sub["compressed_segment_count"].sum() / sub["total"].sum()
        frac_e = sub["evicted_segment_count"].sum() / sub["total"].sum()

        if all_keep > 0.90:
            fail(f"{pol_name}: {all_keep*100:.0f}% of rows keep ALL segments — "
                 "geometry scoring may be degenerate (κ all near zero)")
        elif all_drop > 0.50:
            fail(f"{pol_name}: {all_drop*100:.0f}% of rows drop ALL segments — "
                 "budget constraint may be too aggressive or scoring reversed")
        else:
            ok(f"{pol_name}: keep={frac_k:.2f} compress={frac_c:.2f} drop={frac_e:.2f}")


def check_geometry_scores(cand_df: pd.DataFrame):
    if cand_df.empty or "geometry_score" not in cand_df.columns:
        return
    gs = cand_df["geometry_score"].dropna()
    if len(gs) == 0:
        return
    if gs.std() < 1e-4:
        fail(f"geometry_score has zero variance (all={gs.mean():.4f}) — "
             f"hidden state extraction may be broken")
    elif gs.max() < 0.01:
        fail(f"geometry_score max={gs.max():.4f} — curvature scores near zero, "
             f"check that model is loaded on CUDA with bfloat16")
    elif (gs < 0).mean() > 0.05:
        warn(f"{(gs < 0).mean()*100:.0f}% negative geometry_scores — "
             f"check normalization")
    else:
        ok(f"geometry_score range [{gs.min():.3f}, {gs.max():.3f}] "
           f"mean={gs.mean():.3f} std={gs.std():.3f}")

    # check that keep > drop (geometry should be predictive)
    if "action" in cand_df.columns:
        keep_mean = cand_df.loc[cand_df["action"] == "keep_turn", "geometry_score"].mean()
        drop_mean = cand_df.loc[cand_df["action"] == "drop_turn", "geometry_score"].mean()
        if pd.notna(keep_mean) and pd.notna(drop_mean):
            if keep_mean > drop_mean:
                ok(f"Geometry discriminates: keep mean={keep_mean:.3f} > drop mean={drop_mean:.3f}")
            else:
                warn(f"Geometry NOT discriminating: keep={keep_mean:.3f} <= drop={drop_mean:.3f} — "
                     f"check signal direction")


def check_longllmlingua(eval_df: pd.DataFrame):
    if eval_df.empty or "token_fraction" not in eval_df.columns:
        return
    ling = eval_df[eval_df["policy_name"] == "longllmlingua"]
    if ling.empty:
        info("longllmlingua policy not yet in results")
        return
    # identity check: if token_fraction == 1.0 for most rows, it's not compressing
    no_compress = (ling["token_fraction"] > 0.95).mean()
    if no_compress > 0.50:
        fail(f"LongLLMLingua: {no_compress*100:.0f}% of rows have token_fraction>0.95 — "
             f"compressor is returning uncompressed text (llmlingua not installed or crashing)")
    else:
        ok(f"LongLLMLingua compressing: mean token_fraction={ling['token_fraction'].mean():.3f}")


def check_shard_progress(results_root: Path):
    import json
    progress_files = sorted(results_root.rglob("progress.json"))
    if not progress_files:
        info("No progress.json files found yet")
        return
    done = sum(1 for p in progress_files
               if json.loads(p.read_text()).get("status") == "complete")
    total = len(progress_files)
    info(f"Shards: {done}/{total} complete")
    if done == 0:
        warn("No shards complete yet — check GPU worker logs")
    elif done < total:
        info(f"  In-progress shards: {total - done}")
    else:
        ok("All shards complete")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/reviewer_fixes")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--expected-policies", default=
                    "uniform,longllmlingua,geometry_keep_compress_drop,"
                    "semantic_keep_compress_drop")
    args = ap.parse_args()

    root = Path(args.results_root)
    expected = [p.strip() for p in args.expected_policies.split(",") if p.strip()]

    print(f"\n{'='*60}")
    print(f"  Sanity Check — {root}")
    print(f"{'='*60}\n")

    eval_df  = load_csvs(root, "evaluation_rows.csv")
    cand_df  = load_csvs(root, "candidate_rows.csv")

    check_shard_progress(root)
    print()
    check_row_count(eval_df)
    check_policies(eval_df, expected)
    check_budget_adherence(eval_df)
    check_score_distribution(eval_df)
    check_kcd_actions(eval_df)
    check_longllmlingua(eval_df)
    print()
    check_geometry_scores(cand_df)

    print(f"\n{'='*60}")
    if criticals:
        print(f"{FAIL} CRITICAL ({len(criticals)} issues):")
        for c in criticals:
            print(f"  • {c}")
        print(f"{'='*60}\n")
        sys.exit(2)
    elif issues:
        print(f"{WARN} Warnings ({len(issues)} issues — experiment can continue):")
        for w in issues:
            print(f"  • {w}")
        print(f"{'='*60}\n")
        sys.exit(1)
    else:
        print(f"{OK}  All checks passed — experiment looks healthy")
        print(f"{'='*60}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
