# Benchmarks

This folder documents how to convert a public multi-turn benchmark into the
project's normalized conversation JSONL format.

The Paper 3 benchmark runners expect one JSONL file where each line has:

```json
{
  "conversation_id": "example-001",
  "family": "public_benchmark",
  "turns": [
    {"role": "user", "content": "Earlier turn"},
    {"role": "assistant", "content": "Earlier reply"},
    {"role": "user", "content": "Evaluation question"},
    {"role": "assistant", "content": "Gold answer"}
  ],
  "system_prompt": "Optional system prompt"
}
```

Key requirement:

- the final evaluated pair must be a `user` turn followed by the gold
  `assistant` answer
- earlier turns should preserve the benchmark's original conversational memory
  structure

Recommended first public benchmark:

- `LongMemEval-S cleaned`
  - source dataset: `xiaowu0162/longmemeval-cleaned`
  - file: `longmemeval_s_cleaned.json`
  - normalization mode: `longmemeval`

Fast-iteration public benchmarks with automatic download support:

- `msc_train`, `msc_valid`, `msc_test`
  - source dataset: `gonced8/multi-session_chat`
  - normalization mode: `msc`
- `locomo10`
  - source file: `https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json`
  - normalization mode: `locomo`

Download one of the auto-supported benchmarks in Colab with:

```bash
python scripts/download_public_benchmark.py \
  --benchmark msc_valid \
  --output /content/msc_valid.jsonl
```

Then normalize it:

```bash
python scripts/prepare_public_benchmark_jsonl.py \
  --format msc \
  --input /content/msc_valid.jsonl \
  --output benchmarks/public_benchmark_normalized.jsonl
```

Use the helper below when the source benchmark is in JSON or JSONL:

```bash
python scripts/prepare_public_benchmark_jsonl.py \
  --format locomo \
  --input /path/to/benchmark.json \
  --output benchmarks/locomo_normalized.jsonl
```

Supported input modes:

- `normalized`: validate and rewrite an already-normalized file
- `locomo`: heuristic adapter for LoCoMo-style records
- `msc`: heuristic adapter for multi-session chat records
- `longmemeval`: heuristic adapter for LongMemEval-style records

Current download support:

- automatic in repo scripts/notebooks:
  - `longmemeval_s_cleaned`
  - `longmemeval_m_cleaned`
  - `msc_train`
  - `msc_valid`
  - `msc_test`
  - `locomo10`
- manual-source fallback in the unified notebook:
  - `GapChat`
  - `REALTALK`
  - `EvolMem`
  - any already-normalized JSONL

Quick-benchmark plan:

- [`quick_benchmark_plan.md`](/Users/pranav/Documents/RT/benchmarks/quick_benchmark_plan.md)
- [`sample_packets.md`](/Users/pranav/Documents/RT/benchmarks/sample_packets.md)

Synthetic diagnostic benchmark:

- [`state_update_synthetic_conversations.jsonl`](/Users/pranav/Documents/RT/benchmarks/state_update_synthetic_conversations.jsonl)
  - ten small conversations with one explicit state update each
- [`state_update_synthetic_labels.json`](/Users/pranav/Documents/RT/benchmarks/state_update_synthetic_labels.json)
  - labels for original fact turn, later update turn, and final query turn

If the upstream benchmark schema differs from the heuristics here, preprocess it
into the normalized JSONL structure first. The benchmark runners do not depend
on the original dataset format once this JSONL exists.
