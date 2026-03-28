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

Download it in Colab with:

```bash
python scripts/download_public_benchmark.py \
  --benchmark longmemeval_s_cleaned \
  --output /content/longmemeval_s_cleaned.json
```

Then normalize it:

```bash
python scripts/prepare_public_benchmark_jsonl.py \
  --format longmemeval \
  --input /content/longmemeval_s_cleaned.json \
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
- `longmemeval`: heuristic adapter for LongMemEval-style records

If the upstream benchmark schema differs from the heuristics here, preprocess it
into the normalized JSONL structure first. The benchmark runners do not depend
on the original dataset format once this JSONL exists.
