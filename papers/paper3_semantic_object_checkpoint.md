# Paper 3 Semantic-Object Codec

This note defines the next additive Paper 3 surface for semantic-memory benchmarks.

## Motivation

The current semantic-gap finding is not that semantic retrieval fails. It is that turn-local geometry refinement improves decoder-state fidelity slightly while often failing to improve answer behavior on semantic-memory benchmarks such as `MSC` and `LongMemEval-S cleaned`.

That suggests the selection unit is too fine-grained. These benchmarks often reward preservation of a memory object rather than a single turn:

- persona bundle
- event bundle
- constraint bundle
- update/freshest-state bundle

## New method

The new surface treats semantic shortlists as the front door and then builds budget-aware memory objects inside that shortlist.

Implemented additions:

- `semantic_object_keep_compress_drop`
  - builds semantic objects from the shortlist
  - compresses at the object level rather than the raw segment level
- `semantic_object_harm_keep_compress_drop`
  - uses the same object decomposition
  - replaces heuristic object risk with learned predicted harm

Object-level features are now also exported into the oracle and harm-predictor surfaces:

- `object_size`
- `object_recency`
- `object_memory_score`
- `is_object_anchor`
- `is_object_freshest`
- one-hot object-type indicators for:
  - persona
  - event
  - constraint
  - update
  - generic

## Intended evaluation

The right comparison set for the first object-level check is:

- `semantic`
- `budget_aware_semantic_keep_compress_drop`
- `semantic_object_keep_compress_drop`
- `semantic_object_harm_keep_compress_drop` when an oracle-trained predictor is available

Benchmarks:

- `MSC valid`
- `LongMemEval-S cleaned`
- `LoCoMo10` as supporting evidence

## Success criterion

This surface is only interesting if object-level compression improves answer behavior on semantic-memory benchmarks relative to the budget-aware semantic codec baseline. If it only improves logit and not answer NLL, then the object decomposition is still not matching the benchmark’s true memory object.
