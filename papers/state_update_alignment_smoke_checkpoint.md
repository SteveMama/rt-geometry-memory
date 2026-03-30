# Synthetic State-Update Alignment Smoke Checkpoint

This note records the first falsification-style smoke test for the state-update
supersession hypothesis.

## Goal

Test whether geometric directional alignment can identify when a later turn
supersedes an earlier same-topic turn in a way that plain semantic similarity
cannot.

The intended signal was:

- original/update turn pair: high semantic similarity
- original/update geometric alignment: negative

The synthetic gate was defined before running:

- at least `7 / 10` conversations with directional alignment `< -0.2`
- and original/update semantic similarity `> 0.5`

## Setup

- synthetic benchmark:
  - [`benchmarks/state_update_synthetic_conversations.jsonl`](/Users/pranav/Documents/RT/benchmarks/state_update_synthetic_conversations.jsonl)
- labels:
  - [`benchmarks/state_update_synthetic_labels.json`](/Users/pranav/Documents/RT/benchmarks/state_update_synthetic_labels.json)
- runner:
  - [`scripts/run_state_update_alignment_check.py`](/Users/pranav/Documents/RT/scripts/run_state_update_alignment_check.py)
- shell entrypoint:
  - [`scripts/run_state_update_alignment_smoke.sh`](/Users/pranav/Documents/RT/scripts/run_state_update_alignment_smoke.sh)
- tracked artifact:
  - [`artifacts/paper3/state_update_alignment_smoke_qwen05b`](/Users/pranav/Documents/RT/artifacts/paper3/state_update_alignment_smoke_qwen05b)

Model:

- `qwen25_05b`
- resolved model: `Qwen/Qwen2.5-0.5B-Instruct`
- device: `mps`

The primary directional-alignment metric follows the stated formula directly:

\[
u_t = PT_{h_t \to q}\left(\log_{h_t}(h_{t+1})\right)
\]

and compares the labeled original turn and update turn using:

\[
A(s,t)=\cos(u_s,u_t)
\]

Semantic similarity is computed from isolated-turn hidden states using the same
base model.

## Result

The straightforward same-sign supersession hypothesis **failed cleanly** on the
synthetic benchmark.

Aggregate result from
[`summary.json`](/Users/pranav/Documents/RT/artifacts/paper3/state_update_alignment_smoke_qwen05b/summary.json):

- conversations: `10`
- mean directional alignment: `0.9128`
- median directional alignment: `0.9231`
- negative alignments: `0 / 10`
- alignments below `-0.2`: `0 / 10`
- mean original/update semantic similarity: `0.9942`
- semantic similarities above `0.5`: `10 / 10`
- joint threshold passes: `0 / 10`
- falsification gate pass: `False`

So the key intended pattern did **not** appear:

- semantics stayed very high, as expected
- but directional alignment stayed strongly **positive**, not negative

## Secondary Diagnostic

The runner also recorded three diagnostic alignment variants:

- entry-entry alignment:
  - transported step into the original turn vs transported step into the update
    turn
- original-entry / update-exit
- original-exit / update-entry

Those diagnostics showed an important ambiguity:

- mean entry-entry alignment: `0.9395`
- mean original-entry / update-exit alignment: `-0.9192`
- mean original-exit / update-entry alignment: `-0.9347`
- both mixed-sign diagnostics were negative in `10 / 10` cases

This does **not** rescue the original claim. It means the formal story is
currently underspecified about which turn-associated vector should represent a
state update:

- the direct implementation of the written formula fails
- a mixed entry/exit comparison produces the intended negative sign
- therefore the theorem direction is not yet well posed enough to use as an
  algorithmic result

## Reading

The clean current interpretation is:

1. The simple claim
   - “same-topic state updates yield negative same-sign directional alignment”
   is not supported by this smoke test.
2. The state-update geometry story is not dead, but it is not ready.
3. The sign flip in the mixed diagnostics suggests a representational mismatch:
   - “effect of receiving the old fact”
   - versus
   - “effect of leaving the updated state”
4. That mismatch needs a sharper mathematical definition before expanding this
   into `MSC`, `LoCoMo`, or compression experiments.

## Immediate Consequence

Do **not** proceed to the larger benchmark pass on the current same-sign
alignment formulation.

If this line is pursued further, the next step should be a tighter mathematical
reformulation of the turn-associated vector for state updates, followed by
another synthetic falsification run before touching public benchmarks.
