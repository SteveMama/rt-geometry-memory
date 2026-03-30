# Synthetic State-Update Alignment Smoke Checkpoint

This note records the current falsification-style smoke checkpoint for the
state-update supersession hypothesis.

## Goal

Test whether turn-level geometry can identify when a later turn supersedes an
earlier same-topic turn in a way that plain semantic similarity cannot.

Two candidate signals were tested:

- original/update turn pair: high semantic similarity
- same-sign geometric directional alignment: negative
- state-position / update-entry cross-term: negative

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
- tracked cross-control artifact:
  - [`artifacts/paper3/state_update_cross_control_qwen05b`](/Users/pranav/Documents/RT/artifacts/paper3/state_update_cross_control_qwen05b)

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
synthetic benchmark, and the follow-on state-position / update-entry cross-term
also failed as a sign-based detector.

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

## Follow-On Cross-Term Check

The geometrically consistent cross-term check compared:

\[
\cos\left(\log_q(h_s), u_t^{\mathrm{entry}}\right)
\]

where the original-turn state was mapped into the same tangent space as the
entry increment at the later update turn.

Aggregate result from
[`summary.json`](/Users/pranav/Documents/RT/artifacts/paper3/state_update_cross_control_qwen05b/summary.json):

- mean state/update cross: `0.9707`
- mean sampled non-update control cross: `0.9795`
- mean all-control cross: `0.9780`
- negative update crosses: `0 / 10`
- negative control crosses: `0 / 10`
- update more negative than mean control: `9 / 10`
- update more negative than all controls: `6 / 10`

This is not a usable supersession detector. The update cross is usually
\emph{slightly less positive} than a same-role non-update control, but it is not
negative, and the margin is small.

## Secondary Diagnostic

The runner also recorded three alignment diagnostics:

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

This does **not** rescue the original claim. It means the formal story was
underspecified about which turn-associated geometric quantity should represent a
state update:

- the direct implementation of the written formula fails
- a mixed entry/exit comparison produces the intended negative sign
- but a geometrically consistent state/increment cross-term still fails as a
  negative detector
- therefore the theorem direction is not yet well posed enough to use as an
  algorithmic result

## Reading

The clean current interpretation is:

1. The simple claim
   - “same-topic state updates yield negative same-sign directional alignment”
   is not supported by this smoke test.
2. The follow-on state/increment cross-term is also not a clean detector.
3. The sign flip in the mixed entry/exit diagnostics suggests a representational
   mismatch:
   - “effect of receiving the old fact”
   - versus
   - “effect of leaving the updated state”
4. At current compact-model scale, the surviving effect is only a very weak
   ranking margin rather than a robust sign change.
5. That is not strong enough to justify expansion into `MSC`, `LoCoMo`, or
   compression experiments.

## Immediate Consequence

Do **not** proceed to the larger benchmark pass on either current state-update
formulation.

If this line is pursued further, the next step should be a tighter mathematical
reformulation of the turn-associated representation for state updates, or a
move to larger models where turn-level state geometry may have higher
resolution.
