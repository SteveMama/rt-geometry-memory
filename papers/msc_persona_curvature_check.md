# MSC Persona Curvature Check

This note records a targeted falsification-style check on `MSC`.

The question was narrow:

> if we manually identify earlier persona/support turns that plausibly support
> the final continuation, do those turns have higher curvature than nearby
> filler turns in the same conversation?

This is the right test before adding more geometry on top of semantic
baselines. If support turns are not geometrically distinguished from filler
inside `MSC`, then a semantic-first explanation is more plausible than a
geometry-first one.

## Setup

- benchmark: `msc_valid`
- model: `qwen25_05b`
- input window: `768` tokens
- conversations: `5` manually chosen `MSC` conversations
- labels:
  - support/persona turns: earlier turns judged to provide factual support for
    the final assistant continuation
  - filler turns: nearby non-support turns from the same conversation

The exact labels are tracked in:

- [`benchmarks/msc_persona_curvature_labels.json`](/Users/pranav/Documents/RT/benchmarks/msc_persona_curvature_labels.json)

The executable runner is:

- [`scripts/run_msc_persona_curvature_check.py`](/Users/pranav/Documents/RT/scripts/run_msc_persona_curvature_check.py)

The generated artifact bundle is:

- [`msc_persona_curvature_v1`](/Users/pranav/Documents/RT/results/msc_persona_curvature/msc_persona_curvature_v1)

## Main result

The result is negative for a strong geometry-within-MSC claim.

Across the five manually labeled conversations:

- mean support minus filler curvature delta: `0.1220`
- median delta: `0.0104`
- positive deltas: `3 / 5`
- negative deltas: `2 / 5`
- mean support percentile: `0.6139`
- mean filler percentile: `0.6087`

That is not a robust separation.

If stabilized curvature were strongly identifying persona/support turns inside
`MSC`, we would expect support turns to sit clearly above filler turns across
most or all conversations. That does not happen here.

## Per-conversation read

### `msc-00000`

- support mean curvature: `1.0656`
- filler mean curvature: `0.0000`
- result: positive

This is one of the few favorable cases, but it is partly driven by later turns
collapsing to zero curvature under near-stationary dynamics.

### `1`

- support mean curvature: `0.5025`
- filler mean curvature: `1.0048`
- result: negative

This is a clean counterexample: the chosen wrestling-support turns are not more
curved than the chosen filler turns.

### `2`

- support mean curvature: `0.9594`
- filler mean curvature: `0.5345`
- result: positive

There is some signal here, but the percentile advantage is still small.

### `6`

- support mean curvature: `0.6757`
- filler mean curvature: `1.0642`
- result: negative

Another clean counterexample: the pet-memory support turns are not more curved
than filler persona turns.

### `8`

- support mean curvature: `2.0098`
- filler mean curvature: `1.9993`
- result: essentially tied

This is effectively no separation.

## Scientific reading

The most honest conclusion is:

> On this manual five-conversation `MSC` check, we do not see a stable
> curvature-based separation between earlier support/persona turns and filler
> turns.

That means the current evidence does **not** justify the claim that geometry is
providing a strong within-topic support discriminator on `MSC`.

This supports a more cautious interpretation of the broader benchmark results:

- semantic baselines likely win on `MSC` because they recover the right
  conversational facts and topics directly
- geometry may still help on support-turn-critical structural tasks
- but for `MSC`, the burden of proof is now on any claim that geometry adds
  meaningful turn-selection signal beyond semantics

## What this rules out

This check does not prove that geometry is useless everywhere.

It does rule out a stronger claim that would have been attractive but is not
supported:

> once we are inside a semantic cluster on `MSC`, curvature cleanly identifies
> the important persona/support turns

The current manual evidence does not support that.

## What it suggests next

The next correct move is not another geometry heuristic on `MSC`.

It is one of:

1. a semantic-first codec story on semantic-memory benchmarks
2. a leave-one-out or harm-style support test instead of raw curvature
3. a focused re-check on the hard stress set, where support-turn geometry has
   already shown real value

That is a better use of time than assuming geometry must still be the missing
within-cluster signal on `MSC`.
