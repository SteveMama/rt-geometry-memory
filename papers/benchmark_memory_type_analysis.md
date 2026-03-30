# Benchmark Memory-Type Analysis

This note records a direct read-through of the benchmark data rather than another model run.

The purpose is simple:

> determine what kinds of conversational memory each benchmark actually tests, then check whether the empirical signal winners match those task structures.

This note is intentionally qualitative and dataset-facing. It is the missing explanatory layer behind the Paper 3 benchmark split.

## Datasets inspected

The analysis below is based on direct reads of:

- `MSC valid`
  - raw: `/tmp/msc_valid.jsonl`
  - normalized: `/tmp/msc_valid_normalized.jsonl`
- `LoCoMo10`
  - raw: `/tmp/locomo10.json`
  - normalized: `/tmp/locomo10_normalized.jsonl`
- hard stress set
  - [`paper2_behavior_stress_conversations.jsonl`](/Users/pranav/Documents/RT/paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl)

The analysis used:

- several direct conversation reads from each benchmark
- the final benchmark question or continuation target where applicable
- the task structure implied by the source format itself

## Core result

The benchmarks are not testing the same memory object.

That is the main reason the policy winners differ.

## MSC

## What the raw data is

`MSC` is not primarily a retrieval-QA dataset in the form used here.

The raw schema has:

- `init_personas`
- `sessions`

with `5` dialogue sessions per example.

The normalized form is a continuation-style dialogue example, not a standalone fact question.

## What the examples look like

Representative continuation targets involve things like:

- book preferences and reading-device choice
- post-fight relaxation and future encouragement
- job, family, home-decoration, and budgeting context
- disability, reading preferences, and social support

These are not “find one exact hidden constraint” tasks.

They are mostly:

- persona continuity
- preference continuity
- topic continuation
- emotional tone alignment
- recent conversational plan tracking

## What kind of memory MSC requires

Typical memory types:

- persona trait
- preference or hobby
- recent plan or next action
- emotional stance
- lightweight fact continuity across casual chat

Where the answer usually lives:

- often in one or two recent or mid-recent turns
- sometimes paraphrased across repeated sessions
- often in a broad topical cluster rather than a single uniquely critical support turn

How a human would remember it:

- keep the main topic
- keep the user’s trait or preference
- keep the current plan or emotional state
- ignore most wording details

## Signal prediction

MSC should favor:

- semantic relevance
- persona retrieval
- topic continuity

MSC should not strongly favor:

- geometry-only support-turn preservation
- exact wording fidelity
- sharp constraint transitions

## Match to experiments

This matches the observed pattern:

- semantic baselines win most often
- semantic-led codecs are strongest overall
- old geometry-KCD is poor
- support-aware geometry-KCD becomes viable only after strong repair

So MSC is best understood as a **semantic conversational-memory** benchmark.

## LoCoMo

## What the normalized benchmark is

`LoCoMo10` becomes a long multi-turn QA benchmark in normalized form:

- each normalized example is a very long conversation
- the final user turn is an explicit question
- the final assistant turn is the gold answer

The first `120` normalized questions were dominated by:

- `what`: `59`
- `when`: `35`
- `how`: `9`
- `other`: `12`

with very few `where`, `why`, and `who` questions in that slice.

## What the questions look like

Representative examples:

- “When did Caroline go to the LGBTQ support group?”
- “What fields would Caroline be likely to pursue in her education?”
- “What did Caroline research?”
- “What is Caroline’s identity?”
- “When is Melanie planning on going camping?”
- “What is Caroline’s relationship status?”

These are mostly:

- factual retrieval
- temporal localization
- biography/persona recall
- event-history reconstruction
- some light inference or counterfactual reasoning

## What kind of memory LoCoMo requires

LoCoMo mixes two memory types:

1. semantic-biographical memory
2. temporal-event-chain memory

Where the answer lives:

- often in an earlier factual mention
- sometimes in repeated paraphrases over the conversation
- sometimes in date-anchored or event-chain structure

How a human would remember it:

- keep salient facts about the person
- keep major dates and event anchors
- compress repeated biography into a stable semantic representation
- sometimes keep event order, not exact wording

## Signal prediction

LoCoMo should favor:

- semantic methods for identity, relationship, career, and factual biography questions
- semantic codecs that preserve dense fact clusters

LoCoMo may help geometry when:

- the question depends on event ordering
- the answer depends on a structured event progression
- the relevant region is a coherent local chain rather than a single topical fact

## Match to experiments

This matches the observed mixed outcome:

- semantic and semantic-KCD perform strongly overall
- query-conditioned or support-aware geometry can become competitive under tight budgets
- but geometry is not the dominant signal family overall

So LoCoMo is a **mixed semantic + temporal memory** benchmark, but still much more semantic than the hard stress set.

## Hard stress set

## What the data is

The hard stress set is qualitatively different from both MSC and LoCoMo.

It contains short, dense, adversarial conversations designed around:

- exact launch packets
- travel itineraries
- medication schedules
- exact formatting constraints
- code-generation constraints
- SQL query constraints

Representative tasks:

- “Now give me exactly codename | legal due | review time.”
- “For the rest of this chat, answer in exactly two lowercase lines...”
- “Now show the full query in exactly five lines.”

## What kind of memory it requires

Typical memory types:

- exact support constraint
- exact formatting rule
- multi-field retrieval under strict output specification
- compositional technical constraints

Where the answer lives:

- often in very specific earlier turns
- often spread over a few critical support turns
- often requires exact structural combination, not just topic recall

How a human would remember it:

- remember the instruction exactly
- remember the exact fields or schema
- remember the formatting rule
- ignore surrounding small talk because there is almost none

## Signal prediction

This benchmark should favor:

- support-turn preservation
- structured decoder-faithful memory
- geometry-sensitive or support-aware sparse codecs

It should not strongly favor:

- broad topical retrieval alone

## Match to experiments

This matches the observed results:

- geometry and support-aware geometry-KCD are strongest here
- semantic relevance alone is not enough

So the hard stress set is a **support-turn-critical structural memory** benchmark.

## Human-memory view across benchmarks

If a human were solving these tasks, they would remember different things.

### MSC

- who likes what
- what the current topic is
- what the speaker is planning
- how the speaker feels

### LoCoMo

- major identity facts
- dates and event anchors
- a compressed biographical timeline

### Hard stress set

- exact rules
- exact fields
- exact code/query constraints
- exact formatting commitments

That means the benchmark split is not noise. It reflects genuinely different memory objects.

## What semantic preserves that geometry often misses

- topical relevance
- persona traits
- named entities and roles
- diffuse biographical memory
- broad fact clusters

## What geometry preserves that semantic often misses

- exact support-turn transitions
- structural constraints
- formatting and decoding commitments
- compositional instruction changes
- local event-chain structure when it is expressed as a sharp representational transition

## Main scientific conclusion

The right explanatory framework is:

> different conversational memory tasks require different memory signals.

More specifically:

- `MSC` mostly tests semantic conversational continuity
- `LoCoMo` mixes semantic biography with temporal event reconstruction
- the hard stress set tests support-turn-critical structural memory

This predicts the observed ranking split:

- semantic-led policies win on MSC and often on LoCoMo
- geometry-led sparse codecs win on the hard stress set

## What this means for Paper 3

Paper 3 should no longer be framed as:

- “find one universal best codec policy”

It should be framed as:

- “match codec signal to memory type”

That creates a much stronger research program:

1. semantic-first codec for semantic conversational memory
2. support-aware geometry codec for structural/support-turn-critical memory
3. hybrid selectors for mixed benchmarks such as LoCoMo

## Immediate implication

The next algorithm should be chosen from this benchmark reading, not from blind score chasing.

The cleanest next move is:

> semantic candidate discovery plus support-aware, query-conditioned geometric refinement inside the shortlisted memory.

That follows directly from the benchmark structure:

- semantic is the right front door for MSC and much of LoCoMo
- geometry is still valuable, but as a structural refinement signal rather than the sole ranking signal
