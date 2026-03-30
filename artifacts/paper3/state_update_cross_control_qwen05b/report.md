# Synthetic State-Update Alignment Check

- Benchmark: `state_update_synthetic_v1`
- Model: `qwen25_05b` (Qwen/Qwen2.5-0.5B-Instruct)
- Device: `mps`
- Conversations: 10
- Alignment threshold: `-0.2`
- Semantic threshold: `0.5`
- Mean directional alignment: 0.9128
- Median directional alignment: 0.9231
- Negative alignments: 0 / 10
- Alignments below threshold: 0 / 10
- Mean entry alignment (diagnostic): 0.9395
- Mean original-entry / update-exit alignment (diagnostic): -0.9192
- Mean original-exit / update-entry alignment (diagnostic): -0.9347
- Negative mixed entry/exit counts: 10 / 10 and 10 / 10
- Mean state-position / update-entry cross (diagnostic): 0.9707
- Mean sampled non-update control cross (diagnostic): 0.9795
- Mean all-control cross (diagnostic): 0.9780
- Negative state-update cross count: 0 / 10
- Negative sampled-control cross count: 0 / 10
- Update more negative than mean control: 9 / 10
- Update more negative than all controls: 6 / 10
- Mean original-update semantic similarity: 0.9942
- Semantic similarities above threshold: 10 / 10
- Joint threshold passes: 0 / 10
- Mean query-similarity gap (update - original): -0.0010
- Falsification gate pass: `False`

## Interpretation

Directional alignment is computed exactly from the stated formula: the transported step from each labeled turn to the following turn, all moved into one shared tangent space.
Semantic similarity is computed from isolated-turn hidden states using the same base model.
For debugging, the CSV also includes entry-step alignment and the two mixed entry/exit alignments.

## Per-conversation results

| Conversation | Exit align | Entry align | State/update cross | Mean control cross | Pair semantic | Joint pass |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| state-update-001-employment | 0.8533 | 0.9579 | 0.9782 | 0.9826 | 0.9945 | no |
| state-update-002-city | 0.8863 | 0.9568 | 0.9809 | 0.9802 | 0.9934 | no |
| state-update-003-codename | 0.9369 | 0.9211 | 0.9712 | 0.9725 | 0.9965 | no |
| state-update-004-team-lead | 0.9305 | 0.9098 | 0.9518 | 0.9775 | 0.9923 | no |
| state-update-005-phone | 0.9233 | 0.9546 | 0.9735 | 0.9802 | 0.9943 | no |
| state-update-006-diet | 0.9229 | 0.9508 | 0.9741 | 0.9775 | 0.9927 | no |
| state-update-007-apartment | 0.9164 | 0.9444 | 0.9664 | 0.9695 | 0.9943 | no |
| state-update-008-workout | 0.9317 | 0.9168 | 0.9673 | 0.9798 | 0.9946 | no |
| state-update-009-gate | 0.8866 | 0.9517 | 0.9737 | 0.9803 | 0.9954 | no |
| state-update-010-manager | 0.9406 | 0.9307 | 0.9701 | 0.9796 | 0.9940 | no |

## Labeled update pairs

### state-update-001-employment

- Original turn `2`: One big change is that I just started working at Google as a software engineer.
- Update turn `8`: Actually, plans changed fast. I left Google last week and now I work at Meta on AR tools.
- Query turn `10`: Quick check: where do I work now?
- Expected current answer: `Meta on AR tools`
- Sampled control turn `6` with cross `0.9871`

### state-update-002-city

- Original turn `2`: At first I moved to Seattle for the new apartment near Green Lake.
- Update turn `8`: Small correction though: the Seattle plan fell through, so I moved to Austin instead.
- Query turn `10`: Which city do I live in now?
- Expected current answer: `Austin`
- Sampled control turn `6` with cross `0.9837`

### state-update-003-codename

- Original turn `2`: Earlier today the internal codename for the launch was Atlas.
- Update turn `8`: Update from leadership: Atlas was dropped, and the current codename is Beacon.
- Query turn `10`: What is the current codename?
- Expected current answer: `Beacon`
- Sampled control turn `4` with cross `0.9786`

### state-update-004-team-lead

- Original turn `2`: At the start of the week, my team lead was Jules.
- Update turn `8`: Leadership changed today, so Mina is my team lead now instead of Jules.
- Query turn `10`: Who is my team lead now?
- Expected current answer: `Mina`
- Sampled control turn `6` with cross `0.9801`

### state-update-005-phone

- Original turn `2`: For a while my main phone was an iPhone 15.
- Update turn `8`: Then I changed my mind and switched over. My main phone is a Pixel 9 now, not the iPhone 15.
- Query turn `10`: What is my main phone now?
- Expected current answer: `Pixel 9`
- Sampled control turn `6` with cross `0.9791`

### state-update-006-diet

- Original turn `2`: Until recently I was following a vegetarian diet.
- Update turn `8`: I updated my plan though. I am pescatarian now, so I am no longer vegetarian.
- Query turn `10`: What diet am I following now?
- Expected current answer: `pescatarian`
- Sampled control turn `6` with cross `0.9804`

### state-update-007-apartment

- Original turn `2`: Before the move, I was in apartment 5A.
- Update turn `8`: The paperwork is done now, and I live in apartment 9C instead of 5A.
- Query turn `10`: Which apartment do I live in now?
- Expected current answer: `9C`
- Sampled control turn `6` with cross `0.9638`

### state-update-008-workout

- Original turn `2`: Originally I was doing my workouts in the evenings after work.
- Update turn `8`: I changed the plan, and now I work out in the mornings instead of the evenings.
- Query turn `10`: When do I work out now?
- Expected current answer: `mornings`
- Sampled control turn `6` with cross `0.9778`

### state-update-009-gate

- Original turn `2`: When I first checked the board, my flight was leaving from gate B12.
- Update turn `8`: And it did change. The airline moved my flight from B12 to C7.
- Query turn `10`: What gate is my flight leaving from now?
- Expected current answer: `C7`
- Sampled control turn `6` with cross `0.9847`

### state-update-010-manager

- Original turn `2`: Before the reorg, my manager was Priya.
- Update turn `8`: But the reorg closed today, and my manager is Daniel now instead of Priya.
- Query turn `10`: Who is my manager now?
- Expected current answer: `Daniel`
- Sampled control turn `4` with cross `0.9792`

