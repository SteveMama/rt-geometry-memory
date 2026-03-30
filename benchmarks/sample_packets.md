# Benchmark Sample Packets

This file gives a human-readable sample of the benchmark families we have been
using in Paper 3.

The goal is not to summarize scores. The goal is to let you see what the data
actually looks like.

The samples below were copied from official benchmark downloads or local repo
assets on 2026-03-30 and lightly trimmed for readability.

## MSC

MSC is a multi-session conversation benchmark. In the form we use, it is mostly
about persona continuity, preference continuity, topic continuity, and casual
dialogue carryover. It is not primarily a strict fact QA benchmark.

### Sample 1

Initial personas:

- Speaker 1:
  - Horror movies are my favorites.
  - I'm a stay at home dad.
  - My father used to work for Home Depot.
- Speaker 2:
  - I read twenty books a year.
  - I'm a stunt double as my second job.
  - I only eat kosher.

Session 1 opening:

```text
Speaker 1: Hello what are doing today?
Speaker 2: I am good, I just got off work and tired, I have two jobs.
Speaker 1: I just got done watching a horror movie
Speaker 2: I rather read, I've read about 20 books this year.
```

Session 5 ending:

```text
Speaker 1: You definately should. If you like serial killer books and mysteries you will love it.
Speaker 2: How do you like using your kindle? I use my phone to read my books, but I'm starting to think I should just get a paperwhite and separate my reading from my phone
Speaker 1: It's good. It's got a better screen than my phone and it's bigger. I mostly got it because it was bigger than my phone and it was cheaper than a tablet.
Speaker 2: Yeah I think I'm going to order one right now. Honestly I think it'll help me read even more, which is always a good thing
```

What this tests:

- remembering the speakers' preferences
- maintaining topical continuity across sessions
- continuing a casual conversation in the right semantic neighborhood

### Sample 2

Initial personas:

- Speaker 1:
  - I volunteer in a homeless shelter.
  - I enjoy being around people.
- Speaker 2:
  - I have two dogs.
  - I like to work on vintage cars.
  - My favorite music is country.

Session 1 opening:

```text
Speaker 1: Hi! How are you doing tonight?
Speaker 2: I'm doing great. Just relaxing with my two dogs.
Speaker 1: Great. In my spare time I do volunteer work.
Speaker 2: That's neat. What kind of volunteer work do you do?
```

Session 5 ending:

```text
Speaker 2: you really deserve it. Maybe just relax and listen to some country music tonight.
Speaker 1: Hahaha maybe some old Swift and a bubblebath. My muscles will be hurting tomorrow. My opponent was good!
Speaker 2: That sounds really relaxing. If you were able to beat them then you should be able to beat your next opponents too!
```

What this tests:

- persona and hobby continuity
- emotional and conversational tone
- recent-plan and encouragement continuity

## LoCoMo10

LoCoMo is a long multi-session conversational QA benchmark. It mixes biography,
event memory, and temporal localization.

### Sample 1

Speakers:

- Caroline
- Melanie

Conversation excerpts:

```text
Session 1 date/time: 1:56 pm on 8 May, 2023

D1:1 Caroline: Hey Mel! Good to see you! How have you been?
D1:2 Melanie: Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? Anything new?
D1:3 Caroline: I went to a LGBTQ support group yesterday and it was so powerful.
D1:4 Melanie: Wow, that's cool, Caroline! What happened that was so awesome? Did you hear any inspiring stories?

Session 2 date/time: 1:14 pm on 25 May, 2023

D2:1 Melanie: Hey Caroline, since we last chatted, I've had a lot of things happening to me. I ran a charity race for mental health last Saturday.
D2:2 Caroline: That charity race sounds great, Mel!
```

Questions:

```text
Q: When did Caroline go to the LGBTQ support group?
A: 7 May 2023
Evidence: D1:3

Q: What fields would Caroline be likely to pursue in her education?
A: Psychology, counseling certification
Evidence: D1:9, D1:11
```

What this tests:

- temporal grounding
- fact retrieval from earlier sessions
- compressed biography and life-event memory

### Sample 2

Speakers:

- Jon
- Gina

Conversation excerpts:

```text
Session 1 date/time: 4:04 pm on 20 January, 2023

D1:1 Gina: Hey Jon! Good to see you. What's up? Anything new?
D1:2 Jon: Hey Gina! Good to see you too. Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business.
D1:3 Gina: Sorry about your job Jon, but starting your own business sounds awesome! Unfortunately, I also lost my job at Door Dash this month.
D1:4 Jon: Sorry to hear that! I'm starting a dance studio 'cause I'm passionate about dancing.
```

Questions:

```text
Q: How do Jon and Gina both like to destress?
A: by dancing
Evidence: D1:7, D1:6

Q: What do Jon and Gina both have in common?
A: They lost their jobs and decided to start their own businesses.
Evidence: D1:2, D1:3, D1:4, D2:1
```

What this tests:

- multi-turn fact aggregation
- shared-trait inference
- event and identity reconstruction

## LongMemEval-S Cleaned

LongMemEval is a long-memory QA benchmark built from many haystack sessions.
The final question is explicit, but the relevant evidence is buried inside many
other conversational sessions.

### Sample 1

```text
Question: What degree did I graduate with?
Answer: Business Administration
```

Relevant haystack session excerpt:

```text
user: I'm trying to organize my life a bit better, can you recommend some task management apps that can help me prioritize my work and personal tasks?
assistant: Making the leap from a planner to a digital task management system...
user: I think I'll try out Todoist and Trello. By the way, do you have any tips on creating a routine for my new job?
assistant: Todoist and Trello are both excellent choices...
```

What this tests:

- remembering personal life facts hidden among many unrelated sessions
- ignoring large amounts of distractor content

### Sample 2

```text
Question: How long is my daily commute to work?
Answer: 45 minutes each way
```

Relevant haystack session excerpt:

```text
user: I'm looking for some new fiction audiobook recommendations. I've been enjoying audiobooks a lot lately, especially during my daily commute.
assistant: Audiobooks are a great way to make the most of your commute time...
user: I'm already reading Gone Girl on my Kindle, but The Nightingale sounds interesting.
assistant: Great choice on Gone Girl! The Nightingale is an excellent choice...
```

What this tests:

- retrieval of a personal fact from a semantically broad conversation
- query alignment under heavy haystack noise

## Hard Stress Set

The hard stress set in this repo is not a public benchmark. It is a deliberately
adversarial support-turn benchmark built around exact constraints, exact field
retrieval, formatting commitments, and technical instruction fidelity.

Source:

- [`paper2_behavior_stress_conversations.jsonl`](../paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl)

### Sample 1: exact retrieval packet

```text
user: Remember this launch packet exactly: codename Atlas, owner Rina, launch date June 3, and legal review due May 20.
assistant: Stored: Atlas, owner Rina, launch June 3, legal review May 20.
user: Add these launch logistics exactly: design review Thursday at 2 PM in Cedar, backup reviewer Omar, and status color amber.
assistant: Added: design review Thursday at 2 PM in Cedar, backup reviewer Omar, status amber.
user: Now give me exactly owner | launch date | review room.
assistant: Rina | June 3 | Cedar
```

What this tests:

- exact field preservation
- compositional retrieval across multiple support turns
- decoder-faithful output structure

### Sample 2: exact long-dependency format rule

```text
user: For the rest of this chat, answer in exactly two lowercase lines, and each line must have exactly two words.
assistant: rule stored | format locked
user: Remember these facts exactly: vendor northline, owner rina, due august 9, total 4820 dollars.
assistant: northline vendor | rina owner
user: Now give me vendor and owner only.
assistant: vendor northline | owner rina
```

What this tests:

- exact structural constraint preservation
- formatting fidelity
- support-turn-critical memory

## One-line reading

- MSC: semantic conversational continuity
- LoCoMo: semantic biography plus temporal event memory
- LongMemEval: haystack retrieval of personal facts
- Hard stress set: support-turn-critical structural memory

That is why the benchmark winners differ.
