# Memory-Critical Support Analysis: qwen25_05b @ budget 0.35

This report treats earlier user turns as the support memory units on the hard Paper 2 stress set.
The main question is whether geometry keeps the support turns that the final query depends on while uniform drops them.

## Aggregate

- Cases compared: 36
- Geometry retains more support user turns than uniform: 14 cases
- Uniform retains more support user turns than geometry: 5 cases
- Ties on support-turn retention: 17 cases
- Geometry keeps all support user turns: 0 cases
- Uniform keeps all support user turns: 0 cases
- Geometry keeps the latest support user turn while uniform drops it: 13 cases

## By Family

### code_conversation

- Cases: 12
- Geometry better on support-turn retention: 8
- Uniform better on support-turn retention: 0
- Ties: 4
- Geometry keeps all support user turns: 0
- Uniform keeps all support user turns: 0
- Geometry keeps latest support user turn while uniform drops it: 6

### long_dependency

- Cases: 12
- Geometry better on support-turn retention: 3
- Uniform better on support-turn retention: 2
- Ties: 7
- Geometry keeps all support user turns: 0
- Uniform keeps all support user turns: 0
- Geometry keeps latest support user turn while uniform drops it: 3

### retrieval_heavy

- Cases: 12
- Geometry better on support-turn retention: 3
- Uniform better on support-turn retention: 3
- Ties: 6
- Geometry keeps all support user turns: 0
- Uniform keeps all support user turns: 0
- Geometry keeps latest support user turn while uniform drops it: 4

## Geometry-Only Rescued Support Types

- support constraint: 12
- base memory: 3

## Top Rescued Support Turns

- stress_retrieval_launch_packet t2 (retrieval_heavy, target t5): support constraint, delta logit L2 -733.158
  content: Add these launch logistics exactly: design review Thursday at 2 PM in Cedar, backup reviewer Omar, and status color amber.
- stress_code_js_fetch_helper t2 (code_conversation, target t5): support constraint, delta logit L2 -521.185
  content: Also require GET /users/${id}, throw on non-ok responses, and return parsed JSON.
- stress_code_python_function t2 (code_conversation, target t5): support constraint, delta logit L2 -433.521
  content: Also require stable ordering and use the exact key tuple pattern with x is None and x or empty string.
- stress_retrieval_launch_packet t2 (retrieval_heavy, target t7): support constraint, delta logit L2 -399.785
  content: Add these launch logistics exactly: design review Thursday at 2 PM in Cedar, backup reviewer Omar, and status color amber.
- stress_code_sql_query t2 (code_conversation, target t5): support constraint, delta logit L2 -347.567
  content: Also require only paid orders and return u.id plus o.paid_at, ordered by newest paid_at first.
- stress_code_python_function t2 (code_conversation, target t7): support constraint, delta logit L2 -128.991
  content: Also require stable ordering and use the exact key tuple pattern with x is None and x or empty string.
- stress_longdep_one_sentence t2 (long_dependency, target t5): base memory, delta logit L2 -124.387
  content: Remember these project facts exactly: codename Atlas, owner Rina, deadline May 14, reviewer Omar.
- stress_longdep_two_lines t2 (long_dependency, target t6): base memory, delta logit L2 -74.842
  content: Remember these facts exactly: vendor northline, owner rina, due august 9, total 4820 dollars.
