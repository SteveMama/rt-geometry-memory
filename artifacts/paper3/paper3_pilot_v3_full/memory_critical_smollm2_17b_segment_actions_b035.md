# Paper 3 Memory-Critical Analysis: smollm2_17b | geometry_segment_actions @ budget 0.35

This report compares the selected Paper 3 policy against uniform on support-turn retention.

## Aggregate

- Cases compared: 36
- geometry_segment_actions retains more support user turns than uniform: 8 cases
- Uniform retains more support user turns than geometry_segment_actions: 8 cases
- Ties: 20
- geometry_segment_actions keeps the latest support turn while uniform drops it: 7 cases
- Cases with nonzero compressed segments: 27
- Compressed cases that are not worse than uniform on support retention: 19

## Rescued Support Types

- support constraint: 5
- base memory: 3

## Top Rescued Support Turns

- stress_longdep_csv_items t2 (long_dependency, target t5): base memory, delta logit L2 -121.406, compressed segments 0
  content: Remember these event details exactly: venue Maple Hall, start 6 PM, host Jordan, backup host Nia.
- stress_longdep_one_sentence t2 (long_dependency, target t5): base memory, delta logit L2 -121.079, compressed segments 0
  content: Remember these project facts exactly: codename Atlas, owner Rina, deadline May 14, reviewer Omar.
- stress_longdep_two_lines t2 (long_dependency, target t5): base memory, delta logit L2 -54.565, compressed segments 0
  content: Remember these facts exactly: vendor northline, owner rina, due august 9, total 4820 dollars.
- stress_code_python_function t2 (code_conversation, target t5): support constraint, delta logit L2 -51.596, compressed segments 0
  content: Also require stable ordering and use the exact key tuple pattern with x is None and x or empty string.
- stress_code_js_fetch_helper t2 (code_conversation, target t5): support constraint, delta logit L2 -50.053, compressed segments 0
  content: Also require GET /users/${id}, throw on non-ok responses, and return parsed JSON.
- stress_code_python_function t2 (code_conversation, target t6): support constraint, delta logit L2 -43.304, compressed segments 0
  content: Also require stable ordering and use the exact key tuple pattern with x is None and x or empty string.
- stress_code_sql_query t2 (code_conversation, target t5): support constraint, delta logit L2 -39.971, compressed segments 0
  content: Also require only paid orders and return u.id plus o.paid_at, ordered by newest paid_at first.
- stress_code_python_function t2 (code_conversation, target t7): support constraint, delta logit L2 -34.295, compressed segments 0
  content: Also require stable ordering and use the exact key tuple pattern with x is None and x or empty string.
