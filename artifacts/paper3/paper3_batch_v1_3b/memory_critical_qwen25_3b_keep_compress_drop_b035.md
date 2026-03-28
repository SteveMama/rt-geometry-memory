# Paper 3 Memory-Critical Analysis: qwen25_3b | geometry_keep_compress_drop @ budget 0.35

This report compares the selected Paper 3 policy against uniform on support-turn retention.

## Aggregate

- Cases compared: 36
- geometry_keep_compress_drop retains more support user turns than uniform: 14 cases
- Uniform retains more support user turns than geometry_keep_compress_drop: 6 cases
- Ties: 16
- geometry_keep_compress_drop keeps the latest support turn while uniform drops it: 13 cases
- Cases with nonzero compressed segments: 29
- Compressed cases that are not worse than uniform on support retention: 23

## Rescued Support Types

- support constraint: 9
- base memory: 5

## Top Rescued Support Turns

- stress_longdep_one_sentence t2 (long_dependency, target t5): base memory, delta logit L2 -570.058, compressed segments 0
  content: Remember these project facts exactly: codename Atlas, owner Rina, deadline May 14, reviewer Omar.
- stress_code_js_fetch_helper t2 (code_conversation, target t5): support constraint, delta logit L2 -496.995, compressed segments 0
  content: Also require GET /users/${id}, throw on non-ok responses, and return parsed JSON.
- stress_retrieval_launch_packet t2 (retrieval_heavy, target t7): support constraint, delta logit L2 -432.704, compressed segments 1
  content: Add these launch logistics exactly: design review Thursday at 2 PM in Cedar, backup reviewer Omar, and status color amber.
- stress_retrieval_launch_packet t2 (retrieval_heavy, target t5): support constraint, delta logit L2 -179.225, compressed segments 0
  content: Add these launch logistics exactly: design review Thursday at 2 PM in Cedar, backup reviewer Omar, and status color amber.
- stress_code_sql_query t2 (code_conversation, target t7): support constraint, delta logit L2 -160.909, compressed segments 1
  content: Also require only paid orders and return u.id plus o.paid_at, ordered by newest paid_at first.
- stress_longdep_two_lines t2 (long_dependency, target t6): base memory, delta logit L2 -130.623, compressed segments 1
  content: Remember these facts exactly: vendor northline, owner rina, due august 9, total 4820 dollars.
- stress_longdep_csv_items t2 (long_dependency, target t5): base memory, delta logit L2 -129.724, compressed segments 0
  content: Remember these event details exactly: venue Maple Hall, start 6 PM, host Jordan, backup host Nia.
- stress_longdep_two_lines t2 (long_dependency, target t5): base memory, delta logit L2 -95.654, compressed segments 0
  content: Remember these facts exactly: vendor northline, owner rina, due august 9, total 4820 dollars.
