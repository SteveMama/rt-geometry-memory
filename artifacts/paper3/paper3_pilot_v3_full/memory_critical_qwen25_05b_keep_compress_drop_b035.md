# Paper 3 Memory-Critical Analysis: qwen25_05b | geometry_keep_compress_drop @ budget 0.35

This report compares the selected Paper 3 policy against uniform on support-turn retention.

## Aggregate

- Cases compared: 36
- geometry_keep_compress_drop retains more support user turns than uniform: 14 cases
- Uniform retains more support user turns than geometry_keep_compress_drop: 4 cases
- Ties: 18
- geometry_keep_compress_drop keeps the latest support turn while uniform drops it: 15 cases
- Cases with nonzero compressed segments: 29
- Compressed cases that are not worse than uniform on support retention: 25

## Rescued Support Types

- support constraint: 11
- base memory: 5

## Top Rescued Support Turns

- stress_retrieval_launch_packet t2 (retrieval_heavy, target t5): support constraint, delta logit L2 -726.129, compressed segments 0
  content: Add these launch logistics exactly: design review Thursday at 2 PM in Cedar, backup reviewer Omar, and status color amber.
- stress_code_js_fetch_helper t2 (code_conversation, target t5): support constraint, delta logit L2 -527.194, compressed segments 0
  content: Also require GET /users/${id}, throw on non-ok responses, and return parsed JSON.
- stress_code_python_function t2 (code_conversation, target t5): support constraint, delta logit L2 -441.910, compressed segments 0
  content: Also require stable ordering and use the exact key tuple pattern with x is None and x or empty string.
- stress_retrieval_launch_packet t2 (retrieval_heavy, target t7): support constraint, delta logit L2 -406.606, compressed segments 1
  content: Add these launch logistics exactly: design review Thursday at 2 PM in Cedar, backup reviewer Omar, and status color amber.
- stress_code_sql_query t2 (code_conversation, target t5): support constraint, delta logit L2 -343.322, compressed segments 0
  content: Also require only paid orders and return u.id plus o.paid_at, ordered by newest paid_at first.
- stress_longdep_csv_items t2 (long_dependency, target t5): base memory, delta logit L2 -255.420, compressed segments 0
  content: Remember these event details exactly: venue Maple Hall, start 6 PM, host Jordan, backup host Nia.
- stress_longdep_one_sentence t2 (long_dependency, target t5): base memory, delta logit L2 -139.191, compressed segments 0
  content: Remember these project facts exactly: codename Atlas, owner Rina, deadline May 14, reviewer Omar.
- stress_code_python_function t2 (code_conversation, target t7): support constraint, delta logit L2 -122.676, compressed segments 1
  content: Also require stable ordering and use the exact key tuple pattern with x is None and x or empty string.
