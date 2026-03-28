# Paper 2 Case Analysis: qwen25_05b @ budget 0.35

## Top Geometry Wins

- stress_retrieval_launch_packet target_turn=5: delta logit L2 -733.158, delta answer avg NLL 0.0000
- stress_code_js_fetch_helper target_turn=5: delta logit L2 -521.185, delta answer avg NLL 0.0000
- stress_code_python_function target_turn=5: delta logit L2 -433.521, delta answer avg NLL 0.0000
- stress_retrieval_launch_packet target_turn=7: delta logit L2 -399.785, delta answer avg NLL 0.0000
- stress_code_sql_query target_turn=5: delta logit L2 -347.567, delta answer avg NLL 0.0000

## Case 1: stress_retrieval_launch_packet

- Family: retrieval_heavy
- Target turn: 5
- Geometry vs uniform logit delta: -733.158
- Geometry vs uniform answer avg NLL delta: 0.0000
- Uniform retained turns: [1, 3, 4]
- Geometry retained turns: [2, 3, 4]
- Geometry-only retained turns: [2]

### Prefix Turns

[--] t0 user: Remember this launch packet exactly: codename Atlas, owner Rina, launch date June 3, and legal review due May 20.
[U-] t1 assistant: Stored: Atlas, owner Rina, launch June 3, legal review May 20.
[-G] t2 user: Add these launch logistics exactly: design review Thursday at 2 PM in Cedar, backup reviewer Omar, and status color amber.
[UG] t3 assistant: Added: design review Thursday at 2 PM in Cedar, backup reviewer Omar, status amber.
[UG] t4 user: Now give me exactly owner | launch date | review room.

### Query And Gold Answer

user t5: Rina | June 3 | Cedar
assistant t6: Now give me exactly codename | legal due | review time.

## Case 2: stress_code_js_fetch_helper

- Family: code_conversation
- Target turn: 5
- Geometry vs uniform logit delta: -521.185
- Geometry vs uniform answer avg NLL delta: 0.0000
- Uniform retained turns: [1, 3, 4]
- Geometry retained turns: [2, 3, 4]
- Geometry-only retained turns: [2]

### Prefix Turns

[--] t0 user: We are working in JavaScript. The function name must be loadUser, the auth token variable is authToken, and the base URL variable is apiBase.
[U-] t1 assistant: Stored: JavaScript, loadUser, authToken, apiBase.
[-G] t2 user: Also require GET /users/${id}, throw on non-ok responses, and return parsed JSON.
[UG] t3 assistant: Added: GET /users/${id}, throw on !ok, return JSON.
[UG] t4 user: Show only the fetch line.

### Query And Gold Answer

user t5: const res = await fetch(`${apiBase}/users/${id}`, { headers: { Authorization: `Bearer ${authToken}` } });
assistant t6: Now show the full async function in exactly five lines.

## Case 3: stress_code_python_function

- Family: code_conversation
- Target turn: 5
- Geometry vs uniform logit delta: -433.521
- Geometry vs uniform answer avg NLL delta: 0.0000
- Uniform retained turns: [1, 3, 4]
- Geometry retained turns: [2, 3, 4]
- Geometry-only retained turns: [2]

### Prefix Turns

[--] t0 user: We are working in Python. The function name must be clean_rows, the input variable is rows, and None values must sort last without mutating the input.
[U-] t1 assistant: Stored: Python, function clean_rows, input rows, None last, no mutation.
[-G] t2 user: Also require stable ordering and use the exact key tuple pattern with x is None and x or empty string.
[UG] t3 assistant: Added: stable ordering and the exact tuple key pattern.
[UG] t4 user: Show only the return statement.

### Query And Gold Answer

user t5: return sorted(rows, key=lambda x: (x is None, x or ""))
assistant t6: Now show the full function in exactly three lines.

## Case 4: stress_retrieval_launch_packet

- Family: retrieval_heavy
- Target turn: 7
- Geometry vs uniform logit delta: -399.785
- Geometry vs uniform answer avg NLL delta: 0.0000
- Uniform retained turns: [1, 3, 5, 6]
- Geometry retained turns: [2, 4, 5, 6]
- Geometry-only retained turns: [2, 4]

### Prefix Turns

[--] t0 user: Remember this launch packet exactly: codename Atlas, owner Rina, launch date June 3, and legal review due May 20.
[U-] t1 assistant: Stored: Atlas, owner Rina, launch June 3, legal review May 20.
[-G] t2 user: Add these launch logistics exactly: design review Thursday at 2 PM in Cedar, backup reviewer Omar, and status color amber.
[U-] t3 assistant: Added: design review Thursday at 2 PM in Cedar, backup reviewer Omar, status amber.
[-G] t4 user: Now give me exactly owner | launch date | review room.
[UG] t5 assistant: Rina | June 3 | Cedar
[UG] t6 user: Now give me exactly codename | legal due | review time.

### Query And Gold Answer

user t7: Atlas | May 20 | Thursday 2 PM

## Case 5: stress_code_sql_query

- Family: code_conversation
- Target turn: 5
- Geometry vs uniform logit delta: -347.567
- Geometry vs uniform answer avg NLL delta: 0.0000
- Uniform retained turns: [1, 3, 4]
- Geometry retained turns: [2, 3, 4]
- Geometry-only retained turns: [2]

### Prefix Turns

[--] t0 user: We are working in SQL. Use users u and orders o, join on u.id = o.user_id, and only include active users.
[U-] t1 assistant: Stored: SQL, users u, orders o, join u.id = o.user_id, active users only.
[-G] t2 user: Also require only paid orders and return u.id plus o.paid_at, ordered by newest paid_at first.
[UG] t3 assistant: Added: only paid orders, return u.id and o.paid_at, newest first.
[UG] t4 user: Show only the JOIN line.

### Query And Gold Answer

user t5: JOIN orders o ON u.id = o.user_id
assistant t6: Now show the full query in exactly five lines.
