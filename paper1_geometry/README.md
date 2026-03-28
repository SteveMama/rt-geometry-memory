# Paper 1 Geometry

This package contains the Paper 1 measurement and characterization stack.

Core responsibilities:

- extract turn-level hidden-state summaries
- normalize conversation states
- analyze low-rank transported increments
- measure decoder relevance through logit and KL drift
- evaluate boundary structure and baselines

Main entry points:

- [`run_paper1.py`](/Users/pranav/Documents/RT/paper1_geometry/run_paper1.py)
- [`study.py`](/Users/pranav/Documents/RT/paper1_geometry/study.py)
- [`analysis.py`](/Users/pranav/Documents/RT/paper1_geometry/analysis.py)
- [`geometry.py`](/Users/pranav/Documents/RT/paper1_geometry/geometry.py)

Current checkpoint:

- Paper 1 is frozen around the `expanded_v8_final` evidence bundle
- tracked summaries and plots live under [`artifacts/paper1/expanded_v8_final`](/Users/pranav/Documents/RT/artifacts/paper1/expanded_v8_final)
