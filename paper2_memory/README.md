# Paper 2 Memory

This package contains the Paper 2 geometry-aware memory-control experiments.

Core responsibilities:

- budgeted retention and segment-action policies
- hard stress-set evaluation
- answer-level behavior scoring
- mechanism analysis for memory-critical support turns

Main entry points:

- [`run_paper2.py`](/Users/pranav/Documents/RT/paper2_memory/run_paper2.py)
- [`study.py`](/Users/pranav/Documents/RT/paper2_memory/study.py)
- [`case_analysis.py`](/Users/pranav/Documents/RT/paper2_memory/case_analysis.py)
- [`memory_critical_analysis.py`](/Users/pranav/Documents/RT/paper2_memory/memory_critical_analysis.py)
- [`cross_model_memory_summary.py`](/Users/pranav/Documents/RT/paper2_memory/cross_model_memory_summary.py)

Current checkpoint:

- main result: geometry-aware control beats uniform on the hard stress set
- mechanism result: geometry preserves memory-critical support turns more often than uniform
- tracked summaries live under [`artifacts/paper2`](/Users/pranav/Documents/RT/artifacts/paper2)
