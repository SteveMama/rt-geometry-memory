# Paper 3 Codec

This package contains the current Paper 3 pilot code.

Core responsibilities:

- sparse keep/compress/drop memory policy
- segment-action baseline comparison
- behavior evaluation on the hard stress set
- confidence, significance, and mechanism analysis

Main entry points:

- [`run_paper3.py`](/Users/pranav/Documents/RT/paper3_codec/run_paper3.py)
- [`study.py`](/Users/pranav/Documents/RT/paper3_codec/study.py)
- [`policies.py`](/Users/pranav/Documents/RT/paper3_codec/policies.py)
- [`memory_critical_analysis.py`](/Users/pranav/Documents/RT/paper3_codec/memory_critical_analysis.py)

Current checkpoint:

- `geometry_keep_compress_drop` is now budget-responsive and genuinely uses compression
- `geometry_segment_actions` remains a strong competing family
- next experiment track: cross-model head-to-head plus behavior/significance and artifact publication from Colab
- the latest tracked Paper 3 checkpoint is [`artifacts/paper3/paper3_pilot_v3_full`](/Users/pranav/Documents/RT/artifacts/paper3/paper3_pilot_v3_full)
