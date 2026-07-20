# Shelved GUI surfaces

Nothing here is imported by the running app. These modules are kept as design
records — the interaction loops they encode were built and used, and the reasons
they were retired are specific rather than "it didn't work". Read them before
rebuilding anything in the same space.

Do not fix these if an upstream change breaks them. Note the breakage and move
on. If one is revived, treat it as a rewrite against the current `AppState`, not
a restoration.

| File | Was | Retired because |
| --- | --- | --- |
| `tab1_flow.py` | "3 · Flow cache (commit)" — configured a whole-video flow pass and wrote the per-replicate cache the Behavior tab consumed | Superseded by the tensor path. Detection no longer reads a flow cache; live tuning happens in `gui/tab_live_preprocess.py` with no flow solve. The commit step it performed is being rebuilt as a whole-video detection pass. |
| `tab3_behavior.py` | "4 · Behavior Classification" — the tree editor (flat AND of feature range constraints, live ethogram + video overlay) | Consumed the flow cache that no longer backs detection. The classification model itself (`core/behavior.py`) is untouched; only this editor is shelved. |
| `tab3_behavior_v1.py` | The earlier flat-constraint version of the same tab, shelved when the tree editor replaced it | Predecessor of the above. Kept for its spatial-criteria handling, which the tree editor dropped. |
| `tab2_roi_auto.py` | "Tab 2: ROI discovery" — automatic ROI proposal by cross-filtered histogram intersection | It worked, but on footage where the behavior is not the dominant motion it proposed the wrong regions. Replaced by hand-drawn replicate boxes (`gui/tab2_replicates.py`). |

`tab1_flow.py` also held `_test_cache_suffix`, the human-readable settings
snapshot appended to test-run cache names. Its unit test
(`tests/test_cache_naming.py`) was deleted with the tab rather than kept
importing shelved code. The authoritative cache key was always the config hash
from `PipelineConfig.cache_key`; the suffix was only ever for humans reading the
cache picker, so nothing about cache identity depends on this code.

These modules still import live helpers — `gui/histogram_widget.py`,
`gui/inspector.py`, `gui/timeline.py`, `gui/help.py`, `gui/video_panel.py`.
Several of those now have no other consumer.
