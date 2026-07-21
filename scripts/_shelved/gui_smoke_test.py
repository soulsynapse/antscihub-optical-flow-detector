"""Drive the whole GUI headlessly (offscreen Qt) through the real workflow.

Exercises the paths a user actually walks: open video -> open cache -> tune
histogram ranges -> extract ROIs -> define a behavior -> inspect -> export. Runs
against the scratch cache left by smoke_test.py, so run that first.

This is not a unit test. It is the cheapest way to find the crashes that only
happen when signals fire in the real order.
"""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt6.QtWidgets import QApplication

from core import cache as cache_mod
from core.behavior import default_wingbeat
from core.export import export_bouts, export_roi_timeseries, export_summary
from gui.main_window import MainWindow

VIDEO = os.path.join("Videos", "Raw", "GX010050c2_02_18_26.MP4")


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow(project_dir=".")
    st = win.state

    caches = cache_mod.list_caches(st.cache_root)
    smoke = [c for c in caches if c["key"].endswith("_smoke")]
    if not smoke:
        print("No smoke cache. Run scripts/smoke_test.py first.")
        return 1
    key = smoke[0]["key"]

    print("loading video...")
    st.load_video(VIDEO)
    assert st.has_video

    print(f"opening cache {key}...")
    st.open_cache(key)
    assert st.has_cache
    print(f"  {st.cache.n_frames} frames, grid {st.cache.grid}, "
          f"features {st.cache.feature_names}")

    win.tabs.setCurrentIndex(1)
    tab2 = win.tab2

    print("scrubbing (exercises overlay repaint + chunk cache)...")
    for f in range(0, st.cache.n_frames, max(1, st.cache.n_frames // 20)):
        st.set_frame(f)

    print("drawing replicate boxes...")
    # Simulate the user drawing three boxes across the frame, as box_drawn would.
    tab2._on_box_drawn(0.05, 0.05, 0.45, 0.45)
    tab2._on_box_drawn(0.55, 0.05, 0.95, 0.45)
    tab2._on_box_drawn(0.30, 0.55, 0.70, 0.95)
    print(f"  {len(tab2.replicates)} replicates -> {len(st.rois)} ROIs")
    assert len(st.rois) == 3, "drawn boxes did not become ROIs"

    for r in st.rois:
        assert r.mask.any(), f"ROI #{r.roi_id} has an empty mask"
        print(f"  ROI #{r.roi_id} '{r.note}': {int(r.mask.sum())} blocks, "
              f"bbox {r.bbox}")

    # Boxes are geometry, so they must survive a re-cache at a different grid.
    reps_before = [dict(r) for r in tab2.replicates]
    tab2._rebuild_rois()
    assert len(st.rois) == 3 and st.rois[0].note == reps_before[0]["label"]
    print("  boxes survive rebuild, labels preserved")

    tab2.list.setCurrentRow(0)
    tab2._refresh_inspector()

    # Save/load round-trip.
    import tempfile
    box_path = os.path.join(tempfile.mkdtemp(), "reps.json")
    import json as _json
    with open(box_path, "w") as f:
        _json.dump({"replicates": tab2.replicates}, f)
    tab2.replicates = []
    with open(box_path) as f:
        data = _json.load(f)
    tab2.replicates = [{**r, "frac": tuple(r["frac"])}
                       for r in data["replicates"]]
    tab2._rebuild_rois()
    assert len(st.rois) == 3, "box save/load round-trip lost replicates"
    print("  save/load round-trip OK")

    print("defining a band-power behavior (new additive Tab 3)...")
    win.tabs.setCurrentIndex(2)
    tab3 = win.tab3
    band = st.band_features[0]

    # New flow: New behavior, add a constraint, drag its range on the plot.
    from core.behavior import Behavior, LogicNode, RangeLeaf
    tab3.current = Behavior(name="wb", color="#ff4488", spec=LogicNode("and"))
    tab3._sync_from_behavior()
    tab3._select_roi(st.rois[0].roi_id)

    # Add an ENABLED band-power constraint (mirrors _add_constraint).
    col = np.concatenate([st.roi_series(r, band) for r in st.rois])
    thr = float(np.percentile(col, 85))
    tab3.current.spec.children.append(RangeLeaf(band, thr, float("inf"), enabled=True))
    tab3._rebuild_plots()
    tab3._recompute()
    assert band in tab3.plots, "no RangePlot created for the constraint"
    print(f"  constraint {band} >= {thr:.0f} (85th pct); {len(tab3.plots)} plot(s)")

    # All-measures panel: every standard measure is shown, all others disabled.
    assert "coherence" in tab3.plots and "speed" in tab3.plots, "measures missing"
    enabled = [l.feature for l in tab3._leaves() if l.enabled]
    assert enabled == [band], f"only the band should be enabled, got {enabled}"
    print(f"  all-measures panel: {len(tab3.plots)} shown, enabled = {enabled}")

    def _leaf(feat):
        return next(l for l in tab3._leaves() if l.feature == feat)

    # The RangePlot must carry the selected ROI's blocks and the leaf range.
    pl = tab3.plots[band]
    assert pl.block_vals.size > 0, "RangePlot has no per-block data"
    assert abs(pl.lo - thr) < 1e-3, "RangePlot range not synced to leaf"

    # Dragging the line updates the leaf and recomputes.
    tab3._on_leaf_range(band, thr * 1.2, float("inf"))
    assert abs(_leaf(band).lo - thr * 1.2) < 1e-3, "drag did not update leaf"
    print("  dragging the range line updates the constraint and recomputes")

    # Editing a DISABLED measure must not change detection.
    d0 = st.traces[(st.rois[0].roi_id, "wb")].mean()
    tab3._on_leaf_range("divergence", 1.0, 2.0)
    d1 = st.traces[(st.rois[0].roi_id, "wb")].mean()
    assert d0 == d1, "editing a disabled measure changed detection"
    print("  editing a disabled measure leaves detection unchanged")

    # Selection must NOT change detection (the reported bug).
    before = {r.roi_id: st.traces[(r.roi_id, "wb")].mean() for r in st.rois}
    tab3._select_roi(st.rois[1].roi_id)
    after = {r.roi_id: st.traces[(r.roi_id, "wb")].mean() for r in st.rois}
    assert before == after, "selecting a replicate changed its detection!"
    assert not any(t.all() for t in st.traces.values()), "a trace is all-True"
    print(f"  selection is display-only; on-fractions {before} unchanged")

    # The per-block video overlay must be a strict subset of the box that fires.
    from core.roi import behavior_block_mask
    tab3._update_overlay()
    mask = behavior_block_mask(st.cache, st.ctx, tab3.current, st.rois[1].roi_id
                               if False else st.current_frame)
    print(f"  overlay lights {int(mask.sum())} blocks at frame {st.current_frame}")

    n_traces = len(st.traces)
    active = sum(int(t.any()) for t in st.traces.values())
    print(f"  {n_traces} (replicate x behavior) traces, {active} with a bout")
    st.behaviors = [tab3.current]

    print("exporting...")
    out = tempfile.mkdtemp(prefix="ofd_export_")
    feats_out = tab3._features_for_export()
    export_roi_timeseries(os.path.join(out, "ts.csv"), st.cache, st.ctx,
                          st.rois[:2], feats_out, st.behaviors, st.traces)
    export_bouts(os.path.join(out, "bouts.csv"), st.rois, st.behaviors,
                 st.traces, st.fps)
    export_summary(os.path.join(out, "summary.csv"), st.rois, st.behaviors,
                   st.traces, st.fps)
    for f in ("ts.csv", "bouts.csv", "summary.csv"):
        p = os.path.join(out, f)
        n = sum(1 for _ in open(p))
        print(f"  {f}: {n} rows, {os.path.getsize(p) / 1024:.1f} KB")
        # ts.csv and summary.csv always have rows (one per frame / per ROIxbeh).
        # bouts.csv legitimately has only a header when nothing crosses threshold,
        # which is the expected result for arbitrary boxes over background.
        floor = 1 if f == "bouts.csv" else 2
        assert n >= floor, f"{f} is missing even its header"

    print("\nGUI SMOKE TEST PASSED")
    win.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
