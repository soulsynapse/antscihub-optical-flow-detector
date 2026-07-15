"""Export: per-ROI feature time series, per-behavior binary traces, and the
summary table.

Everything is in seconds. Frame indices appear in exactly one column, labelled as
such, for people who need to cross-reference the raw video.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from core.behavior import Behavior, trace_to_bouts
from core.roi import ROI, roi_time_series


def export_roi_timeseries(path: str, cache, ctx, rois: list[ROI],
                          features: list[str],
                          behaviors: list[Behavior] | None = None,
                          traces: dict[tuple[int, str], np.ndarray] | None = None
                          ) -> None:
    """Long-format CSV: one row per (ROI, frame), one column per feature, plus a
    boolean column per behavior.

    Long format rather than one file per ROI because it drops straight into R or
    pandas without a join, and this is going into someone's analysis, not a
    database.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fps = ctx.fps
    behaviors = behaviors or []
    traces = traces or {}

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi_id", "time_s", "frame_idx"] + features +
                   [f"behavior_{b.name}" for b in behaviors])
        for roi in rois:
            series = {fe: roi_time_series(cache, ctx, roi, fe) for fe in features}
            n = ctx.n_frames
            beh_cols = []
            for b in behaviors:
                tr = traces.get((roi.roi_id, b.name))
                beh_cols.append(tr if tr is not None else np.zeros(n, bool))
            for t in range(n):
                w.writerow(
                    [roi.roi_id, f"{t / fps:.4f}", t] +
                    [f"{series[fe][t]:.5g}" for fe in features] +
                    [int(bool(c[t])) for c in beh_cols]
                )


def export_summary(path: str, rois: list[ROI], behaviors: list[Behavior],
                   traces: dict[tuple[int, str], np.ndarray], fps: float) -> None:
    """One row per (ROI x behavior): total time, bout count, mean bout duration."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi_id", "behavior", "total_time_s", "n_bouts",
                    "mean_bout_s", "median_bout_s", "longest_bout_s",
                    "first_onset_s", "roi_duration_s", "fraction_of_roi"])
        for roi in rois:
            roi_dur = roi.duration_s(fps)
            for b in behaviors:
                tr = traces.get((roi.roi_id, b.name))
                if tr is None:
                    continue
                bouts = trace_to_bouts(tr, fps)
                durs = [bt.duration_s for bt in bouts]
                total = float(np.sum(durs))
                w.writerow([
                    roi.roi_id, b.name,
                    f"{total:.3f}", len(bouts),
                    f"{np.mean(durs):.3f}" if durs else "0",
                    f"{np.median(durs):.3f}" if durs else "0",
                    f"{np.max(durs):.3f}" if durs else "0",
                    f"{bouts[0].start_s:.3f}" if bouts else "",
                    f"{roi_dur:.3f}",
                    f"{total / roi_dur:.4f}" if roi_dur > 0 else "0",
                ])


def export_bouts(path: str, rois: list[ROI], behaviors: list[Behavior],
                 traces: dict[tuple[int, str], np.ndarray], fps: float) -> None:
    """One row per bout. This is usually the table people actually want."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi_id", "behavior", "bout_index", "start_s", "end_s",
                    "duration_s"])
        for roi in rois:
            for b in behaviors:
                tr = traces.get((roi.roi_id, b.name))
                if tr is None:
                    continue
                for i, bt in enumerate(trace_to_bouts(tr, fps)):
                    w.writerow([roi.roi_id, b.name, i,
                                f"{bt.start_s:.3f}", f"{bt.end_s:.3f}",
                                f"{bt.duration_s:.3f}"])


def export_hdf5(path: str, cache, ctx, rois: list[ROI], features: list[str],
                behaviors: list[Behavior],
                traces: dict[tuple[int, str], np.ndarray]) -> None:
    """Same content as the CSVs but as HDF5, for clips where the long CSV would
    be unwieldy (30k frames x 50 ROIs is 1.5M rows)."""
    import h5py

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["fps"] = ctx.fps
        f.attrs["n_frames"] = ctx.n_frames
        f.attrs["block_size"] = ctx.block_size
        f.create_dataset("time_s", data=ctx.times_s())

        for roi in rois:
            g = f.create_group(f"roi_{roi.roi_id:03d}")
            g.attrs["bbox_blocks"] = np.array(roi.bbox)
            if roi.source_frac is not None:
                g.attrs["source_frac"] = np.array(roi.source_frac)
            g.attrs["note"] = roi.note
            for name in ("baseline_start_s", "baseline_end_s",
                         "pixels_per_mm", "body_length_mm"):
                value = getattr(roi, name, None)
                if value is not None:
                    g.attrs[name] = float(value)
            g.create_dataset("mask", data=roi.mask.astype(np.uint8),
                             compression="gzip")
            for fe in features:
                g.create_dataset(fe, data=roi_time_series(cache, ctx, roi, fe),
                                 compression="gzip")
            bg = g.create_group("behaviors")
            for b in behaviors:
                tr = traces.get((roi.roi_id, b.name))
                if tr is not None:
                    bg.create_dataset(b.name, data=tr.astype(np.uint8),
                                      compression="gzip")
