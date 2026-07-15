"""Run standardization ablations on real footage and stored replicate boxes.

This is intentionally a numerical validation driver, not a GUI smoke test. It
regenerates 10-second caches with raw preprocessing and CLAHE, both with exact
pixelwise forward/backward residuals and structure-tensor evidence enabled, then
reports per-replicate scale/error/texture summaries. If manual marks exist next
to the video it also reports marked-vs-unmarked ROC AUC for raw and spatially
standardized speed scores.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import cache as cache_mod
from core.config import (Band, FeatureConfig, FlowConfig, PipelineConfig,
                         PreprocessConfig)
from core.pipeline import Progress, run_pipeline
from core.preprocess import Preprocessor
from core.replicates import build_layout, geometry_hash
from core.video import ReplicateVideoSource, VideoSource


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--layout", required=True,
                   help="JSON containing a top-level 'replicates' list")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--cache-root", default=".cache/standardization-validation")
    p.add_argument("--normalizations", nargs="+", default=["off", "clahe"],
                   choices=["off", "clahe", "zscore"])
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--report", default="screenshots/standardization_validation.json")
    p.add_argument("--boxes-image", default="screenshots/standardization_boxes.png")
    return p.parse_args()


def load_replicates(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        reps = json.load(f).get("replicates", [])
    if not reps:
        raise ValueError(f"No replicate boxes in {path}")
    return reps


def render_boxes(video: str, reps: list[dict], output: str) -> None:
    src = VideoSource(video)
    try:
        frame = src.frame_at(min(30, src.info.frame_count - 1))
    finally:
        src.release()
    if frame is None:
        raise RuntimeError("Could not decode validation frame")
    h, w = frame.shape[:2]
    for rep in reps:
        x0, y0, x1, y1 = rep["frac"]
        p0 = (int(round(x0 * w)), int(round(y0 * h)))
        p1 = (int(round(x1 * w)), int(round(y1 * h)))
        color = rep.get("color", "#00ff00").lstrip("#")
        rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(frame, p0, p1, bgr, max(2, w // 1800))
        cv2.putText(frame, rep.get("label", str(rep.get("id", "?"))),
                    (p0[0], max(24, p0[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.6, w / 5000), bgr, max(2, w // 2200), cv2.LINE_AA)
    # Full 5.3K frames are awkward to inspect interactively; retain enough detail
    # to see tube boundaries while keeping the screenshot manageable.
    if w > 1800:
        scale = 1800 / w
        frame = cv2.resize(frame, (1800, int(round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(output, frame):
        raise OSError(f"Could not write {output}")


def config(normalize: str, block: int) -> PipelineConfig:
    return PipelineConfig(
        preprocess=PreprocessConfig(normalize=normalize),
        flow=FlowConfig(backend="farneback", block_size=block),
        features=FeatureConfig(
            bands=(Band(12.0, 24.0),),
            cache_fb_error=True,
            cache_texture=True,
            dtype="float16",
            compression="zstd",
        ),
    )


def ensure_cache(video: str, normalize: str,
                 args: argparse.Namespace,
                 reps: list[dict]) -> tuple[str, float | None]:
    cfg = config(normalize, args.block)
    src = VideoSource(video)
    try:
        video_hash = src.info.video_hash
    finally:
        src.release()
    suffix = f"_stdval_{normalize}_{int(round(args.duration))}s_b{args.block}"
    key = cfg.cache_key(video_hash, geometry_hash(reps)) + suffix
    if args.regenerate and cache_mod.cache_exists(args.cache_root, key):
        cache_mod.delete_cache(args.cache_root, key)
    if cache_mod.cache_is_complete(args.cache_root, key):
        print(f"[{normalize}] reusing {key}", flush=True)
        return key, None

    last_bucket = -1

    def progress(p: Progress) -> None:
        nonlocal last_bucket
        bucket = int(p.frac * 10)
        if bucket != last_bucket:
            last_bucket = bucket
            print(f"[{normalize}] {p.stage} {p.done}/{p.total} "
                  f"({p.elapsed_s:.1f}s) {p.message}", flush=True)

    started = time.perf_counter()
    key = run_pipeline(video, cfg, args.cache_root,
                       duration_s=args.duration, progress=progress,
                       cache_key_suffix=suffix, replicates=reps)
    return key, time.perf_counter() - started


def frac_slice(frac: list[float], grid: tuple[int, int]) -> tuple[slice, slice]:
    ny, nx = grid
    x0, y0, x1, y1 = frac
    bx0 = max(0, min(nx - 1, int(round(x0 * nx))))
    bx1 = max(bx0 + 1, min(nx, int(round(x1 * nx))))
    by0 = max(0, min(ny - 1, int(round(y0 * ny))))
    by1 = max(by0 + 1, min(ny, int(round(y1 * ny))))
    return slice(by0, by1), slice(bx0, bx1)


def atlas_slice(rep_id: int, cache) -> tuple[slice, slice]:
    tile = next((t for t in cache.meta.get("replicate_tiles", [])
                 if int(t["id"]) == int(rep_id)), None)
    if tile is None:
        raise KeyError(f"Replicate {rep_id} is absent from cache layout")
    y0, x0, y1, x1 = tile["atlas_bbox"]
    return slice(y0, y1), slice(x0, x1)


def auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, bool)
    score = np.asarray(score, np.float64)
    finite = np.isfinite(score)
    y, score = y[finite], score[finite]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    # Average ranks for ties without adding scipy as a validation-only dependency.
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, np.float64)
    i = 0
    while i < order.size:
        j = i + 1
        while j < order.size and score[order[j]] == score[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) /
                 (n_pos * n_neg))


def marked_frames(video: str, n: int, fps: float) -> np.ndarray | None:
    # Use the same sidecar rule as AppState explicitly; Path.with_suffix would
    # mishandle video stems that themselves contain dots.
    path = Path(os.path.splitext(video)[0] + ".marks.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    spans = data.get("marks", {}).get("spans", {})
    out = np.zeros(n, bool)
    for roi_spans in spans.values():
        for start, end, label in roi_spans:
            if str(label).strip().lower() not in ("flying", "wingbeat"):
                continue
            i0 = max(0, int(np.floor(float(start) * fps)))
            i1 = min(n, int(np.ceil(float(end) * fps)))
            out[i0:i1] = True
    return out


def intensity_summary(video: str, normalize: str, reps: list[dict],
                      duration_s: float) -> dict:
    """Per-box intensity/contrast drift after the cache-time preprocessing."""
    src = VideoSource(video)
    try:
        info = src.info
        n = min(src.info.frame_count, max(2, int(round(duration_s * src.info.fps))))
        cfg = PreprocessConfig(normalize=normalize)
        scale = cfg.resolve_downsample(info.width)
        roi_cfg = PreprocessConfig(normalize=normalize, downsample=scale)
        layout = build_layout(reps, info.width, info.height, scale, block_size=4)
        preprocessors = {}
        for tile in layout.tiles:
            x0, y0, x1, y1 = tile.source_box
            preprocessors[tile.replicate_id] = Preprocessor(
                roi_cfg, x1 - x0, y1 - y0)
        decoder = ReplicateVideoSource(video, layout, n)
        means = [[] for _ in reps]
        contrasts = [[] for _ in reps]
        by_id = {int(r["id"]): i for i, r in enumerate(reps)}
        for _, atlas in decoder.iter_frames():
            for tile in layout.tiles:
                i = by_id[tile.replicate_id]
                region = preprocessors[tile.replicate_id].apply(
                    decoder.crop(atlas, tile.replicate_id))
                means[i].append(float(region.mean()))
                contrasts[i].append(float(region.std()))
    finally:
        if "decoder" in locals():
            decoder.release()
        src.release()

    rows = []
    for rep, mu, sd in zip(reps, means, contrasts):
        mu = np.asarray(mu, np.float32)
        sd = np.asarray(sd, np.float32)
        rows.append({
            "id": rep.get("id"),
            "label": rep.get("label"),
            "median_intensity": float(np.median(mu)),
            "median_local_contrast": float(np.median(sd)),
            "mean_intensity_step_p95": float(np.percentile(np.abs(np.diff(mu)), 95)),
        })
    med_mu = np.array([r["median_intensity"] for r in rows])
    med_sd = np.array([r["median_local_contrast"] for r in rows])
    return {
        "between_replicate_intensity_cv": float(
            med_mu.std() / max(abs(med_mu.mean()), 1e-12)),
        "between_replicate_contrast_cv": float(
            med_sd.std() / max(abs(med_sd.mean()), 1e-12)),
        "replicates": rows,
    }


def summarize(cache_root: str, key: str, reps: list[dict],
              runtime_s: float | None) -> dict:
    cache = cache_mod.open_cache(cache_root, key)
    try:
        speed = cache.read("speed").astype(np.float32)
        fb = cache.read("fb_error_p90").astype(np.float32)
        texture = cache.read("texture_min_eigen").astype(np.float32)
        per_rep = []
        for rep in reps:
            ys, xs = atlas_slice(rep["id"], cache)
            vals = speed[:, ys, xs].reshape(speed.shape[0], -1)
            sub = speed[:, ys, xs]
            med_sub = np.empty_like(sub)
            for t in range(speed.shape[0]):
                med_sub[t] = cv2.medianBlur(sub[t], 3)
            med_vals = med_sub.reshape(speed.shape[0], -1)
            spatial_p25 = np.percentile(vals, 25, axis=1)
            observed_floor = spatial_p25[1:] if spatial_p25.size > 1 else spatial_p25
            auto_noise = float(np.percentile(observed_floor, 99))
            ratio = vals / max(auto_noise, 1e-6)
            scores = {
                "raw_speed_frame_p95": np.percentile(vals, 95, axis=1),
                "median3_speed_frame_p95": np.percentile(med_vals, 95, axis=1),
                "speed_over_auto_noise_frame_p95": np.percentile(ratio, 95, axis=1),
            }
            valid_t = slice(1, None)  # frame 0 has defined-zero flow diagnostics
            rec = {
                "id": rep.get("id"),
                "label": rep.get("label"),
                "n_blocks": int(vals.shape[1]),
                "spatial_p25_speed_median_px_s": float(np.median(spatial_p25[1:])),
                "auto_noise_reference_px_s": auto_noise,
                "raw_speed_p95_median_px_s": float(np.median(scores["raw_speed_frame_p95"][1:])),
                "median3_speed_p95_median_px_s": float(np.median(scores["median3_speed_frame_p95"][1:])),
                "auto_noise_ratio_p95_median": float(np.median(
                    scores["speed_over_auto_noise_frame_p95"][1:])),
                "fb_error_p90_block_median_work_px_frame": float(np.median(
                    fb[valid_t, ys, xs])),
                "fb_error_p90_block_p95_work_px_frame": float(np.percentile(
                    fb[valid_t, ys, xs], 95)),
                "texture_median": float(np.median(texture[:, ys, xs])),
            }
            marks = marked_frames(cache.meta["video_path"], speed.shape[0], cache.fps)
            if marks is not None:
                rec["marked_fraction"] = float(marks.mean())
                rec["marked_auc"] = {name: auc(marks, values)
                                     for name, values in scores.items()}
            per_rep.append(rec)

        floors = np.array([r["spatial_p25_speed_median_px_s"] for r in per_rep])
        return {
            "key": key,
            "normalization": cache.meta.get("config", {}).get("preprocess", {}).get(
                "normalize"),
            "runtime_s": runtime_s,
            "cache_reused": runtime_s is None,
            "size_bytes": cache.size_on_disk(),
            "working_resolution": [cache.meta["work_width"], cache.meta["work_height"]],
            "grid": list(cache.grid),
            "between_replicate_background_cv": float(
                floors.std() / max(floors.mean(), 1e-12)),
            "replicates": per_rep,
        }
    finally:
        cache.close()


def main() -> None:
    args = parse_args()
    video = os.path.abspath(args.video)
    source = VideoSource(video)
    try:
        source_duration_s = source.info.duration_s
    finally:
        source.release()
    reps = load_replicates(args.layout)
    render_boxes(video, reps, args.boxes_image)
    results = []
    for normalize in args.normalizations:
        key, runtime_s = ensure_cache(video, normalize, args, reps)
        result = summarize(args.cache_root, key, reps, runtime_s)
        result["preprocessed_intensity"] = intensity_summary(
            video, normalize, reps, args.duration)
        results.append(result)
    report = {
        "video": video,
        "source_duration_s": source_duration_s,
        "analysis_duration_s": args.duration,
        "block_size": args.block,
        "layout": os.path.abspath(args.layout),
        "variants": results,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
