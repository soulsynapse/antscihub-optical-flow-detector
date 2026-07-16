"""Proof of concept: the 3D structure tensor as one representation from which
motion, texture and temporal change all fall out -- plus amplitude variance from
a separately tracked mean intensity (the DC term the tensor discards).

It runs on a SYNTHETIC scene by default, built so each phenomenon is isolated in
its own quadrant, which makes the claim falsifiable at a glance:

    top-left     TRANSLATING TEXTURE   -> flow fires, change modest, var modest
    top-right    FLICKERING BLOB       -> change (J_tt) fires, flow ~0
    bottom-left  STATIC TEXTURE        -> texture fires, flow ~0, change ~0
    bottom-right SLOW BRIGHTNESS RAMP  -> amplitude variance fires, J_tt ~0

The empirical punchline is printed as a table (each channel should peak in its
own quadrant) plus the Pearson correlation between tensor-derived flow speed and
an independent Farneback solve on the same frames -- evidence that "flow is the
tensor", not a second reimplementation.

    python scripts/structure_tensor_poc.py                 # synthetic
    python scripts/structure_tensor_poc.py --video PATH     # real footage
    python scripts/structure_tensor_poc.py --seconds 4 --out montage.png
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import FlowConfig
from core.flow import FarnebackBackend, reduce_scalar_to_blocks, reduce_to_blocks
from core.structure_tensor import (COMPONENTS, tensor_products, flow_from_tensor,
                                    spatial_min_eigen, temporal_energy,
                                    flow_residual, brightness_residual_field)


# -- synthetic scene ---------------------------------------------------------

def build_synthetic(size=256, T=90, drift_px_per_frame=0.5, flicker_hz=8.0,
                    fps=60.0, seed=0):
    """Four quadrants, each isolating one channel. Returns (frames, half).

    The flicker quadrant is TEXTURED structure fading in and out in place (an
    appearance/occlusion signal, like a backlit wing) -- deliberately NOT a
    smooth pulsing blob, because a smoothly scaling blob is locally
    indistinguishable from a diverging motion field and LK "explains" it as
    motion, leaving no residual. Appearing texture has no source position, so no
    displacement explains it: that is what the residual channel must isolate.
    """
    rng = np.random.default_rng(seed)
    h = size // 2
    # A wide random texture we can roll to translate without wrap artifacts in view.
    tex = rng.integers(40, 216, size=(h, h + T * 4)).astype(np.float32)
    static_tex = rng.integers(40, 216, size=(h, h)).astype(np.float32)
    fade_tex = rng.integers(40, 216, size=(h, h)).astype(np.float32)

    frames = []
    for t in range(T):
        f = np.full((size, size), 128.0, np.float32)
        shift = int(round(t * drift_px_per_frame))
        env = 0.5 * (1 + np.sin(2 * np.pi * flicker_hz * t / fps))  # 0..1
        f[:h, :h] = tex[:, shift:shift + h]                       # translating
        f[:h, h:] = 128 + (fade_tex - 128) * env                  # appearing texture
        f[h:, :h] = static_tex                                     # static texture
        f[h:, h:] = 60 + 120 * (t / max(1, T - 1))                # slow ramp
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return frames, h


# -- real footage ------------------------------------------------------------

def load_video_frames(path, seconds, target_w=384):
    from core.video import VideoSource
    src = VideoSource(path)
    fps = float(src.info.fps)
    n = min(src.info.frame_count, max(2, int(round(seconds * fps))))
    scale = target_w / max(1, src.info.width)
    frames = []
    for _, bgr in src.iter_frames(0, n):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(g)
    src.release()
    return frames, fps


# -- the extraction ----------------------------------------------------------

def run(frames, block, fps, sigma=2.0):
    """Accumulate the four channel reads.

    The structure tensor is smoothed at a SMALL spatial scale (``sigma`` px) and
    every read is solved PER PIXEL, then block-reduced. Solving one LK per whole
    block instead couples the aperture problem to the block size and leaves
    spurious residual on textured motion -- the smoothing window is the tensor's
    integration scale and must be a few pixels, not the whole block.
    """
    T = len(frames)
    h, w = frames[0].shape
    ny, nx = h // block, w // block
    fb = FarnebackBackend(FlowConfig())

    def smooth(p):
        return cv2.GaussianBlur(p, (0, 0), sigma)

    def to_blocks(a):
        return reduce_scalar_to_blocks(a, block, "mean")

    acc = {k: np.zeros((ny, nx), np.float64) for k in
           ("texture", "change", "residual", "appearance", "tspeed", "fspeed")}
    intensity_t = np.zeros((T, ny, nx), np.float32)

    prev = frames[0].astype(np.float32)
    intensity_t[0] = to_blocks(prev)
    for t in range(1, T):
        curr = frames[t].astype(np.float32)
        intensity_t[t] = to_blocks(curr)

        prods = tensor_products(prev, curr)                   # (6, H, W)
        J = np.stack([smooth(prods[i]) for i in range(6)])    # small-scale tensor
        uv = flow_from_tensor(J)                              # (H, W, 2) px/frame
        acc["tspeed"] += to_blocks(np.hypot(uv[..., 0], uv[..., 1]) * fps)
        acc["change"] += to_blocks(temporal_energy(J))
        acc["residual"] += to_blocks(flow_residual(J))        # 1st-order self-residual
        acc["texture"] += to_blocks(spatial_min_eigen(J))

        flow = fb.compute(prev, curr)                         # (H, W, 2) px/frame
        _, _, fspeed = reduce_to_blocks(flow, block, fps)
        acc["fspeed"] += fspeed
        # Clean appearance channel: change energy the GOOD flow cannot explain.
        r = brightness_residual_field(prev, curr, flow)
        acc["appearance"] += to_blocks(r * r)
        prev = curr
    fb.close()

    n = max(1, T - 1)
    change = (acc["change"] / n).astype(np.float32)
    appearance = (acc["appearance"] / n).astype(np.float32)
    # Absolute residual energy is confounded by contrast (gradient x flow-error is
    # large on any sharp texture). The contrast-independent discriminant is the
    # FRACTION of change energy motion cannot explain, gated by a change floor so
    # near-static blocks do not divide noise by noise.
    floor = 0.02 * float(change.max()) if change.size else 0.0
    appear_frac = np.where(change > floor, appearance / (change + 1e-6), 0.0)
    return {
        "grid": (ny, nx),
        "texture": (acc["texture"] / n).astype(np.float32),
        "change_mean": change,
        "residual_mean": (acc["residual"] / n).astype(np.float32),
        "appearance": appearance,
        "appear_frac": appear_frac.astype(np.float32),
        "tensor_speed_mean": (acc["tspeed"] / n).astype(np.float32),
        "farne_speed_mean": (acc["fspeed"] / n).astype(np.float32),
        "amp_variance": intensity_t.var(0),                   # DC amplitude var
    }


# -- reporting ---------------------------------------------------------------

def quadrant_table(res):
    ny, nx = res["grid"]
    hy, hx = ny // 2, nx // 2
    quads = {"top-left": (slice(0, hy), slice(0, hx)),
             "top-right": (slice(0, hy), slice(hx, nx)),
             "bot-left": (slice(hy, ny), slice(0, hx)),
             "bot-right": (slice(hy, ny), slice(hx, nx))}
    chans = ["tensor_speed_mean", "change_mean", "appearance", "appear_frac",
             "texture", "amp_variance"]
    labels = {"tensor_speed_mean": "flow(px/s)", "change_mean": "change Jtt",
              "appearance": "appear(abs)", "appear_frac": "appear(frac)",
              "texture": "texture", "amp_variance": "amp var"}
    print("\nper-quadrant channel means (each channel should peak in ONE quadrant):")
    print(f"{'quadrant':<11}" + "".join(f"{labels[c]:>14}" for c in chans))
    col_peak = {c: max(quads, key=lambda q: res[c][quads[q]].mean()) for c in chans}
    for q, sl in quads.items():
        row = f"{q:<11}"
        for c in chans:
            val = res[c][sl].mean()
            mark = " *" if col_peak[c] == q else "  "
            row += f"{val:>12.3g}{mark}"
        print(row)
    print("  (* = the quadrant where that channel is strongest)")


def _panel(arr, title, size=220, cmap=cv2.COLORMAP_TURBO):
    a = arr.astype(np.float32)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    norm = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(img, 22, 4, 4, 4, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(img, title, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (235, 235, 235), 1, cv2.LINE_AA)
    return img


def save_montage(res, frame0, path, ncols=4):
    panels = [
        _panel(res["tensor_speed_mean"], "tensor flow (px/s)"),
        _panel(res["farne_speed_mean"], "Farneback flow (px/s)"),
        _panel(res["change_mean"], "change energy Jtt"),
        _panel(res["texture"], "texture (min-eigen)"),
        _panel(res["appearance"], "appearance abs (confounded)"),
        _panel(res["appear_frac"], "appearance FRACTION"),
        _panel(res["amp_variance"], "amplitude variance"),
        _panel(cv2.resize(frame0, res["grid"][::-1]), "frame 0",
               cmap=cv2.COLORMAP_BONE),
    ]
    blank = np.zeros_like(panels[0])
    while len(panels) % ncols:
        panels.append(blank)
    rows = [np.hstack(panels[i:i + ncols]) for i in range(0, len(panels), ncols)]
    cv2.imwrite(path, np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video")
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--out", default="structure_tensor_poc.png")
    args = ap.parse_args()

    if args.video:
        frames, fps = load_video_frames(args.video, args.seconds)
        print(f"video: {os.path.basename(args.video)}  {len(frames)} frames @ {fps:.1f} fps")
    else:
        fps = 60.0
        frames, _ = build_synthetic(T=max(30, int(args.seconds * fps)), fps=fps)
        print(f"synthetic scene: {len(frames)} frames @ {fps:.1f} fps")

    res = run(frames, args.block, fps)

    ts = res["tensor_speed_mean"].ravel()
    fs = res["farne_speed_mean"].ravel()
    if ts.std() > 1e-6 and fs.std() > 1e-6:
        r_all = float(np.corrcoef(ts, fs)[0, 1])
        # The gradient-method (tensor) flow is a small-displacement, brightness-
        # constant estimator, so it only tracks a pyramidal Farneback solve in
        # LK's actual validity domain: texture present AND sub-pixel displacement
        # (< 1 px/frame == < fps px/s). Defining "valid" any other way (e.g. by
        # residual) misreports the regime -- on real footage that inverted the
        # ordering. Report both so the dependence is visible, not hidden.
        tex = res["texture"].ravel()
        subpixel = fs < fps                          # < 1 px/frame
        good = (tex > np.median(tex)) & subpixel
        print(f"\ntensor-flow vs Farneback speed (Pearson r):")
        print(f"  all blocks                              : {r_all:.3f}")
        if good.sum() > 3 and ts[good].std() > 1e-6 and fs[good].std() > 1e-6:
            r_good = float(np.corrcoef(ts[good], fs[good])[0, 1])
            print(f"  textured, sub-pixel blocks (LK valid)   : {r_good:.3f}  "
                  f"(n={int(good.sum())}) -- flow IS the tensor where LK holds")

    if not args.video:
        quadrant_table(res)

    save_montage(res, frames[0], args.out)
    print(f"\nmontage written: {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
