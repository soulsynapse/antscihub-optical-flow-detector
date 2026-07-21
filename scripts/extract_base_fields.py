"""T31 stage 1: full-video channel extraction for rep3, cached to disk.

Extracts all four live channels at the corpus geometry (block 4, ds 1.0,
grid 113x115) over the whole clip and saves each (T, ny, nx) tensor as a
float32 .npy, so the analysis stage can iterate on stats/bands/plots without
re-decoding. ~4 min, ~6.4 GB on disk.
"""
import json
import os
import sys
import time

import numpy as np

from core.batch import config_from_overrides, load_replicates, default_replicate_path
from core.channel_source import live_channel_source

VID = "Videos/Stabilized/rep3_intermittent_crop.MP4"
OUT = os.environ["T31_DIR"]
os.makedirs(OUT, exist_ok=True)


def main():
    cfg = config_from_overrides(None, 4)          # block 4 to match the corpus
    reps = load_replicates(default_replicate_path(VID))
    t = time.perf_counter()
    cd = live_channel_source(VID, cfg, reps, start=0, n=None)
    dt = time.perf_counter() - t
    meta = cd.meta
    T = int(meta["n_frames"])
    print(f"extracted {T} frames in {dt:.1f}s ({T/dt:.1f} fps) "
          f"grid={meta['grid']} block={meta['block_size']} "
          f"truncated={meta.get('truncated')}", flush=True)
    for name, arr in cd.channels.items():
        arr = np.ascontiguousarray(arr, np.float32)
        np.save(os.path.join(OUT, f"chan_{name}.npy"), arr)
        print(f"  saved {name} {arr.shape} {arr.nbytes/1e9:.2f} GB", flush=True)
    side = {
        "fps": float(meta["fps"]),
        "grid": [int(v) for v in meta["grid"]],
        "block_size": int(meta["block_size"]),
        "downsample": float(meta["downsample"]),
        "n_frames": T,
        "window_start": int(cd.window_start),
        "truncated": bool(meta.get("truncated", False)),
        "channels": sorted(cd.channels),
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(side, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
