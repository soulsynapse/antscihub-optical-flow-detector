"""Standalone launcher for the scalogram explorer (expanded-cache proposal).

Same launch contract as scripts/structure_tensor_explorer.py: pick an existing
feature cache and explore it. This one shows the Morlet-scalogram detection
channel the expanded cache would add, side by side in idiom with the structure-
tensor explorer. Changes nothing in the cache or pipeline. See
docs/expanded_cache_plan.md.

Run:
    .venv\\Scripts\\python.exe scripts/scalogram_explorer.py
    .venv\\Scripts\\python.exe scripts/scalogram_explorer.py --cache <key>
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication, QInputDialog

from core import cache as cache_mod
from core.tensor_channels import CHANNEL_VERSION
from gui.explorers.scalogram_explorer import ScalogramExplorer


def _pick_cache(cache_root: str) -> str | None:
    caches = cache_mod.list_caches(cache_root)
    if not caches:
        print(f"No caches under {cache_root}. Build one in Preprocessing & Flow first.")
        return None
    caches.sort(key=lambda c: c.get("duration_s", 0))
    labels = []
    for c in caches:
        video = os.path.basename(c.get("video_path", "?"))
        n_regions = len(c.get("replicate_tiles", []))
        scope = f"{n_regions} reps" if n_regions else "whole frame"
        labels.append(f"{c['key']}  |  {video}  |  {c.get('n_frames','?')} fr  "
                      f"|  {scope}")
    choice, ok = QInputDialog.getItem(
        None, "Open a feature cache", "Cache to explore:", labels, 0, False)
    return caches[labels.index(choice)]["key"] if ok else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache")
    ap.add_argument("--cache-root", default=os.path.join(".", ".cache"))
    ap.add_argument("--video", help="override the video path stored in cache")
    args = ap.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    key = args.cache or _pick_cache(args.cache_root)
    if not key:
        return 0
    cache = cache_mod.open_cache(args.cache_root, key)
    try:
        video_path = args.video or cache.meta.get("video_path")
        sidecar = os.path.join(args.cache_root, key,
                               f"tensor_channels_v{CHANNEL_VERSION}.npz")
        win = ScalogramExplorer(cache, video_path, sidecar_path=sidecar)
        win.show()
        return app.exec()
    finally:
        cache.close()


if __name__ == "__main__":
    raise SystemExit(main())
