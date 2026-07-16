"""Standalone launcher for the temporal-variance / appearance explorer (tool 3).

The UI lives in :mod:`gui.variance_explorer` so it can move into the main app as
a tab without carrying cache-picking or QApplication ownership. On first open it
precomputes the temporal channels from the video (slow) and writes a sidecar next
to the cache, so re-opens are instant.

Run:
    .venv\\Scripts\\python.exe scripts/variance_explorer.py
    .venv\\Scripts\\python.exe scripts/variance_explorer.py --cache <key>
    .venv\\Scripts\\python.exe scripts/variance_explorer.py --cache-root ./.cache
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication, QInputDialog

from core import cache as cache_mod
from core.tensor_channels import CHANNEL_VERSION
from gui.variance_explorer import TemporalVarianceExplorer


def _pick_cache(cache_root: str) -> str | None:
    caches = cache_mod.list_caches(cache_root)
    if not caches:
        print(f"No caches under {cache_root}. Build one in Preprocessing & Flow "
              f"first.")
        return None
    caches.sort(key=lambda c: c.get("duration_s", 0))
    labels = []
    for cache in caches:
        video = os.path.basename(cache.get("video_path", "?"))
        tag = "test" if cache.get("test_mode") else "full"
        n_regions = len(cache.get("replicate_tiles", []))
        scope = f"{n_regions} reps" if n_regions else "whole frame"
        labels.append(
            f"{cache['key']}  |  {video}  |  {cache.get('n_frames', '?')} fr  "
            f"|  {scope}  |  {cache['grid'][0]}x{cache['grid'][1]}  |  {tag}")
    choice, ok = QInputDialog.getItem(
        None, "Open a feature cache", "Cache to explore:", labels, 0, False)
    if not ok:
        return None
    return caches[labels.index(choice)]["key"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", help="cache key under --cache-root")
    parser.add_argument("--cache-root", default=os.path.join(".", ".cache"))
    parser.add_argument("--video", help="override the video path stored in cache")
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    key = args.cache or _pick_cache(args.cache_root)
    if not key:
        return 0

    cache = cache_mod.open_cache(args.cache_root, key)
    try:
        video_path = args.video or cache.meta.get("video_path")
        sidecar = os.path.join(args.cache_root, key,
                               f"tensor_channels_v{CHANNEL_VERSION}.npz")
        window = TemporalVarianceExplorer(cache, video_path, sidecar_path=sidecar)
        window.show()
        return app.exec()
    finally:
        cache.close()


if __name__ == "__main__":
    raise SystemExit(main())
