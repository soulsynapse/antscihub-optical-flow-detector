"""Render each tab offscreen to a PNG so the layout can be eyeballed without a
display. Run after smoke_test.py."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt6.QtWidgets import QApplication
from core import cache as cache_mod
from core.behavior import default_wingbeat
from gui.main_window import MainWindow

app = QApplication(sys.argv)
win = MainWindow(project_dir=".")
win.resize(1750, 980)
st = win.state
st.load_video("Videos/Raw/GX010050c2_02_18_26.MP4")
key = [c["key"] for c in cache_mod.list_caches(".cache") if c["key"].endswith("_smoke")][0]
st.open_cache(key)

st.set_frame(180)
win.tab2._on_box_drawn(0.05,0.06,0.45,0.45)
win.tab2._on_box_drawn(0.55,0.06,0.95,0.45)
win.tab2._on_box_drawn(0.30,0.55,0.70,0.94)
win.tab2.draw_btn.setChecked(True)
win.tab2.list.setCurrentRow(0)

from core.behavior import Behavior, LogicNode, RangeLeaf
band = st.band_features[0]
col = np.concatenate([st.roi_series(r, band) for r in st.rois])
b = Behavior(name="wingbeat", color="#ff4488",
             spec=LogicNode("and", [RangeLeaf(band, float(np.percentile(col, 96)),
                                              float("inf"))]))
st.library.save(b)
win.tab3._refresh_library(); win.tab3.current = b
win.tab3._sync_from_behavior()
if st.rois:
    win.tab3._select_roi(st.rois[1].roi_id)
win.tab3._recompute()

os.makedirs("screenshots", exist_ok=True)
win.show()
for i, name in enumerate(["tab1_flow", "tab2_roi", "tab3_behavior"]):
    win.tabs.setCurrentIndex(i)
    app.processEvents()
    for _ in range(3):
        app.processEvents()
    win.grab().save(f"screenshots/{name}.png")
    print("wrote", f"screenshots/{name}.png")
