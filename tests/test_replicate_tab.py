"""Batch G: the replicate tab's direct-manipulation gestures (T11, T12).

These drive FrameView with real mouse events rather than calling the handlers,
because every defect this batch can produce lives in the DISPATCH, not in the
handlers: which of press/release owns selection, whether a press inside a box
still falls through to `clicked`, and whether a drag that leaves the frame is
clamped as a move or silently becomes a resize. A test that calls
``_on_box_moved`` directly passes in all of those cases.

The stamp tests exist because T11 removed a piece of state rather than syncing
it. The failure they guard is quiet: a stamp that still reports the last box
DRAWN places boxes of a size the highlighted replicate does not have, and every
downstream "min blocks" comparison silently stops being like-for-like.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

from gui.video_panel import FrameView

_APP = QApplication.instance() or QApplication([])


def _view(boxes, *, drag=True, draw=True) -> FrameView:
    v = FrameView()
    v.resize(400, 300)
    v.set_frame(np.zeros((300, 400, 3), np.uint8))
    v.box_drag_enabled = drag
    v.draw_enabled = draw
    v.set_boxes(boxes)
    # _draw_rect is only assigned during paint; the hit test needs it.
    v.grab()
    return v


def _press(v, x, y, button=Qt.MouseButton.LeftButton):
    v.mousePressEvent(QMouseEvent(
        QMouseEvent.Type.MouseButtonPress, QPointF(x, y), QPointF(x, y),
        button, button, Qt.KeyboardModifier.NoModifier))


def _move(v, x, y):
    v.mouseMoveEvent(QMouseEvent(
        QMouseEvent.Type.MouseMove, QPointF(x, y), QPointF(x, y),
        Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier))


def _release(v, x, y, button=Qt.MouseButton.LeftButton):
    v.mouseReleaseEvent(QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease, QPointF(x, y), QPointF(x, y),
        button, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))


def _center(v, box):
    """Display-pixel centre of a fractional box."""
    r = v._draw_rect
    x0, y0, x1, y1 = box[:4]
    return (r.x() + (x0 + x1) / 2 * r.width(),
            r.y() + (y0 + y1) / 2 * r.height())


BOX_A = (0.10, 0.10, 0.30, 0.30, "a", "#ff0000", False)
BOX_B = (0.60, 0.60, 0.80, 0.80, "b", "#00ff00", False)


class BoxGestureTest(unittest.TestCase):
    """T12: press selects, release-in-place zooms, drag repositions."""

    def test_press_inside_a_box_emits_grabbed_not_clicked_through(self):
        v = _view([BOX_A, BOX_B])
        grabbed, clicked, stamped = [], [], []
        v.box_grabbed.connect(grabbed.append)
        v.clicked.connect(clicked.append)
        v.stamp_at.connect(lambda *a: stamped.append(a))

        x, y = _center(v, BOX_B)
        _press(v, x, y)
        _release(v, x, y)

        self.assertEqual(grabbed, [1], "press inside a box must select it")
        # The critical half: in draw mode this press must NOT also become a
        # stamp, or clicking an existing replicate buries it under a new one.
        self.assertEqual(stamped, [], "a click inside a box must not place one")
        self.assertEqual(clicked, [], "a grabbed box must not fall through")

    def test_release_in_place_is_a_click_not_a_move(self):
        v = _view([BOX_A])
        clicked, moved = [], []
        v.box_clicked.connect(clicked.append)
        v.box_moved.connect(lambda *a: moved.append(a))

        x, y = _center(v, BOX_A)
        _press(v, x, y)
        _move(v, x + 2, y + 1)      # under the 5 px threshold: still a click
        _release(v, x + 2, y + 1)

        self.assertEqual(clicked, [0])
        self.assertEqual(moved, [], "a 2 px wobble must not reposition a box")

    def test_click_on_empty_space_still_stamps(self):
        v = _view([BOX_A])
        stamped, grabbed = [], []
        v.stamp_at.connect(lambda *a: stamped.append(a))
        v.box_grabbed.connect(grabbed.append)

        r = v._draw_rect
        x = r.x() + 0.90 * r.width()
        y = r.y() + 0.20 * r.height()
        _press(v, x, y)
        _release(v, x, y)

        self.assertEqual(grabbed, [])
        self.assertEqual(len(stamped), 1, "empty space must still place a box")

    def test_drag_repositions_and_preserves_size(self):
        v = _view([BOX_A])
        moved = []
        v.box_moved.connect(lambda *a: moved.append(a))

        x, y = _center(v, BOX_A)
        r = v._draw_rect
        dx = 0.25 * r.width()           # +0.25 in fractions
        _press(v, x, y)
        _move(v, x + dx, y)
        _release(v, x + dx, y)

        self.assertEqual(len(moved), 1)
        idx, nx0, ny0, nx1, ny1 = moved[0]
        self.assertEqual(idx, 0)
        self.assertAlmostEqual(nx0, 0.35, places=2)
        self.assertAlmostEqual(ny0, 0.10, places=2)
        self.assertAlmostEqual(nx1 - nx0, 0.20, places=3)
        self.assertAlmostEqual(ny1 - ny0, 0.20, places=3)

    def test_drag_past_the_edge_clamps_without_resizing(self):
        """The dangerous direction: a clamp that moves one corner turns a
        reposition into a resize, and a resized replicate silently holds a
        different number of blocks than its same-stamp siblings."""
        v = _view([BOX_A])
        moved = []
        v.box_moved.connect(lambda *a: moved.append(a))

        x, y = _center(v, BOX_A)
        r = v._draw_rect
        _press(v, x, y)
        _move(v, x + 4 * r.width(), y + 4 * r.height())   # far outside
        _release(v, x + 4 * r.width(), y + 4 * r.height())

        _, nx0, ny0, nx1, ny1 = moved[0]
        self.assertAlmostEqual(nx1 - nx0, 0.20, places=6, msg="size changed")
        self.assertAlmostEqual(ny1 - ny0, 0.20, places=6, msg="size changed")
        self.assertAlmostEqual(nx1, 1.0, places=6)
        self.assertAlmostEqual(ny1, 1.0, places=6)

    def test_selected_box_wins_an_overlap(self):
        """Replicate boxes are routinely placed edge to edge. Grabbing whichever
        one sorts later, rather than the highlighted one, makes repositioning
        feel random exactly where boxes are densest."""
        under = (0.10, 0.10, 0.50, 0.50, "under", "#ff0000", True)
        over = (0.20, 0.20, 0.60, 0.60, "over", "#00ff00", False)
        v = _view([under, over])
        grabbed = []
        v.box_grabbed.connect(grabbed.append)

        r = v._draw_rect                      # a point inside BOTH
        _press(v, r.x() + 0.35 * r.width(), r.y() + 0.35 * r.height())
        self.assertEqual(grabbed, [0], "the selected box must win the overlap")

    def test_drag_is_off_for_every_other_consumer(self):
        """The four explorers, mask_dialog and tab3 all render boxes the user
        must be able to click THROUGH. Default-on would silently swallow those
        clicks."""
        v = _view([BOX_A], drag=False, draw=False)
        grabbed, clicked = [], []
        v.box_grabbed.connect(grabbed.append)
        v.clicked.connect(clicked.append)

        x, y = _center(v, BOX_A)
        _press(v, x, y)
        self.assertEqual(grabbed, [])
        self.assertEqual(len(clicked), 1, "the press must reach `clicked`")

    def test_peek_suppresses_the_hit_test(self):
        """Shift-to-peek hides the boxes. Grabbing one you cannot see is a blind
        edit, so the hit test hides with the drawing."""
        v = _view([BOX_A])
        v.set_overlays_hidden(True)
        grabbed = []
        v.box_grabbed.connect(grabbed.append)
        _press(v, *_center(v, BOX_A))
        self.assertEqual(grabbed, [])

    def test_release_without_a_move_event_does_not_write_a_no_op_move(self):
        """Qt compresses move events under load, so press->release with none in
        between is reachable. The release then reads as a 150 px drag while the
        cached delta is still zero: the box is rewritten to where it already
        was, and the tab's _rebuild_rois and sidecar write both fire for a
        reposition that never happened."""
        v = _view([BOX_A])
        moved = []
        v.box_moved.connect(lambda *a: moved.append(a))

        x, y = _center(v, BOX_A)
        r = v._draw_rect
        dx = 0.25 * r.width()
        _press(v, x, y)
        _release(v, x + dx, y)          # no _move() at all

        self.assertEqual(len(moved), 1)
        _, nx0, _, _, _ = moved[0]
        self.assertAlmostEqual(nx0, 0.35, places=2,
                               msg="the box did not follow the cursor")

    def test_right_click_during_a_drag_cancels_it(self):
        """Committing here would use whatever partial delta the drag reached,
        and back_requested's un-zoom would change the very mapping that delta
        was measured in -- so the box lands somewhere never dragged to."""
        v = _view([BOX_A])
        moved, back = [], []
        v.box_moved.connect(lambda *a: moved.append(a))
        v.back_requested.connect(lambda: back.append(1))

        x, y = _center(v, BOX_A)
        _press(v, x, y)
        _move(v, x + 3, y)
        _press(v, x + 200, y + 120, button=Qt.MouseButton.RightButton)
        _release(v, x + 200, y + 120, button=Qt.MouseButton.RightButton)

        self.assertEqual(moved, [], "a cancelled drag must not reposition")
        self.assertEqual(back, [], "the cancel must not also be a 'back'")
        self.assertIsNone(v._move_idx)

    def test_right_click_still_means_back(self):
        """T12 asked for right-click-to-delete; it was refused. This pins the
        refusal, because the tab now zooms and would otherwise have no way out."""
        v = _view([BOX_A])
        back, grabbed = [], []
        v.back_requested.connect(lambda: back.append(1))
        v.box_grabbed.connect(grabbed.append)
        _press(v, *_center(v, BOX_A), button=Qt.MouseButton.RightButton)
        self.assertEqual(back, [1])
        self.assertEqual(grabbed, [], "right-click must not grab a box")


class StampFollowsSelectionTest(unittest.TestCase):
    """T11: the stamp is derived from the selection, not stored."""

    def _tab(self):
        from gui.state import AppState
        from gui.tab2_replicates import Tab2Replicates
        tab = Tab2Replicates(AppState())
        tab._on_box_drawn(0.10, 0.10, 0.30, 0.30)     # rep1, 0.20 x 0.20
        tab._on_box_drawn(0.50, 0.50, 0.60, 0.90)     # rep2, 0.10 x 0.40
        return tab

    def test_drawing_selects_the_new_box_and_sets_the_stamp(self):
        tab = self._tab()
        self.assertEqual(tab._selected_rep()["label"], "rep2")
        self.assertAlmostEqual(tab.stamp_size[0], 0.10, places=6)
        self.assertAlmostEqual(tab.stamp_size[1], 0.40, places=6)

    def test_selecting_an_older_box_moves_the_stamp_back_to_it(self):
        """The whole of T11. Under the old stored stamp this returned rep2's
        size while rep1 was highlighted."""
        tab = self._tab()
        tab.list.setCurrentRow(0)
        self.assertEqual(tab._selected_rep()["label"], "rep1")
        self.assertAlmostEqual(tab.stamp_size[0], 0.20, places=6)
        self.assertAlmostEqual(tab.stamp_size[1], 0.20, places=6)

    def test_a_stamped_box_has_the_selected_size_not_the_last_drawn(self):
        tab = self._tab()
        tab.list.setCurrentRow(0)             # back to rep1 (0.20 x 0.20)
        tab._on_stamp_at(0.80, 0.20)
        x0, y0, x1, y1 = tab.replicates[-1]["frac"]
        self.assertAlmostEqual(x1 - x0, 0.20, places=6)
        self.assertAlmostEqual(y1 - y0, 0.20, places=6)

    def test_the_label_reports_the_selected_size(self):
        tab = self._tab()
        self.assertIn("10%×40%", tab.stamp_chk.text())
        tab.list.setCurrentRow(0)
        self.assertIn("20%×20%", tab.stamp_chk.text())

    def test_no_selection_means_no_stamp(self):
        from gui.state import AppState
        from gui.tab2_replicates import Tab2Replicates
        tab = Tab2Replicates(AppState())
        self.assertIsNone(tab.stamp_size)
        tab._on_stamp_at(0.5, 0.5)
        self.assertEqual(tab.replicates, [],
                         "a click with nothing selected must place nothing")

    def test_deleting_the_selection_clears_the_stamp_label(self):
        tab = self._tab()
        tab.list.setCurrentRow(1)
        tab._delete()
        tab._delete()
        self.assertIsNone(tab.stamp_size)
        self.assertEqual(tab.stamp_chk.text(), "Fixed-size stamp")


class RepositionUpdatesModelTest(unittest.TestCase):

    def test_moving_a_box_rewrites_its_fraction_and_keeps_the_stamp(self):
        from gui.state import AppState
        from gui.tab2_replicates import Tab2Replicates
        tab = Tab2Replicates(AppState())
        tab._on_box_drawn(0.10, 0.10, 0.30, 0.30)
        tab._on_box_moved(0, 0.50, 0.50, 0.70, 0.70)
        self.assertEqual(tab.replicates[0]["frac"], (0.50, 0.50, 0.70, 0.70))
        self.assertAlmostEqual(tab.stamp_size[0], 0.20, places=6)

    def test_a_stale_index_is_ignored(self):
        """box_moved carries an index into the box list, which the handler for
        box_grabbed may already have rebuilt."""
        from gui.state import AppState
        from gui.tab2_replicates import Tab2Replicates
        tab = Tab2Replicates(AppState())
        tab._on_box_drawn(0.10, 0.10, 0.30, 0.30)
        tab._on_box_moved(7, 0.5, 0.5, 0.7, 0.7)      # must not raise
        self.assertEqual(tab.replicates[0]["frac"], (0.10, 0.10, 0.30, 0.30))


class TabWiringTest(unittest.TestCase):
    """The two halves above are each correct in isolation; this drives real mouse
    events through the assembled tab, which is where a missed `connect` or a
    signal wired to the wrong slot actually shows up."""

    def _tab(self):
        from gui.state import AppState
        from gui.tab2_replicates import Tab2Replicates
        tab = Tab2Replicates(AppState())
        tab.video.view.resize(400, 300)
        tab.video.view.set_frame(np.zeros((300, 400, 3), np.uint8))
        tab._on_box_drawn(0.10, 0.10, 0.30, 0.30)     # rep1
        tab._on_box_drawn(0.60, 0.60, 0.80, 0.80)     # rep2
        tab.video.view.grab()                          # assigns _draw_rect
        return tab

    def test_dragging_a_box_moves_the_replicate(self):
        tab = self._tab()
        v = tab.video.view
        r = v._draw_rect
        x, y = _center(v, tab.replicates[0]["frac"])
        _press(v, x, y)
        _move(v, x + 0.20 * r.width(), y)
        _release(v, x + 0.20 * r.width(), y)

        x0, _, x1, _ = tab.replicates[0]["frac"]
        self.assertAlmostEqual(x0, 0.30, places=2)
        self.assertAlmostEqual(x1 - x0, 0.20, places=3)

    def test_clicking_a_box_selects_and_zooms_to_it(self):
        tab = self._tab()
        v = tab.video.view
        tab.list.setCurrentRow(0)
        self.assertIsNone(v.focus_frac)

        x, y = _center(v, tab.replicates[1]["frac"])
        _press(v, x, y)
        _release(v, x, y)

        self.assertEqual(tab._selected_rep()["label"], "rep2")
        self.assertIsNotNone(v.focus_frac, "clicking a box must zoom to it")
        # The box's frac VERBATIM -- byte for byte the zoom the four explorers
        # apply on a region change. A margin here would show the same replicate
        # at a different magnification than the explorer showing it.
        self.assertEqual(tuple(v.focus_frac), tuple(tab.replicates[1]["frac"]))

    def test_dragging_does_not_zoom(self):
        """A zoom on press would move the frame mid-gesture; a zoom on a
        completed drag would fly the view away every time you nudge a box."""
        tab = self._tab()
        v = tab.video.view
        r = v._draw_rect
        x, y = _center(v, tab.replicates[0]["frac"])
        _press(v, x, y)
        _move(v, x + 0.20 * r.width(), y)
        self.assertIsNone(v.focus_frac, "zoomed during a drag")
        _release(v, x + 0.20 * r.width(), y)
        self.assertIsNone(v.focus_frac, "a reposition must not zoom")

    def test_a_box_can_still_be_dragged_at_full_zoom(self):
        """Load-bearing once the zoom has no margin: the box fills the view, so
        there is no in-frame empty space to drag into and the whole gesture
        rests on _delta_to extrapolating past the widget edge. If that ever
        clipped, repositioning would be impossible whenever zoomed -- and the
        zoom is exactly when you can see well enough to want to."""
        tab = self._tab()
        v = tab.video.view
        c = _center(v, tab.replicates[1]["frac"])
        _press(v, *c); _release(v, *c)
        self.assertEqual(tuple(v.focus_frac), tuple(tab.replicates[1]["frac"]))
        v.grab()                       # re-lay _draw_rect for the zoomed view

        before = tab.replicates[1]["frac"]
        r = v._draw_rect
        c = _center(v, before)
        # Past the right edge of the widget: only reachable by extrapolation.
        _press(v, *c)
        _move(v, r.x() + 1.4 * r.width(), c[1])
        _release(v, r.x() + 1.4 * r.width(), c[1])

        after = tab.replicates[1]["frac"]
        self.assertGreater(after[0], before[0], "the box did not move")
        self.assertAlmostEqual(after[2] - after[0], before[2] - before[0],
                               places=6, msg="a zoomed drag resized the box")

    def test_right_click_zooms_back_out(self):
        tab = self._tab()
        v = tab.video.view
        x, y = _center(v, tab.replicates[1]["frac"])
        _press(v, x, y)
        _release(v, x, y)
        self.assertIsNotNone(v.focus_frac)

        _press(v, x, y, button=Qt.MouseButton.RightButton)
        self.assertIsNone(v.focus_frac, "right-click must un-zoom")

    def test_deleting_the_zoomed_box_releases_the_zoom(self):
        tab = self._tab()
        v = tab.video.view
        x, y = _center(v, tab.replicates[1]["frac"])
        _press(v, x, y)
        _release(v, x, y)
        self.assertIsNotNone(v.focus_frac)

        tab._delete()
        self.assertIsNone(v.focus_frac,
                          "deleting the zoomed box strands the view on it")

    def test_placing_boxes_in_a_row_does_not_zoom(self):
        """T11's workflow: select the size you want, then click repeatedly.
        Zooming on each placement would make the second click land somewhere
        entirely different from where the user aimed it."""
        tab = self._tab()
        tab.list.setCurrentRow(0)
        tab._on_stamp_at(0.50, 0.20)
        tab._on_stamp_at(0.80, 0.20)
        self.assertIsNone(tab.video.view.focus_frac)
        self.assertEqual(len(tab.replicates), 4)


if __name__ == "__main__":
    unittest.main()
