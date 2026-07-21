"""The whole-video track survives a session.

The point of these is not that npz round-trips arrays -- it is that a restore
cannot make a claim the original did not. Coverage, staleness and the value band
each have a way of coming back subtly wrong, and each of those is a strip that
asserts footage was examined when it was not.
"""
import os
import tempfile
import unittest

import numpy as np

from core.live_track import UNCOVERED, TrackStamp, WholeVideoTrack
from gui.track_store import (BP_DISK_CAP, VERSION, load_track, save_track,
                             track_path)

FPS = 30.0
N = 400
B = 6


def _stamp(channel="change", blocks=B, band=(1.0, 5.0)):
    return TrackStamp(channel=channel, freq_band_hz=band, grid=(4, 3),
                      region_index=0, region_blocks=blocks, downsample=1.0,
                      block_size=8)


def _grid(blocks=B):
    """A 3-wide region grid covering ``blocks`` columns. The height is derived,
    not fixed: the oversize case below runs to six figures of blocks, and a grid
    too small for its own columns indexes out of bounds inside the clump pass."""
    gy = np.arange(blocks, dtype=np.int32) // 3
    gx = np.arange(blocks, dtype=np.int32) % 3
    return (int(gy[-1]) + 1, 3, gy, gx)


def _filled(track=None, stamp=None, first=50, rows=120, seed=0):
    """A track with a real written span, as a live pass would leave it."""
    t = track or WholeVideoTrack(n_frames=N, fps=FPS)
    s = stamp or _stamp()
    t.set_stamp(s, region_grid=_grid(s.region_blocks))
    t.set_detector(value_band=(0.5, np.inf), count_band=(1.0, np.inf),
                   detect_window=5, centered=True)
    rng = np.random.default_rng(seed)
    bp = rng.random((rows, s.region_blocks)).astype(np.float32)
    t.write(first, bp, trim=0)
    return t


class TrackStoreRoundTrip(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.video = os.path.join(self.dir, "clip.mp4")

    def test_round_trip_preserves_coverage_and_series(self):
        t = _filled()
        wrote, note = save_track(self.video, t)
        self.assertTrue(wrote, note)
        back, note = load_track(self.video, N, FPS)
        self.assertIsNotNone(back, note)
        np.testing.assert_array_equal(back.stamp_id, t.stamp_id)
        np.testing.assert_allclose(back.count, t.count)
        np.testing.assert_allclose(back.clump, t.clump)
        np.testing.assert_allclose(back.gate, t.gate)
        self.assertEqual(back.stamp, t.stamp)
        self.assertEqual(back.detected_intervals(), t.detected_intervals())

    def test_restored_track_is_current_not_stale_under_same_stamp(self):
        """The restore must not silently re-stamp: a frame computed under the
        stamp still in force is current, and the strip may say so."""
        t = _filled()
        save_track(self.video, t)
        back, _ = load_track(self.video, N, FPS)
        self.assertTrue(back.current.any())
        self.assertFalse(back.stale.any())
        # And re-declaring the very same stamp is a no-op, so band power lives.
        moved = back.set_stamp(_stamp(), region_grid=_grid())
        self.assertFalse(moved)

    def test_different_stamp_makes_restored_work_stale_not_current(self):
        """The load-bearing safety property: work restored from disk and then
        looked at under different settings goes gray, never current."""
        t = _filled()
        save_track(self.video, t)
        back, _ = load_track(self.video, N, FPS)
        self.assertTrue(back.set_stamp(_stamp(channel="tensor_speed"),
                                       region_grid=_grid()))
        self.assertTrue(back.stale.any())
        self.assertFalse(back.current.any())
        self.assertEqual(back.detected_intervals(), [])

    def test_band_power_survives_so_a_retune_needs_no_pass(self):
        t = _filled()
        save_track(self.video, t)
        back, _ = load_track(self.video, N, FPS)
        before = back.count.copy()
        # A value-band move re-derives from retained rows. If band power had not
        # come back this would silently do nothing at all.
        back.set_detector(value_band=(0.9, np.inf), count_band=(1.0, np.inf),
                          detect_window=5, centered=True)
        self.assertFalse(np.array_equal(before, back.count))

    def test_detector_settings_round_trip(self):
        t = _filled()
        save_track(self.video, t)
        back, _ = load_track(self.video, N, FPS)
        # Re-applying the settings it was saved under must be recognised as
        # unchanged; otherwise every restore silently re-derives on first sync.
        before = back.count.copy()
        back.set_detector(value_band=(0.5, np.inf), count_band=(1.0, np.inf),
                          detect_window=5, centered=True)
        np.testing.assert_allclose(before, back.count)


class TrackStoreRefusals(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.video = os.path.join(self.dir, "clip.mp4")

    def test_frame_count_mismatch_is_refused_with_a_note(self):
        """A recrop or re-encode changes what frame 9000 means. Restoring then
        would place every detection at the wrong time."""
        save_track(self.video, _filled())
        back, note = load_track(self.video, N + 17, FPS)
        self.assertIsNone(back)
        self.assertIn("frames", note)

    def test_fps_mismatch_is_refused(self):
        save_track(self.video, _filled())
        back, note = load_track(self.video, N, FPS * 2)
        self.assertIsNone(back)
        self.assertIn("frame rate", note)

    def test_missing_sidecar_is_silent(self):
        back, note = load_track(self.video, N, FPS)
        self.assertIsNone(back)
        self.assertEqual(note, "")

    def test_corrupt_sidecar_is_reported_not_raised(self):
        with open(track_path(self.video), "wb") as f:
            f.write(b"not an npz")
        back, note = load_track(self.video, N, FPS)
        self.assertIsNone(back)
        self.assertTrue(note)

    def test_empty_track_writes_nothing_and_clears_a_stale_sidecar(self):
        save_track(self.video, _filled())
        self.assertTrue(os.path.exists(track_path(self.video)))
        wrote, _ = save_track(self.video, WholeVideoTrack(n_frames=N, fps=FPS))
        self.assertFalse(wrote)
        self.assertFalse(os.path.exists(track_path(self.video)))

    def test_oversized_band_power_is_declined_and_said_so(self):
        """The 1.6 GB geometry this project actually uses. Coverage must still
        round-trip; only the re-tune shortcut is given up."""
        blocks = int(BP_DISK_CAP // (N * 4)) + 64
        t = _filled(stamp=_stamp(blocks=blocks), rows=120)
        wrote, note = save_track(self.video, t)
        self.assertTrue(wrote)
        self.assertIn("too large", note)
        back, _ = load_track(self.video, N, FPS)
        np.testing.assert_array_equal(back.stamp_id, t.stamp_id)
        # Declined, so a re-tune finds nothing to re-derive from -- the
        # documented degradation, not a crash.
        back.set_detector(value_band=(0.9, np.inf), count_band=(1.0, np.inf),
                          detect_window=5, centered=True)

    def test_stale_and_current_spans_both_survive(self):
        """A track that saw two stamps must come back distinguishing them."""
        t = _filled(first=20, rows=60, seed=1)
        t = _filled(track=t, stamp=_stamp(channel="intensity"), first=200,
                    rows=60, seed=2)
        save_track(self.video, t)
        back, _ = load_track(self.video, N, FPS)
        self.assertTrue(back.stale.any())
        self.assertTrue(back.current.any())
        np.testing.assert_array_equal(back.stale, t.stale)
        np.testing.assert_array_equal(back.current, t.current)

    def test_uncovered_stays_uncovered(self):
        t = _filled(first=50, rows=120)
        save_track(self.video, t)
        back, _ = load_track(self.video, N, FPS)
        self.assertTrue((back.stamp_id[:50] == UNCOVERED).all())
        self.assertTrue((back.stamp_id[170:] == UNCOVERED).all())

    def test_version_bump_discards_rather_than_migrates(self):
        save_track(self.video, _filled())
        with np.load(track_path(self.video)) as data:
            payload = {k: data[k] for k in data.files}
        import json
        meta = json.loads(bytes(payload["meta"]).decode("utf-8"))
        meta["version"] = VERSION + 1
        payload["meta"] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), np.uint8)
        np.savez(track_path(self.video), **payload)
        back, _ = load_track(self.video, N, FPS)
        self.assertIsNone(back)


if __name__ == "__main__":
    unittest.main()
