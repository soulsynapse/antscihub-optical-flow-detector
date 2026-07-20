from __future__ import annotations

import unittest
from unittest.mock import patch

from core import sysmem


class AvailableBytesTests(unittest.TestCase):
    def test_reports_a_plausible_number_on_this_platform(self):
        avail = sysmem.available_bytes()
        if avail is None:
            self.skipTest(f"no memory probe for {sysmem.sys.platform}")
        # Loose on purpose: the point is that the probe read a real figure and
        # not a zero or a byte count off by a factor of 1024.
        self.assertGreater(avail, 64 * 1024 ** 2)
        self.assertLess(avail, 8 * 1024 ** 4)

    def test_a_failing_probe_is_not_allowed_to_propagate(self):
        """The caller sizes a cache with this; an unreadable machine must degrade
        to the fallback rather than take down the surface that asked."""
        with patch.object(sysmem, "_windows_available", side_effect=OSError), \
                patch.object(sysmem, "_linux_available", side_effect=OSError), \
                patch.object(sysmem, "_darwin_available", side_effect=OSError):
            self.assertIsNone(sysmem.available_bytes())


class BudgetTests(unittest.TestCase):
    FLOOR = 2 * 1024 ** 3
    CAP = 16 * 1024 ** 3

    def test_scales_with_available_memory(self):
        with patch.object(sysmem, "available_bytes", return_value=32 * 1024 ** 3):
            self.assertEqual(sysmem.budget_bytes(0.25, self.FLOOR, self.CAP),
                             8 * 1024 ** 3)

    def test_small_machine_keeps_the_old_flat_behaviour(self):
        with patch.object(sysmem, "available_bytes", return_value=1 * 1024 ** 3):
            self.assertEqual(sysmem.budget_bytes(0.25, self.FLOOR, self.CAP),
                             self.FLOOR)

    def test_large_machine_is_capped(self):
        with patch.object(sysmem, "available_bytes", return_value=512 * 1024 ** 3):
            self.assertEqual(sysmem.budget_bytes(0.25, self.FLOOR, self.CAP),
                             self.CAP)

    def test_unknown_memory_falls_back_to_the_floor(self):
        with patch.object(sysmem, "available_bytes", return_value=None):
            self.assertEqual(sysmem.budget_bytes(0.25, self.FLOOR, self.CAP),
                             self.FLOOR)


if __name__ == "__main__":
    unittest.main()
