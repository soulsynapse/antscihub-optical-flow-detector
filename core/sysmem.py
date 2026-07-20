"""Available physical memory, without a psutil dependency.

One number is needed (free RAM, to size the live surface's per-pixel cache), and
pulling in psutil for it would be the heaviest dependency in requirements.txt.
Each platform exposes it natively; anything unrecognised returns None and the
caller falls back to a fixed floor rather than guessing.

"Available" means *reclaimable without swapping* -- MemAvailable on Linux and
ullAvailPhys on Windows, both of which count reclaimable page cache. Free-page
counts alone read far too low on a machine that has been up a while and would
shrink the budget for no reason.
"""
from __future__ import annotations

import ctypes
import os
import re
import sys


def _windows_available() -> int | None:
    class _MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = _MemStatus()
    st.dwLength = ctypes.sizeof(_MemStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return int(st.ullAvailPhys)


def _linux_available() -> int | None:
    with open("/proc/meminfo", encoding="ascii") as fh:
        text = fh.read()
    # MemAvailable is the kernel's own estimate and already accounts for
    # reclaimable cache; MemFree alone understates badly.
    m = re.search(r"^MemAvailable:\s+(\d+) kB", text, re.M)
    return int(m.group(1)) * 1024 if m else None


def _darwin_available() -> int | None:
    import subprocess
    out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                         timeout=5).stdout
    page = re.search(r"page size of (\d+) bytes", out)
    page_size = int(page.group(1)) if page else 4096
    pages = 0
    # Free + inactive + speculative is the usual "could be handed out" set.
    for field in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(rf"^{field}:\s+(\d+)\.", out, re.M)
        if m:
            pages += int(m.group(1))
    return pages * page_size or None


def available_bytes() -> int | None:
    """Physical memory obtainable without swapping, or None if unknown."""
    try:
        if sys.platform == "win32":
            return _windows_available()
        if sys.platform.startswith("linux"):
            return _linux_available()
        if sys.platform == "darwin":
            return _darwin_available()
    except Exception:
        # Never let a memory probe break the caller -- an unknown number is a
        # fine answer here, and the fallback is a safe fixed budget.
        return None
    return None


def budget_bytes(fraction: float, floor: int, cap: int) -> int:
    """Size a working-set budget as a fraction of available RAM.

    Clamped both ways on purpose: ``floor`` keeps small or unreadable machines
    at the behaviour they already had, and ``cap`` stops a very large machine
    from sizing a *preview* buffer in the tens of gigabytes, where the decode
    time to fill it stops being interactive long before the memory runs out.
    """
    avail = available_bytes()
    if avail is None:
        return floor
    return int(max(floor, min(cap, avail * fraction)))
