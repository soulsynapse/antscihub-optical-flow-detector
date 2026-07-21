"""Seed candidate channels and score them against the rep3 corpus.

Every candidate below is defined through the SAME channel interface -- raw base
fields, Morlet bands at different frequencies (the frequency-specificity test),
a Butterworth+filtfilt band-energy channel, and a ratio -- to demonstrate the
lab: adding one is a function + @channel(...). The ranked table is span-level AUC.

Usage (from the repo root, with the venv):
    # 1. extract + cache the base fields for a video once (~4 min, ~6 GB):
    T31_DIR=/path/to/cache  python scripts/extract_base_fields.py
    # 2. score every registered channel against that video's .marks.json:
    T31_DIR=/path/to/cache  python scripts/run_lab.py
PYTHONPATH must include the repo root so `core` imports; scripts/ is on sys.path
automatically, so `import channel_lab` resolves.
"""
import os
import pickle

import numpy as np

import channel_lab as L
from channel_lab import butter_band_energy, channel, morlet_band, validate

DIR = os.environ["T31_DIR"]
MARKS = "Videos/Stabilized/rep3_intermittent_crop.marks.json"
WINGBEAT = (13.657522261938405, 25.0)
LOW = (1.0, 5.0)
BROAD = (0.5, 25.0)

# --- raw base fields (no temporal filter) -----------------------------------
for base in L.BASE_FIELDS:
    @channel(f"raw[{base}]", needs=(base,))
    def _raw(f, meta, _b=base):
        return f[_b]

# --- Morlet band power, several bands x bases -------------------------------
for base in ("tensor_speed", "intensity", "change"):
    @channel(f"morlet_wingbeat[{base}]", needs=(base,))
    def _mw(f, meta, _b=base):
        return morlet_band(f[_b], meta["fps"], WINGBEAT)

    @channel(f"morlet_low[{base}]", needs=(base,))
    def _ml(f, meta, _b=base):
        return morlet_band(f[_b], meta["fps"], LOW)

# --- Butterworth + filtfilt band energy (the cheap swap) --------------------
for base in ("tensor_speed", "intensity"):
    @channel(f"butter_wingbeat[{base}]", needs=(base,))
    def _bw(f, meta, _b=base):
        return butter_band_energy(f[_b], meta["fps"], WINGBEAT)

# --- a ratio channel, to show multi-field composition -----------------------
@channel("ratio_change_over_appearance", needs=("change", "appearance"))
def _ratio(f, meta):
    num = np.asarray(f["change"], np.float32)
    den = np.asarray(f["appearance"], np.float32)
    return num / (np.abs(den) + 1e-3)


def main():
    store = L.FieldStore(DIR)
    res = validate(store, MARKS)
    rows = res["rows"]
    print(f"T={res['T']} ({res['T']/res['fps']:.0f}s)  "
          f"flying bouts={len(res['iv'])}  still bouts={res['n_still_bouts']}\n")

    # best statistic per channel, ranked by |AUC-0.5|
    per_ch = {}
    for name, st, af, asp in rows:
        if name not in per_ch or abs(asp - 0.5) > abs(per_ch[name][2] - 0.5):
            per_ch[name] = (st, af, asp)
    ranked = sorted(per_ch.items(), key=lambda kv: -abs(kv[1][2] - 0.5))
    print(f"{'channel':<34} {'best stat':<20} {'AUCframe':>9} {'AUCspan':>8}")
    print("-" * 74)
    for name, (st, af, asp) in ranked:
        print(f"{name:<34} {st:<20} {af:>9.3f} {asp:>8.3f}")

    with open(os.path.join(DIR, "lab_results.pkl"), "wb") as f:
        pickle.dump(res, f)
    print("\nsaved lab_results.pkl")


if __name__ == "__main__":
    main()
