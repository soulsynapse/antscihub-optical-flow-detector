"""Geometry contract for ROI-first, per-replicate flow processing.

Replicate boxes are the ownership boundary.  A pipeline run may use temporary
synthetic pixels outside a box to give dense flow an edge neighbourhood, but it
must never read source pixels from another replicate.  Only the exact box is
reduced to blocks and cached.

The individual block grids are packed into one sparse atlas on disk.  This keeps
the existing time-major cache and feature machinery while every expensive image
operation is still performed independently per replicate.  Atlas coordinates
are an internal storage detail; source-frame fractions remain the public
geometry used by the GUI and exports.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


LAYOUT_VERSION = 1
ATLAS_SEPARATOR_BLOCKS = 1


@dataclass(frozen=True)
class ReplicateTile:
    replicate_id: int
    label: str
    frac: tuple[float, float, float, float]
    source_box: tuple[int, int, int, int]  # x0, y0, x1, y1
    work_width: int
    work_height: int
    grid: tuple[int, int]                 # ny, nx
    atlas_bbox: tuple[int, int, int, int]  # y0, x0, y1, x1 in blocks

    def to_meta(self) -> dict:
        return {
            "id": self.replicate_id,
            "label": self.label,
            "frac": list(self.frac),
            "source_box": list(self.source_box),
            "work_width": self.work_width,
            "work_height": self.work_height,
            "grid": list(self.grid),
            "atlas_bbox": list(self.atlas_bbox),
        }


@dataclass(frozen=True)
class ReplicateLayout:
    tiles: tuple[ReplicateTile, ...]
    atlas_grid: tuple[int, int]
    scale: float
    geometry_hash: str

    @property
    def work_pixels_per_frame(self) -> int:
        return sum(t.work_width * t.work_height for t in self.tiles)

    @property
    def block_cells(self) -> int:
        return self.atlas_grid[0] * self.atlas_grid[1]


def _canonical_geometry(replicates: list[dict]) -> list[dict]:
    """Geometry-only payload. Labels/calibration do not invalidate flow."""
    out = []
    for rep in sorted(replicates, key=lambda r: int(r["id"])):
        frac = tuple(float(v) for v in rep["frac"])
        out.append({
            "id": int(rep["id"]),
            # UI fractions are serialized decimals; rounding removes harmless
            # binary-float noise without making a visibly moved edge collide.
            "frac": [round(v, 12) for v in frac],
        })
    return out


def geometry_hash(replicates: list[dict]) -> str:
    blob = json.dumps({
        "version": LAYOUT_VERSION,
        "replicates": _canonical_geometry(replicates),
    }, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def validate_replicates(replicates: list[dict]) -> None:
    if not replicates:
        raise ValueError(
            "No replicate boxes are defined. Draw or import boxes in Replicates "
            "before running optical flow.")

    ids: set[int] = set()
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for rep in replicates:
        if "id" not in rep or "frac" not in rep:
            raise ValueError("Every replicate needs an id and frac box.")
        rid = int(rep["id"])
        if rid in ids:
            raise ValueError(f"Replicate id {rid} appears more than once.")
        ids.add(rid)
        frac = tuple(float(v) for v in rep["frac"])
        if len(frac) != 4:
            raise ValueError(f"Replicate {rid} does not have four box coordinates.")
        x0, y0, x1, y1 = frac
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(
                f"Replicate {rid} has an invalid fractional box {frac}; "
                "coordinates must be ordered and within 0..1.")
        boxes.append((rid, frac))

    # Source ownership must be unambiguous. Touching edges are fine; positive
    # area overlap would let the same animal/pixels contribute to two isolates.
    for i, (aid, a) in enumerate(boxes):
        ax0, ay0, ax1, ay1 = a
        for bid, b in boxes[i + 1:]:
            bx0, by0, bx1, by1 = b
            iw = min(ax1, bx1) - max(ax0, bx0)
            ih = min(ay1, by1) - max(ay0, by0)
            if iw > 1e-12 and ih > 1e-12:
                raise ValueError(
                    f"Replicate boxes {aid} and {bid} overlap. ROI-first flow "
                    "requires non-overlapping source ownership; adjust either box.")


def build_layout(replicates: list[dict], src_width: int, src_height: int,
                 scale: float, block_size: int) -> ReplicateLayout:
    validate_replicates(replicates)
    if src_width <= 0 or src_height <= 0:
        raise ValueError("Source dimensions must be positive.")
    if scale <= 0 or block_size <= 0:
        raise ValueError("Downsample scale and block size must be positive.")

    pending = []
    for rep in sorted(replicates, key=lambda r: int(r["id"])):
        rid = int(rep["id"])
        frac = tuple(float(v) for v in rep["frac"])
        x0, y0, x1, y1 = frac
        # Round both sides so adjacent fractional boxes share an integer edge
        # instead of overlapping by one pixel through floor/ceil asymmetry.
        sx0 = max(0, min(src_width - 1, int(round(x0 * src_width))))
        sy0 = max(0, min(src_height - 1, int(round(y0 * src_height))))
        sx1 = max(sx0 + 1, min(src_width, int(round(x1 * src_width))))
        sy1 = max(sy0 + 1, min(src_height, int(round(y1 * src_height))))
        work_w = max(1, int(round((sx1 - sx0) * scale)))
        work_h = max(1, int(round((sy1 - sy0) * scale)))
        # Partial edge blocks are retained with valid-pixel weighting; ceil is
        # therefore intentional and no strip of the drawn ROI is discarded.
        nx = (work_w + block_size - 1) // block_size
        ny = (work_h + block_size - 1) // block_size
        pending.append((rep, frac, (sx0, sy0, sx1, sy1), work_w, work_h,
                        ny, nx))

    atlas_w = max(p[-1] for p in pending)
    atlas_h = sum(p[-2] for p in pending) + \
        ATLAS_SEPARATOR_BLOCKS * max(0, len(pending) - 1)
    tiles = []
    ay = 0
    for rep, frac, source_box, work_w, work_h, ny, nx in pending:
        tiles.append(ReplicateTile(
            replicate_id=int(rep["id"]),
            label=str(rep.get("label", f"rep{rep['id']}")),
            frac=frac,
            source_box=source_box,
            work_width=work_w,
            work_height=work_h,
            grid=(ny, nx),
            atlas_bbox=(ay, 0, ay + ny, nx),
        ))
        ay += ny + ATLAS_SEPARATOR_BLOCKS

    return ReplicateLayout(
        tiles=tuple(tiles),
        atlas_grid=(atlas_h, atlas_w),
        scale=float(scale),
        geometry_hash=geometry_hash(replicates),
    )


def tiles_from_meta(meta: dict) -> list[dict]:
    return list(meta.get("replicate_tiles", []))


def block_weight_plane(meta: dict) -> np.ndarray:
    """Per-block valid-area weight over the atlas grid: (ny, nx) float32 in (0, 1].

    ``build_layout`` rounds each replicate's block grid up (ceil), so the final
    row/column of a box can be a fraction of a block tall/wide -- as little as one
    working pixel. ``reduce_to_blocks`` already averages only the valid pixels, so
    a sliver block carries a legitimate *value*; but every downstream area count
    (clump size, passing-block count, strength denominator) treated each block as
    a full unit of area, which let a row of one-pixel edge slivers masquerade as a
    real clump. The weight is that block's valid pixel area over a full block, so a
    full block is 1.0 and a 1-of-16-px-tall block is 1/16.

    Cells outside any tile (atlas separators) get 0: they own no source pixels and
    must never contribute area. Legacy full-frame caches have no replicate tiles
    and no partial blocks, so every block weighs 1.
    """
    ny, nx = (int(v) for v in meta["grid"])
    tiles = meta.get("replicate_tiles")
    if not tiles:
        return np.ones((ny, nx), np.float32)

    block = int(meta["block_size"])
    plane = np.zeros((ny, nx), np.float32)
    for tile in tiles:
        ty0, tx0, ty1, tx1 = (int(v) for v in tile["atlas_bbox"])
        # Absent work dims (older/partial meta) mean we cannot know the remainder,
        # so assume every block is full -- weight 1, the pre-weighting behaviour.
        work_w = int(tile.get("work_width", (tx1 - tx0) * block))
        work_h = int(tile.get("work_height", (ty1 - ty0) * block))
        # Valid extent of each block row/column: full `block`, except the last,
        # which spans only the remainder of the box's working dimension. Clip to
        # [0, block] so malformed meta (more atlas cells than the grid covers)
        # can never inject a negative weight into a clump-area sum.
        hs = np.clip(work_h - np.arange(ty1 - ty0) * block, 0, block)
        ws = np.clip(work_w - np.arange(tx1 - tx0) * block, 0, block)
        plane[ty0:ty1, tx0:tx1] = \
            np.outer(hs, ws).astype(np.float32) / float(block * block)
    return plane
