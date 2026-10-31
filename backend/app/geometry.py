"""Box helpers. Boxes are [x, y, w, h] in original-image pixels."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Tile:
    x: int
    y: int
    w: int
    h: int


def make_tiles(width: int, height: int, overlap: float = 0.2) -> list[Tile]:
    tile_w = round(width * (1 + overlap) / 2)
    tile_h = round(height * (1 + overlap) / 2)
    return [
        Tile(x, y, tile_w, tile_h)
        for y in (0, height - tile_h)
        for x in (0, width - tile_w)
    ]


def from_normalized(box: list[float], tile: Tile) -> list[float] | None:
    """Convert [x1, y1, x2, y2] on a 0-1000 scale within a tile to image pixels."""
    if len(box) != 4:
        return None
    x1, y1, x2, y2 = (min(max(float(v), 0.0), 1000.0) for v in box)
    if x2 <= x1 or y2 <= y1:
        return None
    return [
        tile.x + x1 / 1000 * tile.w,
        tile.y + y1 / 1000 * tile.h,
        (x2 - x1) / 1000 * tile.w,
        (y2 - y1) / 1000 * tile.h,
    ]


def to_normalized(box: list[float], width: int, height: int) -> list[int]:
    x, y, w, h = box
    return [round(x / width * 1000), round(y / height * 1000), round((x + w) / width * 1000), round((y + h) / height * 1000)]


def intersection(a: list[float], b: list[float]) -> float:
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


def area(a: list[float]) -> float:
    return a[2] * a[3]


def iou(a: list[float], b: list[float]) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(a: list[float], b: list[float]) -> float:
    """Fraction of the smaller box covered by the other one."""
    smaller = min(area(a), area(b))
    return intersection(a, b) / smaller if smaller > 0 else 0.0


def touches_inner_edge(box: list[float], tile: Tile, width: int, height: int, margin: float = 6.0) -> bool:
    """True when the box is cut by a tile border that is not also an image border."""
    x, y, w, h = box
    return (
        (tile.x > 0 and x <= tile.x + margin)
        or (tile.y > 0 and y <= tile.y + margin)
        or (tile.x + tile.w < width and x + w >= tile.x + tile.w - margin)
        or (tile.y + tile.h < height and y + h >= tile.y + tile.h - margin)
    )


def merge_candidates(candidates: list[dict]) -> list[dict]:
    """Merge duplicates found by overlapping views.

    Candidates carry `box`, `category` and `rank` (lower wins). A small object sitting on a
    larger one (mouse on mouse pad) must survive, so containment only merges same-category
    boxes; plain high IoU merges regardless of category.
    """
    kept: list[dict] = []
    for cand in sorted(candidates, key=lambda c: (c["rank"], -area(c["box"]))):
        duplicate = any(
            iou(cand["box"], other["box"]) >= 0.5
            or (
                cand["category"] == other["category"]
                and containment(cand["box"], other["box"]) >= 0.8
            )
            for other in kept
        )
        if not duplicate:
            kept.append(cand)
    return kept


def clamp_box(box: list[float], width: int, height: int) -> list[int]:
    x, y, w, h = box
    x1, y1 = max(round(x), 0), max(round(y), 0)
    x2, y2 = min(round(x + w), width), min(round(y + h), height)
    return [x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)]


def pad_box(box: list[float], ratio: float, min_pad: float, width: int, height: int) -> list[int]:
    x, y, w, h = box
    px, py = max(w * ratio, min_pad), max(h * ratio, min_pad)
    return clamp_box([x - px, y - py, w + 2 * px, h + 2 * py], width, height)
