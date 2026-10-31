"""Object detection: agent proposals over overlapping views -> merge -> SAM -> agent review."""
from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import Awaitable, Callable

from PIL import Image, ImageDraw, ImageFont

from . import sam
from .agents.base import AgentAdapter, AgentRequest
from .geometry import (
    Tile,
    area,
    clamp_box,
    from_normalized,
    make_tiles,
    merge_candidates,
    pad_box,
    to_normalized,
    touches_inner_edge,
)

log = logging.getLogger(__name__)

CATEGORIES = [
    "display", "audio", "input_device", "camera", "lighting", "computer", "tablet_phone",
    "appliance", "furniture", "storage", "mount_arm", "cable", "tableware", "book_stationery",
    "wearable", "accessory", "other",
]
PROPOSAL_TIMEOUT = 480.0
REVIEW_TIMEOUT = 480.0

_box_props = {name: {"type": "number"} for name in ("x1", "y1", "x2", "y2")}
_object_props = {
    "label": {"type": "string"},
    "category": {"type": "string", "enum": CATEGORIES},
    **_box_props,
}
_object_schema = {
    "type": "object",
    "properties": _object_props,
    "required": list(_object_props),
    "additionalProperties": False,
}
PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {"objects": {"type": "array", "items": _object_schema}},
    "required": ["objects"],
    "additionalProperties": False,
}
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "keep": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "label": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                },
                "required": ["id", "label", "category"],
                "additionalProperties": False,
            },
        },
        "add": {"type": "array", "items": _object_schema},
        "same_product": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
    },
    "required": ["keep", "add", "same_product"],
    "additionalProperties": False,
}

SPLIT_RULES = """分割ルール:
- 物理的に別の物は別オブジェクトにする。例: ヘッドフォンとそこから垂れるケーブル、スタンドと載っている機器、マウスとマウスパッド、マグカップとコースター。
- 左右で分かれた製品(左右のスピーカー、分割キーボードの左右)は別々のオブジェクトにする。
- 製品の構成部品(キーボードのキースイッチ、モニターの画面、引き出しの取っ手)は分けない。
- 製品・市販品として特定しうる物をすべて、小さい物や暗い物、影の中や一部が隠れた物も含めて挙げる。家具、収納、壁掛けボード、モニターアーム、ケーブル類も対象。
- 壁、床、天井、影、反射、配線の見えない部分は挙げない。
- 枠は見えている部分全体をちょうど囲む。"""


def proposal_prompt(is_tile: bool) -> str:
    scope = (
        "この画像は1枚の写真の一部を切り出したものです。この範囲に写っている物を挙げてください。端で切れている物も、見えている部分を囲みます。"
        if is_tile
        else "この写真に写っている物を挙げてください。"
    )
    return f"""あなたは物体検出の専門家です。{scope}

{SPLIT_RULES}

出力: 各オブジェクトについて、
- label: 日本語の簡潔な名前(例: 「ヘッドフォン」「ヘッドフォンのケーブル」「右スピーカー」)。わかる範囲の見た目も含めてよいが、商品名の断定はしない。
- category: 指定の値から選ぶ。
- x1, y1, x2, y2: 画像の左上を(0,0)、右下を(1000,1000)とする正規化座標の枠。

ツールは使わず、画像だけを見て答えてください。"""


def review_prompt(items: list[dict], width: int, height: int) -> str:
    listing = "\n".join(
        f"- id={i}: {it['label']} ({it['category']}) 枠={to_normalized(it['box'], width, height)}"
        for i, it in enumerate(items)
    )
    return f"""画像は写真に検出候補の番号付きの枠を描いたものです。候補を点検して最終リストを作ってください。

候補([x1,y1,x2,y2] は写真全体を0〜1000とする正規化座標):
{listing}

{SPLIT_RULES}

やること:
1. keep: 残す候補の id を返す。同じ物の重複は最も枠が正確な1つだけ残し、物が写っていない枠や壁・床の枠は外す。label と category は必要なら直す。
2. add: 枠がない物が写っていれば、label、category、x1,y1,x2,y2(写真全体を0〜1000とする正規化座標)で追加する。ないなら空配列。
3. same_product: keep の中で、枠は別でも1つの商品として買う物の id の組を返す。枠は分けたまま、商品検索だけをまとめるためのものです。ないなら空配列。
   - 組にする例: 左右のスピーカー、分割キーボードの左右、昇降デスクの天板と脚とコントローラー、ヘッドフォンとそこにつながるケーブル、途切れて複数の枠に分かれた同じケーブル。
   - 組にしない例: 単体の商品名が付く付属品や周辺機器(離れて置かれたバッテリーや充電器、モニターアーム、スタンド、ハンガー、マウスパッド)、デスクに載っているだけの物、見た目が似ているだけの物。

ツールは使わず、画像だけを見て答えてください。"""


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def save_view(image: Image.Image, path: Path, max_side: int) -> None:
    view = image.copy()
    view.thumbnail((max_side, max_side))
    view.save(path, "JPEG", quality=88)


def draw_numbered(image: Image.Image, items: list[dict], path: Path, max_side: int = 2000) -> None:
    scale = min(1.0, max_side / max(image.size))
    view = image.resize((round(image.width * scale), round(image.height * scale)))
    draw = ImageDraw.Draw(view)
    font = _font(20)
    palette = ["#ff3b30", "#34c759", "#0a84ff", "#ffcc00", "#ff2dff", "#00e5ff"]
    for i, item in enumerate(items):
        x, y, w, h = (v * scale for v in item["box"])
        color = palette[i % len(palette)]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        text = str(i)
        tx1, ty1, tx2, ty2 = draw.textbbox((x, y), text, font=font)
        draw.rectangle([tx1 - 2, ty1 - 2, tx2 + 2, ty2 + 2], fill=color)
        draw.text((x, y), text, fill="#000000", font=font)
    view.save(path, "JPEG", quality=88)


def draw_target(image: Image.Image, boxes: list[list[float]], path: Path, max_side: int = 1600) -> None:
    """Whole photo with the target marked, so an agent sees the object in context."""
    scale = min(1.0, max_side / max(image.size))
    view = image.resize((round(image.width * scale), round(image.height * scale)))
    draw = ImageDraw.Draw(view)
    for box in boxes:
        x, y, w, h = (v * scale for v in box)
        draw.rectangle([x, y, x + w, y + h], outline="#ff3b30", width=4)
    view.save(path, "JPEG", quality=85)


def write_crop(image: Image.Image, box: list[float], path: Path) -> None:
    x, y, w, h = pad_box(box, 0.05, 6, image.width, image.height)
    crop = image.crop((x, y, x + w, y + h))
    crop.thumbnail((1200, 1200))
    crop.save(path, "JPEG", quality=90)


def _parse_objects(raw: dict, tile: Tile, width: int, height: int, rank: int, check_edge: bool) -> list[dict]:
    parsed = []
    for obj in raw.get("objects", []) if "objects" in raw else raw.get("add", []):
        box = from_normalized([obj.get(k, 0) for k in ("x1", "y1", "x2", "y2")], tile)
        label = str(obj.get("label", "")).strip()
        if box is None or not label or area(box) < 16:
            continue
        cut = check_edge and touches_inner_edge(box, tile, width, height)
        parsed.append(
            {
                "label": label[:80],
                "category": obj.get("category", "other"),
                "box": box,
                "rank": rank + (1 if cut else 0),
            }
        )
    return parsed


StageCallback = Callable[[str, str], Awaitable[None]]


async def detect_objects(
    image_path: Path,
    work_dir: Path,
    adapter: AgentAdapter,
    model: str,
    on_stage: StageCallback,
) -> tuple[list[dict], list[str]]:
    """Return final objects ({label, category, box}) and warnings for the UI."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    warnings: list[str] = []
    work_dir.mkdir(parents=True, exist_ok=True)

    await on_stage("proposing", "写真全体と4分割した画像からオブジェクトを探しています")
    full_tile = Tile(0, 0, width, height)
    full_path = work_dir / "view_full.jpg"
    save_view(image, full_path, 2000)
    views: list[tuple[Tile, Path]] = [(full_tile, full_path)]
    for i, tile in enumerate(make_tiles(width, height)):
        path = work_dir / f"view_tile{i}.jpg"
        save_view(image.crop((tile.x, tile.y, tile.x + tile.w, tile.y + tile.h)), path, 1800)
        views.append((tile, path))

    async def propose(tile: Tile, path: Path) -> dict:
        return await adapter.run(
            AgentRequest(
                prompt=proposal_prompt(tile is not full_tile),
                schema=PROPOSAL_SCHEMA,
                model=model,
                images=[path],
                timeout=PROPOSAL_TIMEOUT,
            )
        )

    results = await asyncio.gather(*(propose(t, p) for t, p in views), return_exceptions=True)
    candidates: list[dict] = []
    failures = 0
    for (tile, _), result in zip(views, results):
        if isinstance(result, BaseException):
            failures += 1
            log.warning("proposal failed: %s", result)
            continue
        is_full = tile is full_tile
        candidates += _parse_objects(result, tile, width, height, 0 if is_full else 1, not is_full)
    if failures == len(views):
        raise RuntimeError(f"検出に失敗しました: {results[0]}")
    if failures:
        warnings.append(f"{len(views)} 視点のうち {failures} 視点の検出に失敗したため、検出漏れの可能性があります")

    items = merge_candidates(candidates)
    log.info("proposals: %d candidates -> %d merged", len(candidates), len(items))

    await on_stage("refining", f"{len(items)} 件の枠をオブジェクトに合わせています(SAM 2.1)")
    sam_ok = await _refine(image, items, warnings)

    await on_stage("reviewing", f"{len(items)} 件の候補を点検しています")
    overlay = work_dir / "review.jpg"
    draw_numbered(image, items, overlay)
    try:
        review = await adapter.run(
            AgentRequest(
                prompt=review_prompt(items, width, height),
                schema=REVIEW_SCHEMA,
                model=model,
                images=[overlay],
                timeout=REVIEW_TIMEOUT,
            )
        )
        items, added = _apply_review(review, items, full_tile, width, height)
        if added and sam_ok:
            await on_stage("refining", f"追加された {len(added)} 件の枠を調整しています")
            await _refine(image, added, warnings)
    except Exception as exc:  # the unreviewed list is still usable
        log.warning("review failed: %s", exc)
        warnings.append("候補の点検に失敗したため、重複や誤検出が残っている可能性があります")

    for item in items:
        item["box"] = clamp_box(item["box"], width, height)
    items.sort(key=lambda it: (round(it["box"][1] / 150), it["box"][0]))
    return items, warnings


async def _refine(image: Image.Image, items: list[dict], warnings: list[str]) -> bool:
    try:
        boxes, count = await asyncio.to_thread(sam.tighten, image, [it["box"] for it in items])
    except Exception as exc:
        log.warning("SAM failed: %s", exc)
        warnings.append("SAM を実行できなかったため、枠は精密化されていません")
        return False
    for item, box in zip(items, boxes):
        item["box"] = box
    log.info("SAM refined %d/%d", count, len(items))
    return True


def _apply_review(
    review: dict, items: list[dict], full: Tile, width: int, height: int
) -> tuple[list[dict], list[dict]]:
    kept: dict[int, dict] = {}
    for entry in review.get("keep", []):
        idx = entry.get("id")
        if not isinstance(idx, int) or not 0 <= idx < len(items) or idx in kept:
            continue
        kept[idx] = {
            **items[idx],
            "label": str(entry.get("label") or items[idx]["label"])[:80],
            "category": entry.get("category") or items[idx]["category"],
        }
    added = _parse_objects({"add": review.get("add", [])}, full, width, height, 0, False)
    if not kept:  # a review that keeps nothing is a bad answer, not an empty photo
        return items + added, added
    grouped: set[int] = set()
    for n, ids in enumerate(review.get("same_product", []), start=1):
        members = [i for i in dict.fromkeys(ids) if isinstance(i, int) and i in kept and i not in grouped]
        if len(members) >= 2:
            grouped.update(members)
            for i in members:
                kept[i]["group"] = f"g{n}"
    return list(kept.values()) + added, added


async def run_detection(session: dict, store, adapters: dict[str, AgentAdapter]) -> None:
    """Background task: fills session["objects"] and writes crops."""
    sid = session["id"]
    root = store.dir(sid)
    detect = session["detect"]

    async def on_stage(stage: str, message: str) -> None:
        detect.update(stage=stage, message=message)
        store.save(sid)

    try:
        items, warnings = await detect_objects(
            root / "original.jpg", root / "work", adapters[detect["provider"]], detect["model"], on_stage
        )
        image = Image.open(root / "original.jpg").convert("RGB")
        (root / "crops").mkdir(exist_ok=True)
        objects = []
        for i, item in enumerate(items, start=1):
            oid = f"o{i}"
            write_crop(image, item["box"], root / "crops" / f"{oid}.jpg")
            objects.append(
                {
                    "id": oid,
                    "number": i,
                    "label": item["label"],
                    "category": item["category"],
                    "bbox": item["box"],
                    "group": item.get("group"),
                    "hint": "",
                    "excluded": False,
                    "status": "idle",
                    "activity": None,
                    "started_at": None,
                    "queued_at": None,
                    "result": None,
                    "history": [],
                    "feedback": "",
                    "confirmed_name": None,
                    "confirmed_by": None,
                    "error": None,
                }
            )
        session["objects"] = objects
        session["status"] = "ready"
        detect.update(stage="done", message=f"{len(objects)} 件検出しました", warnings=warnings, finished_at=time.time())
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.exception("detection failed")
        session["status"] = "detect_failed"
        detect.update(stage="failed", message=str(exc)[:1000])
    finally:
        shutil.rmtree(root / "work", ignore_errors=True)
    store.save(sid)
