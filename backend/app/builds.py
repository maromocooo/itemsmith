"""GLB generation: masks from SAM, then one glbsmith job per product, tracked by polling.

glbsmith has its own queue and concurrency setting, so jobs are simply handed over. A build
lives on the first member of its group and is picked up again after a restart.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import sam
from .geometry import pad_box
from .glbsmith import GlbsmithClient, GlbsmithError, GlbsmithUnreachable
from .search import group_members

log = logging.getLogger(__name__)

ACTIVE = ("preparing", "queued", "running")
CROP_PADDING = 0.35
MIN_CROP_SIDE = 1024
MAX_PHOTOS = 8  # glbsmith's limit
# Beyond this share of the photo a crop adds nothing over the whole photo.
MAX_CROP_SHARE = 0.7
POLL_SECONDS = float(os.environ.get("GLBSMITH_POLL_SECONDS", "5"))


def is_active(lead: dict) -> bool:
    return (lead.get("build") or {}).get("status") in ACTIVE


def _jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=92)
    return buffer.getvalue()


def _png(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buffer, "PNG")
    return buffer.getvalue()


def box_mask(box: list[float], size: tuple[int, int]) -> np.ndarray:
    """The box itself, for when SAM rejected its mask."""
    mask = np.zeros((size[1], size[0]), dtype=bool)
    x, y, w, h = (round(v) for v in box)
    mask[y : y + h, x : x + w] = True
    return mask


def build_photos(image: Image.Image, mask: np.ndarray) -> list[tuple[bytes, bytes]]:
    """A close crop as the main photo, because glbsmith shrinks big photos, then the whole photo for context."""
    ys, xs = np.nonzero(mask)
    x1, y1 = int(xs.min()), int(ys.min())
    box = [x1, y1, int(xs.max()) - x1 + 1, int(ys.max()) - y1 + 1]
    x, y, w, h = pad_box(box, CROP_PADDING, 40, image.width, image.height)
    x, y, w, h = round(x), round(y), round(w), round(h)
    whole = (_jpeg(image), _png(mask))
    if w * h > MAX_CROP_SHARE * image.width * image.height:
        return [whole]
    crop = image.crop((x, y, x + w, y + h))
    part = mask[y : y + h, x : x + w]
    if max(crop.size) < MIN_CROP_SIDE:
        scale = MIN_CROP_SIDE / max(crop.size)
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)
        part = np.asarray(Image.fromarray(part).resize(crop.size, Image.NEAREST))
    return [(_jpeg(crop), _png(part)), whole]


def product_photo_files(root: Path, lead: dict) -> list[Path]:
    photos = (lead.get("size") or {}).get("photos") or []
    return [root / "photos" / lead["id"] / f"{p['id']}.jpg" for p in photos if p.get("selected")]


def warnings_from(job: dict) -> list[str]:
    report = job.get("report") or {}
    warnings = []
    labels = {"width_x": "幅", "depth_y": "奥行き", "height_z": "高さ"}
    for axis, sizes in (report.get("smaller_than_box") or {}).items():
        warnings.append(f"{labels.get(axis, axis)}が指定より小さい({sizes['model_cm']} cm / 指定 {sizes['box_cm']} cm)")
    check = report.get("photo_check") or {}
    if check.get("verdict") == "fail":
        warnings.append("写真との重ね合わせ確認に合格しないまま書き出されました")
    if report.get("flat_materials_warning"):
        warnings.append("単色のままの素材があります")
    return warnings


class BuildManager:
    def __init__(self, store, client: GlbsmithClient, poll_seconds: float = POLL_SECONDS) -> None:
        self.store = store
        self.client = client
        self.poll_seconds = poll_seconds
        self._prepare: asyncio.Queue[tuple[str, list[str]]] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self.connected = True

    def start(self) -> None:
        self._prepare = asyncio.Queue()
        self._tasks = [asyncio.create_task(self._prepare_loop()), asyncio.create_task(self._poll_loop())]
        for session in self.store.all():
            waiting = [o["id"] for o in session.get("objects", []) if (o.get("build") or {}).get("status") == "preparing"]
            if waiting:
                self._prepare.put_nowait((session["id"], waiting))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    def request(self, session: dict, leads: list[dict], agent: str, model: str | None) -> None:
        """Each lead's size is already final; it is copied so later edits do not rewrite history."""
        for lead in leads:
            size = lead["size"]
            lead["build"] = {
                "status": "preparing", "job_id": None, "agent": agent, "model": model,
                "width": size["width"], "depth": size["depth"], "height": size["height"],
                "kind": size.get("kind") or "single", "requested_at": time.time(),
                "error": None, "message": None, "seconds": None, "tokens": None, "warnings": [],
                "has_model": False, "has_render": False,
                "product_photos": sum(1 for p in size.get("photos") or [] if p.get("selected")),
            }
        self.store.save(session["id"])
        self._prepare.put_nowait((session["id"], [lead["id"] for lead in leads]))

    async def cancel(self, session: dict, lead: dict) -> None:
        build = lead["build"]
        if build["job_id"]:
            await self.client.cancel(build["job_id"])
        build["status"] = "cancelled"
        self.store.save(session["id"])

    def _targets(self, session: dict, lead: dict) -> list[dict]:
        members = group_members(session, lead)
        if lead["build"]["kind"] == "identical":
            return [max(members, key=lambda m: m["bbox"][2] * m["bbox"][3])]
        return members

    async def _prepare_loop(self) -> None:
        while True:
            sid, oids = await self._prepare.get()
            try:
                await self._prepare_batch(sid, oids)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("preparing builds failed for %s", sid)

    async def _prepare_batch(self, sid: str, oids: list[str]) -> None:
        try:
            session = self.store.get(sid)
        except KeyError:
            return
        by_id = {o["id"]: o for o in session["objects"]}
        leads = [by_id[oid] for oid in oids if oid in by_id and (by_id[oid].get("build") or {}).get("status") == "preparing"]
        if not leads:
            return
        with Image.open(self.store.dir(sid) / "original.jpg") as opened:
            image = opened.convert("RGB")
        targets = [self._targets(session, lead) for lead in leads]
        boxes = [member["bbox"] for group in targets for member in group]
        try:
            found = iter(await asyncio.to_thread(sam.masks, image, boxes))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a box still tells glbsmith which object is meant
            log.warning("SAM failed, using boxes as masks: %s", exc)
            found = iter([None] * len(boxes))
        prepared = []
        for lead, group in zip(leads, targets):
            mask = np.zeros((image.height, image.width), dtype=bool)
            for member in group:
                part = next(found)
                mask |= part if part is not None else box_mask(member["bbox"], image.size)
            photos = await asyncio.to_thread(build_photos, image, mask)
            # The photo of the real item stays first: it decides the proportions and is what the check uses.
            extra = product_photo_files(self.store.dir(sid), lead)[: MAX_PHOTOS - len(photos)]
            photos += [(path.read_bytes(), b"") for path in extra if path.exists()]
            prepared.append((lead, photos))
        for lead, photos in prepared:
            build = lead["build"]
            if build["status"] != "preparing":  # cancelled in the meantime
                continue
            try:
                job = await self.client.submit(
                    photos, width=build["width"], depth=build["depth"], height=build["height"],
                    agent=build["agent"], model=build["model"], name=lead["confirmed_name"],
                )
                build.update(status=job.get("status", "queued"), job_id=job["id"])
            except GlbsmithError as exc:
                build.update(status="failed", error=str(exc)[:1000])
            self.store.save(sid)

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(self.poll_seconds)
            try:
                await self._poll_once()
                self.connected = True
            except GlbsmithUnreachable:
                self.connected = False  # builds stay as they are and resume with the server
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("polling glbsmith failed")

    async def _poll_once(self) -> None:
        for session in self.store.all():
            for lead in session.get("objects", []):
                build = lead.get("build") or {}
                if build.get("job_id") and build["status"] in ("queued", "running"):
                    await self._refresh(session, lead)

    async def _refresh(self, session: dict, lead: dict) -> None:
        build = lead["build"]
        job = await self.client.job(build["job_id"])
        sid = session["id"]
        if job is None:
            build.update(status="failed", error="glbsmith にジョブが見つかりません")
            self.store.save(sid)
            return
        status = job.get("status")
        if status == "done":
            await self._collect(sid, lead, job)
            build.update(has_model=True)
        if status in ("done", "failed", "cancelled"):
            build.update(
                status=status, seconds=job.get("seconds"), tokens=job.get("tokens"),
                message=(job.get("message") or "")[:2000], error=(job.get("error") or None) if status == "failed" else None,
                warnings=warnings_from(job),
            )
            self.store.save(sid)
        elif status != build["status"]:
            build["status"] = status
            self.store.save(sid)

    async def _collect(self, sid: str, lead: dict, job: dict) -> None:
        """Copied here so results stay available without the glbsmith server."""
        models = self.store.dir(sid) / "models"
        models.mkdir(exist_ok=True)
        _write_atomic(models / f"{lead['id']}.glb", await self.client.model(job["id"]))
        try:
            render = await self.client.latest_render(job["id"])
        except GlbsmithUnreachable:
            raise
        except GlbsmithError:
            render = None
        if render:
            _write_atomic(models / f"{lead['id']}.png", render)
            lead["build"]["has_render"] = True


def _write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
