from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from .builds import BuildManager, is_active
from .detect import run_detection
from .glbsmith import GlbsmithClient, GlbsmithError, GlbsmithUnreachable
from .models_catalog import ADAPTERS, catalog_with_versions, label_claude_options, validate_choice
from .search import SearchManager, confirmed_name, group_members, merge_results, representatives
from .sizes import MAX_SIZE_CM, SizeManager
from .store import SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data")).resolve()
FRONTEND_DIST = Path(os.environ.get("FRONTEND_DIST", ROOT / "frontend" / "dist")).resolve()
MAX_UPLOAD = 50 * 1024 * 1024
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1", "testserver"}
SSE_MIN_INTERVAL = 0.25

store = SessionStore(DATA_DIR)
manager = SearchManager(store, ADAPTERS)
sizes = SizeManager(store, ADAPTERS)
glbsmith = GlbsmithClient()
builds = BuildManager(store, glbsmith)
detection_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(_: FastAPI):
    for session in store.all():
        if session["status"] == "detecting":  # the server stopped mid-detection
            session["status"] = "detect_failed"
            session["detect"].update(stage="failed", message="サーバー再起動で検出が中断されました")
            store.save(session["id"])
    manager.start()
    sizes.start()
    builds.start()
    yield
    for task in list(detection_tasks):
        task.cancel()
    await manager.stop()
    await sizes.stop()
    await builds.stop()


app = FastAPI(title="itemsmith", lifespan=lifespan)


@app.middleware("http")
async def local_only(request: Request, call_next):
    """This app can spend subscription quota, so other websites must not be able to drive it."""
    host = urlsplit(f"//{request.headers.get('host', '')}").hostname
    origin = request.headers.get("origin")
    if host not in LOCAL_HOSTS or (origin and urlsplit(origin).hostname not in LOCAL_HOSTS):
        return JSONResponse({"detail": "localhost 以外からのアクセスは許可されていません"}, status_code=403)
    return await call_next(request)


def _session(sid: str) -> dict:
    try:
        return store.get(sid)
    except KeyError:
        raise HTTPException(404, "セッションが見つかりません") from None


def _object(session: dict, oid: str) -> dict:
    obj = next((o for o in session.get("objects", []) if o["id"] == oid), None)
    if obj is None:
        raise HTTPException(404, "オブジェクトが見つかりません")
    return obj


def _lead(session: dict, obj: dict) -> dict:
    """Sizes and builds belong to a product and are stored on its first member."""
    return group_members(session, obj)[0]


def _is_busy(session: dict, members: list[dict]) -> bool:
    return any(
        m["status"] in ("queued", "running")
        or (m.get("size") or {}).get("status") in ("queued", "running")
        or is_active(m)
        for m in members
    )


def _check_regroup(session: dict, members: list[dict]) -> None:
    """Regrouping changes what a product is, so it must not race with work on it or orphan a GLB."""
    if _is_busy(session, members):
        raise HTTPException(409, "処理中の製品は変更できません")
    if any(m.get("build") for m in members):
        raise HTTPException(409, "GLB を作成した製品は変更できません")


def _check_choice(provider: str, model: str) -> None:
    if error := validate_choice(provider, model):
        raise HTTPException(400, error)


def _start_detection(session: dict) -> None:
    task = asyncio.create_task(run_detection(session, store, ADAPTERS))
    detection_tasks.add(task)
    task.add_done_callback(detection_tasks.discard)


@app.get("/api/agents")
def get_agents() -> list[dict]:
    return catalog_with_versions()


@app.get("/api/sessions")
def list_sessions() -> list[dict]:
    return [
        {
            "id": s["id"],
            "filename": s["image"]["filename"],
            "created_at": s["created_at"],
            "status": s["status"],
            "object_count": len(s.get("objects", [])),
            "search_started": s.get("search") is not None,
        }
        for s in store.all()
    ]


@app.post("/api/sessions", status_code=201)
async def create_session(
    file: UploadFile = File(...), provider: str = Form(...), model: str = Form(...)
) -> dict:
    _check_choice(provider, model)
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "画像は 50MB までです")
    try:
        with Image.open(io.BytesIO(data)) as raw:
            if raw.format not in ("JPEG", "PNG", "WEBP"):
                raise HTTPException(400, "JPEG / PNG / WebP の画像を指定してください")
            image = ImageOps.exif_transpose(raw).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "画像として読み込めませんでした") from None
    sid = store.new_id()
    store.dir(sid).mkdir(parents=True)
    image.save(store.dir(sid) / "original.jpg", "JPEG", quality=95)
    session = {
        "id": sid,
        "created_at": time.time(),
        "status": "detecting",
        "image": {
            "filename": Path(file.filename or "photo").name[:200],
            "width": image.width,
            "height": image.height,
        },
        "detect": {
            "provider": provider, "model": model, "stage": "queued",
            "message": "検出を開始します", "warnings": [], "started_at": time.time(),
        },
        "search": None,
        "objects": [],
    }
    store.add(session)
    _start_detection(session)
    return session


@app.get("/api/sessions/{sid}")
def get_session(sid: str) -> dict:
    return _session(sid)


@app.delete("/api/sessions/{sid}", status_code=204)
async def delete_session(sid: str) -> None:
    session = _session(sid)
    if session["status"] == "detecting" or _is_busy(session, session["objects"]):
        raise HTTPException(409, "処理中のセッションは削除できません")
    store.delete(sid)


@app.post("/api/sessions/{sid}/redetect")
async def redetect(sid: str) -> dict:
    session = _session(sid)
    if session["status"] == "detecting" or session.get("search"):
        raise HTTPException(409, "検出中、または商品検索を開始済みです")
    session["status"] = "detecting"
    session["objects"] = []
    session["detect"].update(stage="queued", message="検出を開始します", warnings=[], started_at=time.time())
    store.save(sid)
    _start_detection(session)
    return session


@app.get("/api/sessions/{sid}/image")
def get_image(sid: str) -> FileResponse:
    _session(sid)
    return FileResponse(store.dir(sid) / "original.jpg", media_type="image/jpeg")


@app.get("/api/sessions/{sid}/objects/{oid}/crop")
def get_crop(sid: str, oid: str) -> FileResponse:
    obj = _object(_session(sid), oid)
    return FileResponse(store.dir(sid) / "crops" / f"{obj['id']}.jpg", media_type="image/jpeg")


class ObjectPatch(BaseModel):
    hint: str | None = Field(default=None, max_length=1000)
    excluded: bool | None = None
    ungroup: bool = False  # dissolves the whole group
    leave_group: bool = False


@app.patch("/api/sessions/{sid}/objects/{oid}")
async def patch_object(sid: str, oid: str, body: ObjectPatch) -> dict:
    session = _session(sid)
    obj = _object(session, oid)
    if (body.hint is not None or body.excluded is not None) and obj["status"] != "idle":
        raise HTTPException(409, "検索開始後は変更できません")
    if body.hint is not None:
        obj["hint"] = body.hint.strip()
    members = group_members(session, obj)
    if body.excluded is not None:
        for member in members:
            member["excluded"] = body.excluded
    if body.ungroup or body.leave_group:
        _check_regroup(session, members)
        members[0].pop("size", None)
    if body.ungroup or (body.leave_group and len(members) <= 2):
        for member in members:
            member["group"] = None
    elif body.leave_group:
        obj["group"] = None
    store.save(sid)
    return obj


class GroupCreate(BaseModel):
    object_ids: list[str] = Field(max_length=500)
    # Drafts still being typed, as in SearchStart.
    hints: dict[str, str] = Field(default_factory=dict)


@app.post("/api/sessions/{sid}/groups")
async def create_group(sid: str, body: GroupCreate) -> dict:
    """Merge the chosen objects, and the groups they already belong to, into one product."""
    session = _session(sid)
    chosen = {o["id"] for oid in body.object_ids for o in group_members(session, _object(session, oid))}
    members = [o for o in session["objects"] if o["id"] in chosen]
    if len(representatives(members)) < 2:
        raise HTTPException(400, "2 件以上選択してください")
    _check_regroup(session, members)
    for oid, hint in body.hints.items():
        _object(session, oid)["hint"] = hint.strip()[:1000]
    hint = " / ".join(dict.fromkeys(o["hint"] for o in members if o["hint"]))[:1000]
    group = f"u{uuid.uuid4().hex[:8]}"
    merged = merge_results(members) if session.get("search") else {}
    for member in members:
        member.update(group=group, excluded=False, hint="", **merged)
        member.pop("size", None)
    members[0]["hint"] = hint  # only the first member's hint field is shown
    store.save(sid)
    return session


class SearchStart(BaseModel):
    provider: str
    model: str
    # Sent with the start request so drafts still being typed cannot be lost to a racing autosave.
    hints: dict[str, str] = Field(default_factory=dict)


@app.post("/api/sessions/{sid}/search")
async def start_search(sid: str, body: SearchStart) -> dict:
    session = _session(sid)
    _check_choice(body.provider, body.model)
    if session["status"] != "ready":
        raise HTTPException(409, "検出が完了していません")
    if session.get("search"):
        raise HTTPException(409, "商品検索は開始済みです")
    for oid, hint in body.hints.items():
        _object(session, oid)["hint"] = hint.strip()[:1000]
    targets = [o for o in session["objects"] if not o["excluded"]]
    if not targets:
        raise HTTPException(400, "検索対象のオブジェクトがありません")
    session["search"] = {"provider": body.provider, "model": body.model, "started_at": time.time()}
    for obj in representatives(targets):
        manager.enqueue(session, obj)
    return session


class Verdict(BaseModel):
    verdict: str  # correct | wrong | manual | reopen
    name: str | None = Field(default=None, max_length=300)


@app.post("/api/sessions/{sid}/objects/{oid}/verdict")
async def post_verdict(sid: str, oid: str, body: Verdict) -> dict:
    session = _session(sid)
    obj = _object(session, oid)
    status = obj["status"]
    if body.verdict == "correct" and status == "done":
        fields = {"status": "confirmed", "confirmed_name": confirmed_name(obj["result"]), "confirmed_by": "agent"}
    elif body.verdict == "wrong" and status == "done":
        fields = {"status": "rejected"}
    elif body.verdict == "manual" and status in ("rejected", "failed"):
        name = (body.name or "").strip()
        if not name:
            raise HTTPException(400, "商品名を入力してください")
        fields = {"status": "confirmed", "confirmed_name": name, "confirmed_by": "user", "error": None}
    elif body.verdict == "reopen" and status == "confirmed":
        if obj["confirmed_by"] == "user":
            reopened = "rejected" if obj["result"] else "failed"
        else:
            reopened = "done"
        fields = {"status": reopened, "confirmed_name": None, "confirmed_by": None}
    else:
        raise HTTPException(409, f"現在の状態({status})ではこの操作はできません")
    members = group_members(session, obj)
    if body.verdict == "reopen" and _is_busy(session, members):
        raise HTTPException(409, "サイズ取得または GLB 作成の処理中です")
    for member in members:
        member.update(fields)
    # A size belongs to the name it was looked up for.
    if (members[0].get("size") or {}).get("for_name") != members[0]["confirmed_name"] and members[0]["confirmed_name"]:
        members[0].pop("size", None)
    store.save(sid)
    return obj


class Retry(BaseModel):
    feedback: str = Field(default="", max_length=1000)


@app.post("/api/sessions/{sid}/objects/{oid}/retry")
async def retry_object(sid: str, oid: str, body: Retry) -> dict:
    session = _session(sid)
    obj = _object(session, oid)
    feedback = body.feedback.strip()
    if obj["status"] == "rejected":
        if not feedback:
            raise HTTPException(400, "特徴や商品名を入力してください")
        for member in group_members(session, obj):
            member["history"].append({"result": member["result"], "feedback": feedback})
            member["result"] = None
    elif obj["status"] == "failed":
        if feedback:
            obj["hint"] = f"{obj['hint']} / {feedback}".strip(" /")
    else:
        raise HTTPException(409, "再検索できる状態ではありません")
    manager.enqueue(session, obj)
    return obj


class SizeStart(BaseModel):
    provider: str
    model: str
    object_ids: list[str] = Field(max_length=500)


def _confirmed_leads(session: dict, object_ids: list[str]) -> list[dict]:
    leads = representatives([_lead(session, _object(session, oid)) for oid in object_ids])
    if not leads:
        raise HTTPException(400, "製品を選択してください")
    if any(lead["status"] != "confirmed" for lead in leads):
        raise HTTPException(409, "商品名が確定していない製品があります")
    return leads


@app.post("/api/sessions/{sid}/sizes")
async def start_sizes(sid: str, body: SizeStart) -> dict:
    session = _session(sid)
    _check_choice(body.provider, body.model)
    leads = _confirmed_leads(session, body.object_ids)
    if any((lead.get("size") or {}).get("status") in ("queued", "running") for lead in leads):
        raise HTTPException(409, "サイズを取得中の製品があります")
    session["size_agent"] = {"provider": body.provider, "model": body.model}
    for lead in leads:
        sizes.enqueue(session, lead)
    return session


class SizePatch(BaseModel):
    width: float | None = Field(default=None, gt=0, le=MAX_SIZE_CM)
    depth: float | None = Field(default=None, gt=0, le=MAX_SIZE_CM)
    height: float | None = Field(default=None, gt=0, le=MAX_SIZE_CM)
    kind: str | None = Field(default=None, pattern="^(parts|identical)$")


@app.patch("/api/sessions/{sid}/objects/{oid}/size")
async def patch_size(sid: str, oid: str, body: SizePatch) -> dict:
    session = _session(sid)
    lead = _lead(session, _object(session, oid))
    if lead["status"] != "confirmed":
        raise HTTPException(409, "商品名が確定していません")
    size = lead.get("size") or {}
    if size.get("status") in ("queued", "running"):
        raise HTTPException(409, "サイズを取得中です")
    edits = body.model_dump(exclude_none=True, exclude={"kind"})
    if not size:
        size = {"status": "done", "source_kind": "manual", "source_url": "", "confidence": None, "note": "",
                "for_name": lead["confirmed_name"]}
    size.update(status="done" if size.get("status") == "done" or edits else size["status"], **edits)
    if edits:
        size["edited"] = True
        size["error"] = None
    if body.kind:
        size["kind"] = body.kind
    elif len(group_members(session, lead)) == 1:
        size["kind"] = "single"
    lead["size"] = size
    store.save(sid)
    return lead


class PhotoPatch(BaseModel):
    selected: bool


@app.patch("/api/sessions/{sid}/objects/{oid}/photos/{photo_id}")
async def patch_photo(sid: str, oid: str, photo_id: str, body: PhotoPatch) -> dict:
    session = _session(sid)
    lead = _lead(session, _object(session, oid))
    photo = next((p for p in (lead.get("size") or {}).get("photos") or [] if p["id"] == photo_id), None)
    if photo is None:
        raise HTTPException(404, "写真が見つかりません")
    photo["selected"] = body.selected
    store.save(sid)
    return lead


@app.get("/api/sessions/{sid}/objects/{oid}/photos/{photo_id}.jpg")
def get_photo(sid: str, oid: str, photo_id: str) -> FileResponse:
    session = _session(sid)
    lead = _lead(session, _object(session, oid))
    photo = next((p for p in (lead.get("size") or {}).get("photos") or [] if p["id"] == photo_id), None)
    if photo is None:
        raise HTTPException(404, "写真が見つかりません")
    return FileResponse(store.dir(sid) / "photos" / lead["id"] / f"{photo['id']}.jpg", media_type="image/jpeg")


@app.get("/api/glbsmith")
async def glbsmith_status() -> dict:
    try:
        status = await glbsmith.status()
        settings = await glbsmith.settings()
    except GlbsmithError as exc:
        return {"available": False, "url": glbsmith.base_url, "error": str(exc)}
    return {"available": True, "url": glbsmith.base_url, **status, "settings": settings}


@app.get("/api/glbsmith/agents/{agent}/models")
async def glbsmith_models(agent: str) -> dict:
    if not re.fullmatch(r"[\w-]{1,40}", agent):
        raise HTTPException(404, "エージェントが見つかりません")
    try:
        models = await glbsmith.models(agent)
        if agent == "claude":
            models["options"] = await asyncio.to_thread(label_claude_options, models.get("options") or [])
        return models
    except GlbsmithUnreachable as exc:
        raise HTTPException(503, str(exc)) from None
    except GlbsmithError as exc:
        raise HTTPException(400, str(exc)) from None


class BuildStart(BaseModel):
    agent: str = Field(pattern=r"^[\w-]{1,40}$")
    model: str | None = Field(default=None, max_length=80)
    object_ids: list[str] = Field(max_length=500)


@app.post("/api/sessions/{sid}/builds")
async def start_builds(sid: str, body: BuildStart) -> dict:
    session = _session(sid)
    leads = _confirmed_leads(session, body.object_ids)
    for lead in leads:
        size = lead.get("size") or {}
        if not all(size.get(k) for k in ("width", "depth", "height")):
            raise HTTPException(400, f"{lead['confirmed_name']} のサイズが入力されていません")
        if size.get("status") != "done":
            raise HTTPException(409, f"{lead['confirmed_name']} のサイズを取得中です")
        if is_active(lead):
            raise HTTPException(409, f"{lead['confirmed_name']} は作成中です")
    try:
        status = await glbsmith.status()
    except GlbsmithError as exc:
        raise HTTPException(503, str(exc)) from None
    if not status.get("blender"):
        raise HTTPException(400, "glbsmith から Blender が見つかりません")
    if not any(a["id"] == body.agent and a["available"] for a in status.get("agents", [])):
        raise HTTPException(400, f"glbsmith で使えないエージェントです: {body.agent}")
    builds.request(session, leads, body.agent, body.model)
    return session


@app.post("/api/sessions/{sid}/objects/{oid}/build/cancel")
async def cancel_build(sid: str, oid: str) -> dict:
    session = _session(sid)
    lead = _lead(session, _object(session, oid))
    if not is_active(lead):
        raise HTTPException(409, "作成中ではありません")
    try:
        await builds.cancel(session, lead)
    except GlbsmithError as exc:
        raise HTTPException(503, str(exc)) from None
    return lead


def _model_file(sid: str, oid: str, suffix: str) -> Path:
    obj = _object(_session(sid), oid)
    path = store.dir(sid) / "models" / f"{obj['id']}{suffix}"
    if not path.exists():
        raise HTTPException(404, "まだ作成されていません")
    return path


@app.get("/api/sessions/{sid}/objects/{oid}/model.glb")
def get_model(sid: str, oid: str) -> FileResponse:
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", _object(_session(sid), oid)["confirmed_name"] or oid)[:80]
    return FileResponse(_model_file(sid, oid, ".glb"), media_type="model/gltf-binary", filename=f"{name}.glb")


@app.get("/api/sessions/{sid}/objects/{oid}/render.png")
def get_render(sid: str, oid: str) -> FileResponse:
    return FileResponse(_model_file(sid, oid, ".png"), media_type="image/png")


@app.get("/api/sessions/{sid}/events")
async def session_events(sid: str, request: Request) -> StreamingResponse:
    session = _session(sid)

    async def stream():
        changed = store.subscribe(sid)
        try:
            while not await request.is_disconnected():
                changed.clear()
                yield f"event: session\ndata: {json.dumps(session, ensure_ascii=False)}\n\n"
                try:
                    await asyncio.wait_for(changed.wait(), timeout=15)
                except asyncio.TimeoutError:
                    pass  # resending the state doubles as the keep-alive
                await asyncio.sleep(SSE_MIN_INTERVAL)
        finally:
            store.unsubscribe(sid, changed)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
