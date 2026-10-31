"""API and state transitions, with agents replaced by fakes."""
import asyncio
import importlib
import io
import os

import httpx
import pytest
from PIL import Image


class FakeAdapter:
    """Answers the detection prompts and search prompts without any CLI."""

    def __init__(self):
        self.searches = []
        self.fail_next = False
        self.gate: asyncio.Event | None = None
        self.running = 0
        self.max_running = 0
        self.same_product = []

    async def run(self, req, on_activity=None):
        if req.web:
            self.searches.append(req)
            self.running += 1
            self.max_running = max(self.max_running, self.running)
            try:
                if on_activity:
                    on_activity("検索: fake")
                await asyncio.sleep(0)
                if self.gate:
                    await self.gate.wait()
                if self.fail_next:
                    self.fail_next = False
                    raise RuntimeError("rate limited")
                return {
                    "brand": "Acme", "product_name": f"Thing{len(self.searches)}", "model_number": "",
                    "confidence": "high", "reasoning": "r", "source_urls": ["https://example.com"],
                }
            finally:
                self.running -= 1
        if "keep" in req.schema["properties"]:
            count = req.prompt.count("- id=")
            keep = [{"id": i, "label": f"物{i}", "category": "other"} for i in range(count)]
            return {"keep": keep, "add": [], "same_product": self.same_product}
        if "切り出したもの" in req.prompt:
            return {"objects": []}
        return {
            "objects": [
                {"label": f"物{i}", "category": "other", "x1": 50 + i * 200, "y1": 50, "x2": 200 + i * 200, "y2": 300}
                for i in range(4)
            ]
        }


@pytest.fixture
async def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app import detect, main, models_catalog

    main = importlib.reload(main)
    fake = FakeAdapter()
    monkeypatch.setitem(models_catalog.ADAPTERS, "claude", fake)
    monkeypatch.setitem(models_catalog.ADAPTERS, "codex", fake)
    main.manager.adapters = models_catalog.ADAPTERS
    monkeypatch.setattr(detect.sam, "tighten", lambda image, boxes: (boxes, 0))
    monkeypatch.setattr(main, "validate_choice", lambda p, m: None)
    main.manager.start()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        yield client, fake, main
    await main.manager.stop()


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1000, 500), "#888").save(buf, "JPEG")
    return buf.getvalue()


async def wait_until(client, sid, predicate, timeout=5.0):
    async def poll():
        while True:
            session = (await client.get(f"/api/sessions/{sid}")).json()
            if predicate(session):
                return session
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(poll(), timeout)


async def ready_session(client) -> dict:
    res = await client.post(
        "/api/sessions",
        files={"file": ("p.jpg", jpeg(), "image/jpeg")},
        data={"provider": "claude", "model": "opus"},
    )
    assert res.status_code == 201
    return await wait_until(client, res.json()["id"], lambda s: s["status"] != "detecting")


async def test_detection_creates_objects_and_crops(ctx):
    client, _, _ = ctx
    session = await ready_session(client)
    assert session["status"] == "ready" and len(session["objects"]) == 4
    crop = await client.get(f"/api/sessions/{session['id']}/objects/o1/crop")
    assert crop.status_code == 200 and crop.headers["content-type"] == "image/jpeg"


async def test_rejects_non_images_and_bad_ids(ctx):
    client, _, _ = ctx
    res = await client.post(
        "/api/sessions", files={"file": ("a.jpg", b"not an image", "image/jpeg")},
        data={"provider": "claude", "model": "opus"},
    )
    assert res.status_code == 400
    assert (await client.get("/api/sessions/..%2f..%2fetc")).status_code == 404
    assert (await client.get("/api/sessions/" + "0" * 32)).status_code == 404


async def test_foreign_origin_and_host_are_blocked(ctx):
    client, _, _ = ctx
    assert (await client.get("/api/agents", headers={"Origin": "https://evil.example"})).status_code == 403
    assert (await client.get("/api/agents", headers={"Host": "evil.example"})).status_code == 403


async def test_full_flow_with_verdicts(ctx):
    client, fake, _ = ctx
    session = await ready_session(client)
    sid = session["id"]
    res = await client.patch(f"/api/sessions/{sid}/objects/o4", json={"excluded": True})
    assert res.status_code == 200

    # Hints in the start request win over what autosave has stored.
    await client.patch(f"/api/sessions/{sid}/objects/o1", json={"hint": "old"})
    res = await client.post(f"/api/sessions/{sid}/search", json={"provider": "codex", "model": "m", "hints": {"o1": " Acme 曲面 "}})
    assert res.status_code == 200
    assert (await client.post(f"/api/sessions/{sid}/search", json={"provider": "codex", "model": "m"})).status_code == 409

    session = await wait_until(client, sid, lambda s: all(o["status"] == "done" for o in s["objects"][:3]))
    assert session["objects"][3]["status"] == "idle"
    assert len(fake.searches) == 3
    assert any("Acme 曲面" in r.prompt for r in fake.searches) and not any("old" in r.prompt for r in fake.searches)

    verdict = lambda oid, **body: client.post(f"/api/sessions/{sid}/objects/{oid}/verdict", json=body)
    assert (await verdict("o1", verdict="correct")).json()["status"] == "confirmed"
    assert (await verdict("o1", verdict="correct")).status_code == 409
    assert (await verdict("o1", verdict="reopen")).json()["status"] == "done"

    # wrong -> re-search with feedback: the rejected answer goes into the next prompt.
    rejected = (await verdict("o2", verdict="wrong")).json()
    rejected_name = rejected["result"]["product_name"]
    assert rejected["status"] == "rejected"
    assert (await client.post(f"/api/sessions/{sid}/objects/o2/retry", json={"feedback": ""})).status_code == 400
    assert (await client.post(f"/api/sessions/{sid}/objects/o2/retry", json={"feedback": "大型"})).status_code == 200
    session = await wait_until(client, sid, lambda s: s["objects"][1]["status"] == "done")
    assert "大型" in fake.searches[-1].prompt and rejected_name in fake.searches[-1].prompt
    assert len(session["objects"][1]["history"]) == 1

    # wrong -> enter the name directly.
    await verdict("o3", verdict="wrong")
    assert (await verdict("o3", verdict="manual", name=" ")).status_code == 400
    obj = (await verdict("o3", verdict="manual", name="Acme X100-B")).json()
    assert obj["status"] == "confirmed" and obj["confirmed_name"] == "Acme X100-B" and obj["confirmed_by"] == "user"
    assert (await verdict("o3", verdict="reopen")).json()["status"] == "rejected"


async def test_grouped_objects_are_one_search_and_one_verdict(ctx):
    client, fake, _ = ctx
    fake.same_product = [[0, 1]]
    session = await ready_session(client)
    sid = session["id"]
    pair = [o["id"] for o in session["objects"] if o["group"]]
    assert len(pair) == 2 and len({o["group"] for o in session["objects"] if o["group"]}) == 1
    objects = lambda s: {o["id"]: o for o in s["objects"]}

    # Excluding one part excludes the product.
    await client.patch(f"/api/sessions/{sid}/objects/{pair[0]}", json={"excluded": True})
    assert (await client.get(f"/api/sessions/{sid}")).json()["objects"][1]["excluded"] is True
    await client.patch(f"/api/sessions/{sid}/objects/{pair[1]}", json={"excluded": False})

    await client.post(f"/api/sessions/{sid}/search", json={"provider": "codex", "model": "m", "hints": {pair[0]: "B&O"}})
    session = await wait_until(client, sid, lambda s: all(o["status"] == "done" for o in s["objects"]))
    assert len(fake.searches) == 3
    paired_search = next(r for r in fake.searches if "B&O" in r.prompt)
    assert len(paired_search.images) == 3
    a, b = (objects(session)[oid] for oid in pair)
    assert a["result"] == b["result"]

    verdict = lambda oid, **body: client.post(f"/api/sessions/{sid}/objects/{oid}/verdict", json=body)
    await verdict(pair[1], verdict="correct")
    current = objects((await client.get(f"/api/sessions/{sid}")).json())
    assert current[pair[0]]["status"] == "confirmed"
    assert current[pair[0]]["confirmed_name"] == current[pair[1]]["confirmed_name"] is not None
    await verdict(pair[0], verdict="reopen")
    await verdict(pair[0], verdict="wrong")

    before = len(fake.searches)
    assert (await client.post(f"/api/sessions/{sid}/objects/{pair[1]}/retry", json={"feedback": "金色"})).status_code == 200
    session = await wait_until(client, sid, lambda s: all(objects(s)[oid]["status"] == "done" for oid in pair))
    assert len(fake.searches) == before + 1 and "金色" in fake.searches[-1].prompt
    assert all(len(objects(session)[oid]["history"]) == 1 for oid in pair)


async def test_group_can_be_split_before_search(ctx):
    client, fake, _ = ctx
    fake.same_product = [[0, 1]]
    session = await ready_session(client)
    sid = session["id"]
    oid = next(o["id"] for o in session["objects"] if o["group"])
    await client.patch(f"/api/sessions/{sid}/objects/{oid}", json={"ungroup": True})
    await client.post(f"/api/sessions/{sid}/search", json={"provider": "codex", "model": "m"})
    await wait_until(client, sid, lambda s: all(o["status"] == "done" for o in s["objects"]))
    assert len(fake.searches) == 4


async def test_manual_grouping(ctx):
    client, fake, _ = ctx
    fake.same_product = [[0, 1]]
    session = await ready_session(client)
    sid = session["id"]
    pair = [o["id"] for o in session["objects"] if o["group"]]
    single = [o["id"] for o in session["objects"] if not o["group"]]
    create = lambda **body: client.post(f"/api/sessions/{sid}/groups", json=body)

    assert (await create(object_ids=[single[0]])).status_code == 400
    assert (await create(object_ids=pair[:1])).status_code == 400  # already one product
    assert (await create(object_ids=["nope", single[0]])).status_code == 404

    # Picking one part of a group brings the whole group; drafts and excluded parts are merged in.
    await client.patch(f"/api/sessions/{sid}/objects/{pair[0]}", json={"hint": "デスク"})
    await client.patch(f"/api/sessions/{sid}/objects/{single[0]}", json={"excluded": True})
    res = await create(object_ids=[pair[1], single[0]], hints={single[0]: "昇降式"})
    assert res.status_code == 200
    objects = {o["id"]: o for o in res.json()["objects"]}
    trio = [o for o in res.json()["objects"] if o["group"]]
    assert {o["id"] for o in trio} == {*pair, single[0]} and len({o["group"] for o in trio}) == 1
    assert not objects[single[0]]["excluded"]
    assert trio[0]["hint"] == "デスク / 昇降式" and not any(o["hint"] for o in trio[1:])

    # Leaving keeps the rest grouped until only one part would remain.
    await client.patch(f"/api/sessions/{sid}/objects/{trio[2]['id']}", json={"leave_group": True})
    current = (await client.get(f"/api/sessions/{sid}")).json()["objects"]
    assert sum(bool(o["group"]) for o in current) == 2
    await create(object_ids=[trio[0]["id"], trio[2]["id"]])

    await client.post(f"/api/sessions/{sid}/search", json={"provider": "codex", "model": "m"})
    await wait_until(client, sid, lambda s: all(o["status"] == "done" for o in s["objects"]))
    assert len(fake.searches) == 2
    assert next(r for r in fake.searches if "デスク" in r.prompt).images.__len__() == 4
    # Still possible after the search: the merged product keeps one answer.
    assert (await create(object_ids=[single[1], trio[0]["id"]])).status_code == 200


async def test_redetect_replaces_objects(ctx):
    client, fake, _ = ctx
    session = await ready_session(client)
    sid = session["id"]
    fake.same_product = [[0, 1]]
    res = await client.post(f"/api/sessions/{sid}/redetect")
    assert res.status_code == 200
    session = await wait_until(client, sid, lambda s: s["status"] == "ready")
    assert sum(bool(o["group"]) for o in session["objects"]) == 2


async def test_failure_can_be_retried_and_concurrency_is_bounded(ctx):
    client, fake, main = ctx
    session = await ready_session(client)
    sid = session["id"]
    fake.gate = asyncio.Event()
    fake.fail_next = True
    await client.post(f"/api/sessions/{sid}/search", json={"provider": "codex", "model": "m"})
    await wait_until(client, sid, lambda s: sum(o["status"] == "running" for o in s["objects"]) == 3)
    current = (await client.get(f"/api/sessions/{sid}")).json()
    assert [o["status"] for o in current["objects"]].count("queued") == 1
    fake.gate.set()
    session = await wait_until(client, sid, lambda s: all(o["status"] in ("done", "failed") for o in s["objects"]))
    assert fake.max_running == 3
    failed = [o for o in session["objects"] if o["status"] == "failed"]
    assert len(failed) == 1 and "rate limited" in failed[0]["error"]
    oid = failed[0]["id"]
    assert (await client.post(f"/api/sessions/{sid}/objects/{oid}/retry", json={"feedback": ""})).status_code == 200
    await wait_until(client, sid, lambda s: next(o for o in s["objects"] if o["id"] == oid)["status"] == "done")


async def test_interrupted_searches_resume_on_restart(ctx):
    client, fake, main = ctx
    session = await ready_session(client)
    sid = session["id"]
    fake.gate = asyncio.Event()
    await client.post(f"/api/sessions/{sid}/search", json={"provider": "codex", "model": "m"})
    await wait_until(client, sid, lambda s: any(o["status"] == "running" for o in s["objects"]))
    await main.manager.stop()
    fake.gate.set()

    # A new manager over a store re-read from disk models the server restarting.
    from app.search import SearchManager
    from app.store import SessionStore

    store = SessionStore(main.DATA_DIR)
    assert any(o["status"] == "running" for o in store.get(sid)["objects"])
    manager = SearchManager(store, main.manager.adapters)
    manager.start()
    try:
        async def done():
            while not all(o["status"] == "done" for o in store.get(sid)["objects"]):
                await asyncio.sleep(0.01)
        await asyncio.wait_for(done(), 5)
    finally:
        await manager.stop()
    main.manager.start()  # the fixture stops it again
