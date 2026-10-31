"""Size lookup and GLB builds, with agents and the glbsmith server replaced by fakes."""
import asyncio
import importlib
import io

import httpx
import numpy as np
import pytest
from PIL import Image

from app.builds import box_mask, build_photos, warnings_from
from app.glbsmith import GlbsmithError, GlbsmithUnreachable
from app import product_photos as pp
from app.sizes import sanitize_size


class FakeAgent:
    def __init__(self):
        self.size_requests = []
        self.check_requests = []
        self.image_urls = []
        self.check_answer = []
        self.search_count = 0
        self.same_product = []
        self.names = iter(["Desk", "Legs", "Thing", "Other"])

    async def run(self, req, on_activity=None):
        if req.web and "width_cm" in req.schema["properties"]:
            self.size_requests.append(req)
            await asyncio.sleep(0)
            return {
                "width_cm": 120.0, "depth_cm": 60.0, "height_cm": 72.5, "source_url": "https://example.com/spec",
                "source_kind": "official", "group_kind": "parts", "confidence": "high", "note": "公式仕様",
                "image_urls": self.image_urls,
            }
        if "product_only" in str(req.schema):
            self.check_requests.append(req)
            return {"photos": self.check_answer}
        if req.web:
            self.search_count += 1
            return {
                "brand": "Acme", "product_name": next(self.names), "model_number": "",
                "confidence": "low" if self.search_count == 1 else "high", "reasoning": "r", "source_urls": ["https://example.com"],
            }
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


class FakeGlbsmith:
    base_url = "http://glbsmith.test"

    def __init__(self):
        self.submitted = []
        self.jobs = {}
        self.down = False
        self.cancelled = []

    def _check(self):
        if self.down:
            raise GlbsmithUnreachable("glbsmith に接続できません")

    async def status(self):
        self._check()
        return {"agents": [{"id": "claude", "label": "Claude Code", "available": True, "billing": "subscription"}], "blender": True, "sandbox": True}

    async def settings(self):
        self._check()
        return {"concurrency": 1, "max": 8, "running": 0, "queued": 0}

    async def models(self, agent):
        self._check()
        return {"default": None, "options": [{"id": "opus", "label": "Opus"}]}

    async def submit(self, photos, **fields):
        self._check()
        if fields["agent"] == "refuse":
            raise GlbsmithError("Agent not available")
        job_id = f"job{len(self.submitted)}"
        self.submitted.append({"photos": photos, **fields})
        self.jobs[job_id] = {"id": job_id, "status": "queued"}
        return self.jobs[job_id]

    async def job(self, job_id):
        self._check()
        return self.jobs.get(job_id)

    async def cancel(self, job_id):
        self._check()
        self.cancelled.append(job_id)
        self.jobs[job_id]["status"] = "cancelled"

    async def model(self, job_id):
        self._check()
        return b"glTF-" + job_id.encode()

    async def latest_render(self, job_id):
        self._check()
        return b"\x89PNG"


@pytest.fixture
async def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GLBSMITH_POLL_SECONDS", "0.01")
    from app import builds, detect, main, models_catalog, sam

    importlib.reload(builds)
    main = importlib.reload(main)
    agent = FakeAgent()
    glb = FakeGlbsmith()
    monkeypatch.setitem(models_catalog.ADAPTERS, "claude", agent)
    monkeypatch.setitem(models_catalog.ADAPTERS, "codex", agent)
    monkeypatch.setattr(detect.sam, "tighten", lambda image, boxes: (boxes, 0))
    monkeypatch.setattr(sam, "masks", lambda image, boxes: [None for _ in boxes])
    monkeypatch.setattr(main, "validate_choice", lambda p, m: None)
    main.glbsmith = glb
    main.builds.client = glb
    main.manager.adapters = main.sizes.adapters = models_catalog.ADAPTERS
    main.manager.start()
    main.sizes.start()
    main.builds.start()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        yield client, agent, glb, main
    await main.manager.stop()
    await main.sizes.stop()
    await main.builds.stop()


async def wait_until(client, sid, predicate, timeout=5.0):
    async def poll():
        while True:
            session = (await client.get(f"/api/sessions/{sid}")).json()
            if predicate(session):
                return session
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(poll(), timeout)


async def searched_session(client) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (1000, 500), "#888").save(buf, "JPEG")
    res = await client.post(
        "/api/sessions", files={"file": ("p.jpg", buf.getvalue(), "image/jpeg")},
        data={"provider": "claude", "model": "opus"},
    )
    sid = res.json()["id"]
    await wait_until(client, sid, lambda s: s["status"] == "ready")
    await client.post(f"/api/sessions/{sid}/search", json={"provider": "claude", "model": "opus"})
    await wait_until(client, sid, lambda s: all(o["status"] == "done" for o in s["objects"]))
    return sid


async def confirm(client, sid, oid):
    res = await client.post(f"/api/sessions/{sid}/objects/{oid}/verdict", json={"verdict": "correct"})
    assert res.status_code == 200


def by_id(session, oid):
    return next(o for o in session["objects"] if o["id"] == oid)


async def test_size_lookup_then_edit(ctx):
    client, agent, _, _ = ctx
    sid = await searched_session(client)
    start = lambda **body: client.post(f"/api/sessions/{sid}/sizes", json={"provider": "claude", "model": "opus", **body})

    assert (await start(object_ids=["o1"])).status_code == 409  # not confirmed yet
    assert (await start(object_ids=[])).status_code == 400
    await confirm(client, sid, "o1")
    assert (await start(object_ids=["o1"])).status_code == 200
    session = await wait_until(client, sid, lambda s: (by_id(s, "o1").get("size") or {}).get("status") == "done")
    size = by_id(session, "o1")["size"]
    assert (size["width"], size["depth"], size["height"]) == (120.0, 60.0, 72.5)
    assert size["kind"] == "single" and size["source_kind"] == "official"
    assert by_id(session, "o1")["confirmed_name"] in agent.size_requests[0].prompt

    patch = lambda oid, **body: client.patch(f"/api/sessions/{sid}/objects/{oid}/size", json=body)
    assert (await patch("o1", width=0)).status_code == 422
    assert (await patch("o2", width=10)).status_code == 409  # not confirmed
    res = (await patch("o1", width=100.5)).json()
    assert res["size"]["width"] == 100.5 and res["size"]["edited"] is True

    # Reconfirming under another name drops the size looked up for the old one.
    await client.post(f"/api/sessions/{sid}/objects/o1/verdict", json={"verdict": "reopen"})
    await client.post(f"/api/sessions/{sid}/objects/o1/verdict", json={"verdict": "wrong"})
    await client.post(f"/api/sessions/{sid}/objects/o1/verdict", json={"verdict": "manual", "name": "New Name"})
    assert "size" not in by_id((await client.get(f"/api/sessions/{sid}")).json(), "o1")


async def test_manual_size_without_lookup_and_builds(ctx):
    client, _, glb, _ = ctx
    sid = await searched_session(client)
    await confirm(client, sid, "o1")
    build = lambda ids, agent="claude": client.post(f"/api/sessions/{sid}/builds", json={"agent": agent, "object_ids": ids})

    assert (await build(["o1"])).status_code == 400  # no size yet
    await client.patch(f"/api/sessions/{sid}/objects/o1/size", json={"width": 30, "depth": 20, "height": 10})
    assert (await build(["o1"], "nope")).status_code == 400

    glb.down = True
    assert (await build(["o1"])).status_code == 503
    assert (await client.get("/api/glbsmith")).json()["available"] is False
    glb.down = False

    assert (await build(["o1"])).status_code == 200
    session = await wait_until(client, sid, lambda s: by_id(s, "o1")["build"]["job_id"])
    assert (await build(["o1"])).status_code == 409  # already running
    sent = glb.submitted[0]
    assert (sent["width"], sent["depth"], sent["height"]) == (30, 20, 10)
    assert sent["name"] == by_id(session, "o1")["confirmed_name"]
    assert len(sent["photos"]) == 2  # crop first, then the whole photo
    assert (await client.delete(f"/api/sessions/{sid}")).status_code == 409

    # Survives glbsmith being away, then collects the result.
    glb.down = True
    await asyncio.sleep(0.05)
    glb.down = False
    glb.jobs["job0"].update(
        status="done", seconds=12, tokens={"total": 1000}, message="ok",
        report={"smaller_than_box": {"depth_y": {"box_cm": 20, "model_cm": 2}}, "photo_check": {"verdict": "fail"}},
    )
    session = await wait_until(client, sid, lambda s: by_id(s, "o1")["build"]["status"] == "done")
    built = by_id(session, "o1")["build"]
    assert built["has_model"] and built["has_render"] and built["tokens"]["total"] == 1000 and len(built["warnings"]) == 2
    model = await client.get(f"/api/sessions/{sid}/objects/o1/model.glb")
    assert model.status_code == 200 and model.content == b"glTF-job0"
    assert (await client.get(f"/api/sessions/{sid}/objects/o1/render.png")).status_code == 200
    assert (await client.get(f"/api/sessions/{sid}/objects/o2/model.glb")).status_code == 404


async def test_build_failure_cancel_and_prepare_refused(ctx):
    client, _, glb, _ = ctx
    sid = await searched_session(client)
    for oid in ("o1", "o2"):
        await confirm(client, sid, oid)
        await client.patch(f"/api/sessions/{sid}/objects/{oid}/size", json={"width": 30, "depth": 20, "height": 10})
    await client.post(f"/api/sessions/{sid}/builds", json={"agent": "claude", "object_ids": ["o1", "o2"]})
    await wait_until(client, sid, lambda s: by_id(s, "o2")["build"]["job_id"])

    assert (await client.post(f"/api/sessions/{sid}/objects/o3/build/cancel")).status_code == 409
    assert (await client.post(f"/api/sessions/{sid}/objects/o1/build/cancel")).status_code == 200
    assert glb.cancelled == ["job0"]
    glb.jobs["job1"].update(status="failed", error="no model")
    session = await wait_until(client, sid, lambda s: by_id(s, "o2")["build"]["status"] == "failed")
    assert by_id(session, "o2")["build"]["error"] == "no model" and by_id(session, "o1")["build"]["status"] == "cancelled"
    # A failed one can go again.
    assert (await client.post(f"/api/sessions/{sid}/builds", json={"agent": "claude", "object_ids": ["o2"]})).status_code == 200


async def test_merge_after_search_keeps_the_best_answer(ctx):
    client, _, _, _ = ctx
    sid = await searched_session(client)
    session = (await client.get(f"/api/sessions/{sid}")).json()
    low = next(o for o in session["objects"] if o["result"]["confidence"] == "low")
    high = next(o for o in session["objects"] if o["result"]["confidence"] == "high")
    res = await client.post(f"/api/sessions/{sid}/groups", json={"object_ids": [low["id"], high["id"]]})
    assert res.status_code == 200
    merged = [o for o in res.json()["objects"] if o["group"]]
    assert {o["result"]["product_name"] for o in merged} == {high["result"]["product_name"]}
    assert {o["status"] for o in merged} == {"done"}

    # Judged as one, and a confirmed part beats a more confident answer.
    third = next(o for o in session["objects"] if o["id"] not in (low["id"], high["id"]))
    await confirm(client, sid, third["id"])
    res = await client.post(f"/api/sessions/{sid}/groups", json={"object_ids": [merged[0]["id"], third["id"]]})
    trio = [o for o in res.json()["objects"] if o["group"]]
    assert {o["status"] for o in trio} == {"confirmed"} and {o["confirmed_name"] for o in trio} == {f"Acme {third['result']['product_name']}"}

    # A product with a size loses it when split, because it is no longer the same product.
    lead = trio[0]["id"]
    await client.patch(f"/api/sessions/{sid}/objects/{lead}/size", json={"width": 1, "depth": 1, "height": 1})
    await client.patch(f"/api/sessions/{sid}/objects/{lead}", json={"ungroup": True})
    current = (await client.get(f"/api/sessions/{sid}")).json()
    assert "size" not in by_id(current, lead) and not by_id(current, lead)["group"]
    assert by_id(current, lead)["status"] == "confirmed"


async def test_built_product_cannot_be_regrouped(ctx):
    client, _, _, _ = ctx
    sid = await searched_session(client)
    await confirm(client, sid, "o1")
    await client.patch(f"/api/sessions/{sid}/objects/o1/size", json={"width": 30, "depth": 20, "height": 10})
    await client.post(f"/api/sessions/{sid}/builds", json={"agent": "claude", "object_ids": ["o1"]})
    res = await client.post(f"/api/sessions/{sid}/groups", json={"object_ids": ["o1", "o2"]})
    assert res.status_code == 409


async def test_groups_use_parts_or_one_unit(ctx):
    client, agent, glb, _ = ctx
    agent.same_product = [[0, 1]]
    sid = await searched_session(client)
    session = (await client.get(f"/api/sessions/{sid}")).json()
    pair = [o for o in session["objects"] if o["group"]]
    lead = pair[0]["id"]
    await confirm(client, sid, lead)
    await client.post(f"/api/sessions/{sid}/sizes", json={"provider": "claude", "model": "opus", "object_ids": [pair[1]["id"]]})
    session = await wait_until(client, sid, lambda s: (by_id(s, lead).get("size") or {}).get("status") == "done")
    assert by_id(session, lead)["size"]["kind"] == "parts" and len(agent.size_requests[0].images) == 3

    res = await client.patch(f"/api/sessions/{sid}/objects/{lead}/size", json={"kind": "identical"})
    assert res.json()["size"]["kind"] == "identical" and not res.json()["size"].get("edited")
    await client.post(f"/api/sessions/{sid}/builds", json={"agent": "claude", "object_ids": [lead]})
    session = await wait_until(client, sid, lambda s: by_id(s, lead)["build"]["job_id"])
    assert by_id(session, lead)["build"]["kind"] == "identical"


def mask_of(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)))


def test_build_photos_crop_small_and_keep_large_whole():
    image = Image.new("RGB", (2000, 1000), "#555")
    small = build_photos(image, box_mask([1000, 400, 100, 100], image.size))
    assert len(small) == 2
    crop = Image.open(io.BytesIO(small[0][0]))
    assert max(crop.size) == 1024  # enlarged, so a small item survives glbsmith's handling
    assert mask_of(small[0][1]).shape == crop.size[::-1]
    assert Image.open(io.BytesIO(small[1][0])).size == (2000, 1000)

    large = build_photos(image, box_mask([50, 50, 1900, 900], image.size))
    assert len(large) == 1 and Image.open(io.BytesIO(large[0][0])).size == (2000, 1000)


def test_sanitize_size():
    raw = {"width_cm": 10, "depth_cm": 5.55, "height_cm": 2, "source_url": "javascript:alert(1)",
           "source_kind": "?", "group_kind": "identical", "confidence": "?", "note": "n"}
    size = sanitize_size(raw, multiple=True)
    assert size["depth"] == round(5.55, 1)
    assert size["source_url"] == "" and size["source_kind"] == "estimate" and size["kind"] == "identical"
    assert sanitize_size(raw, multiple=False)["kind"] == "single"
    assert size["source_quotes"] == []
    quoted = {**raw, "source_url": "https://a.example/spec", "source_quotes": [" 99.3  cm ", "99.3 cm", "", "b", "c", "d"]}
    assert sanitize_size(quoted, multiple=False)["source_quotes"] == ["99.3 cm", "b", "c"]
    assert sanitize_size({**quoted, "source_url": ""}, multiple=False)["source_quotes"] == []
    for bad in (0, -1, 5000, "10", True, None):
        with pytest.raises(ValueError):
            sanitize_size({**raw, "width_cm": bad}, multiple=False)


def test_warnings_are_readable():
    assert warnings_from({"report": {}}) == []
    assert warnings_from({}) == []
    assert "幅" in warnings_from({"report": {"smaller_than_box": {"width_x": {"box_cm": 9, "model_cm": 1}}}})[0]


def jpeg(colour="#123456", size=(800, 600)) -> bytes:
    buf = io.BytesIO()
    image = Image.new("RGB", size, colour)
    image.paste("#ffffff", (0, 0, size[0] // 2, size[1] // 2))
    image.save(buf, "JPEG")
    return buf.getvalue()


def stripes() -> bytes:
    buf = io.BytesIO()
    image = Image.new("RGB", (900, 400), "#000000")
    for x in range(0, 900, 100):
        image.paste("#ffffff", (x, 0, x + 50, 400))
    image.save(buf, "JPEG")
    return buf.getvalue()


def gif() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (800, 600)).save(buf, "GIF")
    return buf.getvalue()


@pytest.fixture
def public_dns(monkeypatch):
    monkeypatch.setattr(pp, "_resolve", lambda host: ["10.0.0.5"] if "internal" in host else ["93.184.216.34"])


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/a.jpg", "http://user:pw@example.com/a.jpg", "http://internal.test/a.jpg",
])
async def test_unsafe_urls_are_refused(url, public_dns):
    with pytest.raises(pp.UnsafeUrl):
        await pp.check_url(url)


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "192.168.1.1", "169.254.169.254", "::1", "fe80::1%eth0", "fd00::1"])
async def test_non_public_addresses_are_refused(address, monkeypatch):
    monkeypatch.setattr(pp, "_resolve", lambda host: ["93.184.216.34", address])
    with pytest.raises(pp.UnsafeUrl):
        await pp.check_url("https://example.com/a.jpg")


async def test_redirects_are_checked_and_limited(public_dns):
    def handler(request):
        if request.url.path == "/to-internal":
            return httpx.Response(302, headers={"location": "http://internal.test/secret.jpg"})
        if request.url.path == "/loop":
            return httpx.Response(302, headers={"location": "/loop"})
        if request.url.path == "/moved":
            return httpx.Response(301, headers={"location": "/ok.jpg"})
        assert request.url.host != "internal.test"
        return httpx.Response(200, headers={"content-type": "image/jpeg"}, content=jpeg())

    transport = httpx.MockTransport(handler)
    assert await pp.fetch_image("https://example.com/moved", transport)
    with pytest.raises(pp.UnsafeUrl):
        await pp.fetch_image("https://example.com/to-internal", transport)
    with pytest.raises(ValueError):
        await pp.fetch_image("https://example.com/loop", transport)


async def test_download_filters_and_dedupes(tmp_path, public_dns, monkeypatch):
    monkeypatch.setattr(pp, "MAX_DOWNLOAD", 500_000)

    def handler(request):
        path = request.url.path
        kinds = {"/html": ("text/html", b"<html>"), "/huge": ("image/jpeg", b"x" * 600_000), "/gif": ("image/gif", gif()),
                 "/small": ("image/jpeg", jpeg(size=(100, 100))), "/broken": ("image/jpeg", b"not an image"),
                 "/dup.jpg": ("image/jpeg", jpeg(size=(1000, 750))), "/one.jpg": ("image/jpeg", jpeg()),
                 "/two.jpg": ("image/jpeg", stripes())}
        if path == "/404":
            return httpx.Response(404)
        kind, body = kinds[path]
        return httpx.Response(200, headers={"content-type": kind}, content=body)

    urls = ["html", "huge", "gif", "small", "broken", "404", "one.jpg", "dup.jpg", "two.jpg"]
    found = await pp.download_all([{"url": f"https://example.com/{u}", "page_url": ""} for u in urls], tmp_path / "p", httpx.MockTransport(handler))
    assert [p["url"].rsplit("/", 1)[1] for p in found] == ["one.jpg", "two.jpg"]
    assert sorted(f.name for f in (tmp_path / "p").iterdir()) == ["00.jpg", "01.jpg"]


def test_candidates_are_cleaned_and_capped():
    raw = {"image_urls": [{"url": f"https://e.com/{i}.jpg", "page_url": "javascript:x"} for i in range(30)]
           + [{"url": "https://e.com/0.jpg", "page_url": ""}, {"url": "ftp://e.com/x.jpg", "page_url": ""}, {"page_url": ""}]}
    found = pp.candidates_from(raw)
    assert len(found) == pp.MAX_CANDIDATES and found[0]["page_url"] == ""
    assert pp.candidates_from({}) == []


def test_check_keeps_same_product_and_prefers_variety():
    photos = [{"id": f"{i:02d}"} for i in range(9)]
    answer = {"photos": [
        {"index": 0, "same_product": True, "view": "front", "product_only": True},
        {"index": 1, "same_product": True, "view": "front", "product_only": True},
        {"index": 2, "same_product": False, "view": "back", "product_only": True},
        {"index": 3, "same_product": True, "view": "back", "product_only": True},
        {"index": 4, "same_product": True, "view": "package", "product_only": True},
        {"index": 5, "same_product": True, "view": "left", "product_only": True},
        {"index": 6, "same_product": True, "view": "angled", "product_only": True},
        {"index": 7, "same_product": True, "view": "top", "product_only": True},
        {"index": 8, "same_product": True, "view": "in_use", "product_only": False},
        {"index": 99, "same_product": True, "view": "front", "product_only": True},
    ]}
    kept = pp.apply_check(photos, answer)
    assert [p["id"] for p in kept] == ["00", "01", "03", "05", "06", "07", "08"]
    selected = {p["id"] for p in kept if p["selected"]}
    assert len(selected) == pp.MAX_SELECTED
    # Every distinct view of the bare product makes it in, before a second front.
    assert {"00", "03", "05", "06", "07"} <= selected


async def test_photos_reach_glbsmith_after_the_real_item(ctx, monkeypatch):
    client, agent, glb, _ = ctx
    sid = await searched_session(client)
    await confirm(client, sid, "o1")
    agent.image_urls = [{"url": f"https://example.com/{i}.jpg", "page_url": "https://example.com/p"} for i in range(4)]
    agent.check_answer = [
        {"index": 0, "same_product": True, "view": "front", "product_only": True},
        {"index": 1, "same_product": False, "view": "back", "product_only": True},
        {"index": 2, "same_product": True, "view": "back", "product_only": True},
        {"index": 3, "same_product": True, "view": "left", "product_only": False},
    ]
    colours = iter(["#111111", "#ff0000", "#00ff00", "#0000ff"])

    async def fake_fetch(url, transport=None):
        buf = io.BytesIO()
        image = Image.new("RGB", (600, 400), next(colours))
        image.paste("#ffffff", (0, 0, 100 + 100 * int(url[-5]), 300))
        image.save(buf, "JPEG")
        return buf.getvalue()

    monkeypatch.setattr(pp, "fetch_image", fake_fetch)
    await client.post(f"/api/sessions/{sid}/sizes", json={"provider": "claude", "model": "opus", "object_ids": ["o1"]})
    session = await wait_until(client, sid, lambda s: (by_id(s, "o1").get("size") or {}).get("status") == "done")
    photos = by_id(session, "o1")["size"]["photos"]
    assert len(photos) == 3 and all(p["selected"] for p in photos) and {p["view"] for p in photos} == {"front", "back", "left"}
    assert len(agent.check_requests[0].images) == 5 and not agent.check_requests[0].web
    served = await client.get(f"/api/sessions/{sid}/objects/o1/photos/{photos[0]['id']}.jpg")
    assert served.status_code == 200 and served.headers["content-type"] == "image/jpeg"
    assert (await client.get(f"/api/sessions/{sid}/objects/o1/photos/..%2f..%2fx.jpg")).status_code == 404

    rejected = photos[2]["id"]
    await client.patch(f"/api/sessions/{sid}/objects/o1/photos/{rejected}", json={"selected": False})
    assert (await client.patch(f"/api/sessions/{sid}/objects/o1/photos/99", json={"selected": True})).status_code == 404
    await client.post(f"/api/sessions/{sid}/builds", json={"agent": "claude", "object_ids": ["o1"]})
    session = await wait_until(client, sid, lambda s: by_id(s, "o1")["build"]["job_id"])
    sent = glb.submitted[0]["photos"]
    assert len(sent) == 4 and sent[0][1] and sent[1][1]  # crop and whole photo carry masks
    assert sent[2][1] == b"" and sent[3][1] == b""  # product photos have no mask
    assert by_id(session, "o1")["build"]["product_photos"] == 2


async def test_size_survives_photo_problems(ctx, monkeypatch):
    client, agent, _, _ = ctx
    sid = await searched_session(client)
    await confirm(client, sid, "o1")
    agent.image_urls = [{"url": "https://example.com/0.jpg", "page_url": ""}]

    async def blocked(url, transport=None):
        raise ValueError("403")

    monkeypatch.setattr(pp, "fetch_image", blocked)
    await client.post(f"/api/sessions/{sid}/sizes", json={"provider": "claude", "model": "opus", "object_ids": ["o1"]})
    session = await wait_until(client, sid, lambda s: (by_id(s, "o1").get("size") or {}).get("status") == "done")
    size = by_id(session, "o1")["size"]
    assert size["width"] == 120.0 and size["photos"] == [] and "ダウンロードできません" in size["photo_note"]


async def test_latest_render_skips_gaps():
    from app.glbsmith import GlbsmithClient

    def handler(request):
        number = int(request.url.path.rsplit("/", 1)[1][:3])
        if number in (1, 2, 5):
            return httpx.Response(200, content=b"render%d" % number)
        return httpx.Response(404, json={"detail": "No such render."})

    client = GlbsmithClient("http://glbsmith.test", httpx.MockTransport(handler))
    assert await client.latest_render("job") == b"render5"
