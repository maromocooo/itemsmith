"""Client for a glbsmith server, which turns a photo and a size into a GLB."""
from __future__ import annotations

import os

import httpx

GLBSMITH_URL = os.environ.get("GLBSMITH_URL", "http://127.0.0.1:8021").rstrip("/")
MAX_RENDERS = 200
MAX_RENDER_GAP = 10


class GlbsmithError(RuntimeError):
    """glbsmith refused a request; the message is meant for the person."""


class GlbsmithUnreachable(GlbsmithError):
    """The server is not running. Unlike other errors, this may pass by itself."""


class GlbsmithClient:
    def __init__(self, base_url: str = GLBSMITH_URL, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = base_url
        self._transport = transport

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=120, transport=self._transport) as client:
                response = await client.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            raise GlbsmithUnreachable(f"glbsmith に接続できません({self.base_url})") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = None
            raise GlbsmithError(str(detail or f"glbsmith がエラーを返しました ({response.status_code})"))
        return response

    async def status(self) -> dict:
        return (await self._request("GET", "/api/status")).json()

    async def settings(self) -> dict:
        return (await self._request("GET", "/api/settings")).json()

    async def models(self, agent: str) -> dict:
        return (await self._request("GET", f"/api/agents/{agent}/models")).json()

    async def submit(
        self, photos: list[tuple[bytes, bytes]], *, width: float, depth: float, height: float,
        agent: str, model: str | None, name: str,
    ) -> dict:
        """photos: (JPEG, PNG mask or b"" for none) pairs, main photo first."""
        files = []
        for index, (photo, mask) in enumerate(photos):
            files.append(("images", (f"photo{index}.jpg", photo, "image/jpeg")))
            files.append(("masks", (f"mask{index}.png", mask, "image/png")))
        data = {"width": width, "depth": depth, "height": height, "agent": agent, "name": name[:120]}
        if model:
            data["model"] = model
        return (await self._request("POST", "/api/jobs", data=data, files=files)).json()

    async def job(self, job_id: str) -> dict | None:
        try:
            return (await self._request("GET", f"/api/jobs/{job_id}")).json()
        except GlbsmithUnreachable:
            raise
        except GlbsmithError:
            return None

    async def cancel(self, job_id: str) -> None:
        await self._request("POST", f"/api/jobs/{job_id}/cancel")

    async def model(self, job_id: str) -> bytes:
        return (await self._request("GET", f"/api/jobs/{job_id}/model.glb")).content

    async def latest_render(self, job_id: str) -> bytes | None:
        """The last angled view the agent rendered. A round may skip that view, so gaps are expected."""
        latest = None
        misses = 0
        for number in range(1, MAX_RENDERS):
            try:
                latest = (await self._request("GET", f"/api/jobs/{job_id}/renders/{number:03d}_angled.png")).content
                misses = 0
            except GlbsmithUnreachable:
                raise
            except GlbsmithError:
                misses += 1
                if misses >= MAX_RENDER_GAP:
                    break
        return latest
