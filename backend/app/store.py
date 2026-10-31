"""One session = one JSON file, written atomically; subscribers get change notifications."""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import uuid
from pathlib import Path

SESSION_ID = re.compile(r"[0-9a-f]{32}")


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, dict] = {}
        self._subscribers: dict[str, set[asyncio.Event]] = {}
        for path in sorted(root.glob("*/session.json")):
            if SESSION_ID.fullmatch(path.parent.name):
                self._sessions[path.parent.name] = json.loads(path.read_text())

    def new_id(self) -> str:
        return uuid.uuid4().hex

    def dir(self, sid: str) -> Path:
        if not SESSION_ID.fullmatch(sid):
            raise KeyError(sid)
        return self.root / sid

    def add(self, session: dict) -> None:
        self.dir(session["id"]).mkdir(parents=True, exist_ok=True)
        self._sessions[session["id"]] = session
        self.save(session["id"])

    def get(self, sid: str) -> dict:
        return self._sessions[sid]

    def all(self) -> list[dict]:
        return sorted(self._sessions.values(), key=lambda s: s["created_at"], reverse=True)

    def delete(self, sid: str) -> None:
        self._sessions.pop(sid)
        shutil.rmtree(self.dir(sid))

    def save(self, sid: str) -> None:
        session = self._sessions[sid]
        path = self.dir(sid) / "session.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(session, ensure_ascii=False, indent=1))
        os.replace(tmp, path)
        for event in self._subscribers.get(sid, ()):
            event.set()

    def subscribe(self, sid: str) -> asyncio.Event:
        event = asyncio.Event()
        self._subscribers.setdefault(sid, set()).add(event)
        return event

    def unsubscribe(self, sid: str, event: asyncio.Event) -> None:
        self._subscribers.get(sid, set()).discard(event)
