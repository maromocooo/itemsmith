"""Provider and model choices, read from the local CLI configuration."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agents.base import AgentAdapter, subscription_env
from .agents.claude import ClaudeAdapter
from .agents.codex import CodexAdapter

CLAUDE_ALIASES = [("opus", "Opus"), ("sonnet", "Sonnet"), ("haiku", "Haiku")]
ADAPTERS: dict[str, AgentAdapter] = {"claude": ClaudeAdapter(), "codex": CodexAdapter()}
RESOLVE_TIMEOUT = 20
RESOLVED_TTL = 3600  # an alias moves to a new version when the CLI updates
UNRESOLVED_TTL = 60
_MODEL_NAME = r"[A-Za-z0-9][A-Za-z0-9._:\-\[\]]*"
_resolved: dict[str, tuple[str | None, float]] = {}
_resolve_lock = threading.Lock()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def resolve_claude_model(model: str) -> str | None:
    """Ask the CLI what an alias means. A local slash command reports the model without using tokens."""
    message = {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "/cost"}]}}
    with tempfile.TemporaryDirectory(prefix="itemsmith-model-") as workdir:
        try:
            done = subprocess.run(
                [
                    "claude", "-p", "--model", model,
                    "--output-format", "stream-json", "--input-format", "stream-json", "--verbose",
                    "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
                ],
                input=json.dumps(message) + "\n", capture_output=True, text=True,
                env=subscription_env(), cwd=workdir, timeout=RESOLVE_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError):
            return None
    for line in done.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "system" and event.get("subtype") == "init":
            return event.get("model") or None
    return None


def refresh_claude_versions(models: list[str]) -> None:
    """Blocking; keeps catalog() itself free of subprocesses."""
    if shutil.which("claude") is None:
        return
    with _resolve_lock:
        now = time.time()
        stale = [
            m for m in dict.fromkeys(models)
            if re.fullmatch(_MODEL_NAME, m) and (m not in _resolved or _resolved[m][1] <= now)
        ]
        if not stale:
            return
        with ThreadPoolExecutor(max_workers=4) as pool:
            for model, resolved in zip(stale, pool.map(resolve_claude_model, stale)):
                _resolved[model] = (resolved, now + (RESOLVED_TTL if resolved else UNRESOLVED_TTL))


def version_label(resolved: str) -> str:
    """claude-opus-5-5 -> Opus 5.5; anything unfamiliar is shown as it is."""
    match = re.fullmatch(r"claude-([a-z]+)-(\d+(?:-\d{1,2})*)(?:-\d{8})?", resolved)
    if not match:
        return resolved
    return f"{match[1].capitalize()} {match[2].replace('-', '.')}"


def claude_label(model: str, fallback: str) -> str:
    resolved = _resolved.get(model, (None, 0))[0]
    if not resolved:
        return fallback.split(" · ")[0]  # the CLI's cached labels can carry a sales line
    return version_label(resolved) + ("(1M)" if model.endswith("[1m]") else "")


def label_claude_options(options: list[dict]) -> list[dict]:
    """For Claude model lists that come from elsewhere (glbsmith)."""
    refresh_claude_versions([o["id"] for o in options])
    return [{**o, "label": claude_label(o["id"], o.get("label") or o["id"])} for o in options]


def claude_catalog(home: Path) -> dict:
    # Claude Code has no command that lists models: built-in aliases plus what it cached locally.
    models = [{"id": value, "label": claude_label(value, label)} for value, label in CLAUDE_ALIASES]
    cached = _read_json(home / ".claude.json").get("additionalModelOptionsCache") or []
    for option in cached:
        if isinstance(option, dict) and option.get("value"):
            models.append({"id": option["value"], "label": claude_label(option["value"], option.get("label") or option["value"])})
    configured = _read_json(home / ".claude" / "settings.json").get("model")
    if configured and all(m["id"] != configured for m in models):
        models.append({"id": configured, "label": claude_label(configured, configured)})
    return {
        "id": "claude",
        "label": "Claude Code",
        "available": shutil.which("claude") is not None,
        "default_model": configured or "opus",
        "models": models,
    }


def codex_catalog(home: Path) -> dict:
    cache = _read_json(home / ".codex" / "models_cache.json").get("models") or []
    listed = [m for m in cache if isinstance(m, dict) and m.get("visibility") == "list" and m.get("slug")]
    listed.sort(key=lambda m: m.get("priority", 1 << 30))
    models = [{"id": m["slug"], "label": m.get("display_name") or m["slug"]} for m in listed]
    configured = None
    try:
        configured = tomllib.loads((home / ".codex" / "config.toml").read_text()).get("model")
    except (OSError, tomllib.TOMLDecodeError):
        pass
    if configured and all(m["id"] != configured for m in models):
        models.insert(0, {"id": configured, "label": configured})
    return {
        "id": "codex",
        "label": "Codex",
        "available": shutil.which("codex") is not None,
        "default_model": configured or (models[0]["id"] if models else None),
        "models": models,
    }


def catalog(home: Path | None = None) -> list[dict]:
    home = home or Path.home()
    return [claude_catalog(home), codex_catalog(home)]


def catalog_with_versions() -> list[dict]:
    refresh_claude_versions([m["id"] for m in claude_catalog(Path.home())["models"]])
    return catalog()


def validate_choice(provider: str, model: str, home: Path | None = None) -> str | None:
    """Return an error message, or None when the pair can be passed to the CLI."""
    if provider not in ADAPTERS:
        return f"不明なプロバイダです: {provider}"
    entry = next(p for p in catalog(home) if p["id"] == provider)
    if not entry["available"]:
        return f"{entry['label']} の CLI が見つかりません"
    if not re.fullmatch(_MODEL_NAME, model or ""):
        return "モデル名が不正です"
    if all(m["id"] != model for m in entry["models"]):
        return f"{entry['label']} のモデル一覧にありません: {model}"
    return None
