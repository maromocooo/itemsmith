from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .base import AgentAdapter, AgentRequest, ParsedEvent, parse_json_text

# The read-only sandbox alone still leaves shell reads and ChatGPT connector tools,
# which web content could steer. Only web search and passed-in images are needed.
DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "apps", "plugins", "goals", "image_generation",
    "tool_suggest", "skill_mcp_dependency_install", "view_image", "hooks",
)


class CodexAdapter(AgentAdapter):
    provider = "codex"

    def build_command(self, req: AgentRequest, workdir: Path) -> list[str]:
        schema_path = workdir / "output_schema.json"
        schema_path.write_text(json.dumps(req.schema))
        cmd = ["codex"]
        if req.web:
            cmd.append("--search")
        cmd += [
            "exec", "--json",
            "-s", "read-only",
            "--ignore-user-config", "--ignore-rules",
            "--ephemeral", "--skip-git-repo-check",
            "-m", req.model,
            "--output-schema", str(schema_path),
        ]
        for feature in DISABLED_FEATURES:
            cmd += ["--disable", feature]
        for path in req.images:
            cmd += ["-i", str(path)]
        cmd += ["--", req.prompt]
        return cmd

    def new_parser(self) -> Callable[[dict], ParsedEvent]:
        return EventParser()


class EventParser:
    """Codex emits schema-shaped preambles too, so only the last agent_message counts."""

    def __init__(self) -> None:
        self.last_message: str | None = None

    def __call__(self, event: dict) -> ParsedEvent:
        kind = event.get("type")
        item = event.get("item") or {}
        if kind == "item.completed" and item.get("type") == "agent_message":
            self.last_message = item.get("text")
        elif kind == "item.completed" and item.get("type") == "web_search":
            action = item.get("action") or {}
            if query := item.get("query") or action.get("query"):
                return ParsedEvent(activity=f"検索: {query}")
            if url := action.get("url"):
                return ParsedEvent(activity=f"ページ確認: {url}")
        elif kind == "turn.failed":
            message = (event.get("error") or {}).get("message", "不明なエラー")
            return ParsedEvent(error=str(message))
        elif kind == "error":
            return ParsedEvent(error=str(event.get("message", "不明なエラー")))
        elif kind == "turn.completed":
            parsed = parse_json_text(self.last_message or "")
            if parsed is None:
                return ParsedEvent(error="構造化された結果を取得できませんでした")
            return ParsedEvent(result=parsed)
        return ParsedEvent()
