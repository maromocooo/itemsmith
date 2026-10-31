from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Callable

from .base import AgentAdapter, AgentRequest, ParsedEvent, parse_json_text

WEB_TOOLS = "WebSearch,WebFetch"


class ClaudeAdapter(AgentAdapter):
    provider = "claude"

    def build_command(self, req: AgentRequest, workdir: Path) -> list[str]:
        tools = WEB_TOOLS if req.web else ""
        cmd = [
            "claude", "-p",
            "--model", req.model,
            "--output-format", "stream-json",
            "--input-format", "stream-json",
            "--verbose",
            "--tools", tools,
            "--strict-mcp-config",
            "--setting-sources", "",
            "--no-session-persistence",
            "--json-schema", json.dumps(req.schema),
        ]
        if req.web:
            cmd += ["--allowedTools", WEB_TOOLS]
        return cmd

    def build_stdin(self, req: AgentRequest) -> bytes:
        content: list[dict] = []
        for path in req.images:
            media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            data = base64.b64encode(path.read_bytes()).decode()
            content.append(
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
            )
        content.append({"type": "text", "text": req.prompt})
        message = {"type": "user", "message": {"role": "user", "content": content}}
        return (json.dumps(message) + "\n").encode()

    def new_parser(self) -> Callable[[dict], ParsedEvent]:
        return parse_event


def parse_event(event: dict) -> ParsedEvent:
    kind = event.get("type")
    if kind == "assistant":
        for block in event.get("message", {}).get("content", []):
            if block.get("type") != "tool_use":
                continue
            tool_input = block.get("input") or {}
            if block.get("name") == "WebSearch":
                return ParsedEvent(activity=f"検索: {tool_input.get('query', '')}")
            if block.get("name") == "WebFetch":
                return ParsedEvent(activity=f"ページ確認: {tool_input.get('url', '')}")
    if kind == "result":
        if event.get("is_error") or event.get("subtype") != "success":
            return ParsedEvent(error=str(event.get("result") or event.get("subtype") or "不明なエラー"))
        structured = event.get("structured_output")
        if isinstance(structured, dict):
            return ParsedEvent(result=structured)
        parsed = parse_json_text(event.get("result") or "")
        if parsed is not None:
            return ParsedEvent(result=parsed)
        return ParsedEvent(error="構造化された結果を取得できませんでした")
    return ParsedEvent()
