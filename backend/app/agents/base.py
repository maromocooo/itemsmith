"""Common runner for the subscription agent CLIs (Claude Code / Codex)."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

STREAM_LIMIT = 64 * 1024 * 1024  # tool results arrive as one very long JSONL line
STDERR_TAIL = 2000

ActivityCallback = Callable[[str], Awaitable[None] | None]


class AgentError(RuntimeError):
    pass


@dataclass
class AgentRequest:
    prompt: str
    schema: dict
    model: str
    images: list[Path] = field(default_factory=list)
    web: bool = False
    timeout: float = 600.0


@dataclass
class ParsedEvent:
    activity: str | None = None
    result: dict | None = None
    error: str | None = None


def subscription_env() -> dict[str, str]:
    """Child environment that can only authenticate through the subscription login.

    API keys would silently switch the CLIs to metered billing, and CLAUDE* variables
    would make a nested Claude Code think it is a child of the current session.
    """
    blocked = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY"}
    return {
        k: v
        for k, v in os.environ.items()
        if k not in blocked and not k.startswith("CLAUDE")
    }


def parse_json_text(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


class AgentAdapter:
    provider: str

    def build_command(self, req: AgentRequest, workdir: Path) -> list[str]:
        raise NotImplementedError

    def build_stdin(self, req: AgentRequest) -> bytes | None:
        return None

    def new_parser(self) -> Callable[[dict], ParsedEvent]:
        raise NotImplementedError

    async def run(self, req: AgentRequest, on_activity: ActivityCallback | None = None) -> dict:
        with tempfile.TemporaryDirectory(prefix="itemsmith-agent-") as tmp:
            workdir = Path(tmp)
            return await self._run_in(req, workdir, on_activity)

    async def _run_in(
        self, req: AgentRequest, workdir: Path, on_activity: ActivityCallback | None
    ) -> dict:
        stdin_data = self.build_stdin(req)
        proc = await asyncio.create_subprocess_exec(
            *self.build_command(req, workdir),
            cwd=workdir,
            env=subscription_env(),
            stdin=asyncio.subprocess.PIPE if stdin_data else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=STREAM_LIMIT,
            start_new_session=True,
        )
        parser = self.new_parser()
        outcome = ParsedEvent()
        stderr_tail = bytearray()

        async def feed_stdin() -> None:
            if stdin_data:
                proc.stdin.write(stdin_data)
                await proc.stdin.drain()
                proc.stdin.close()

        async def drain_stderr() -> None:
            while chunk := await proc.stderr.read(4096):
                stderr_tail.extend(chunk)
                del stderr_tail[:-STDERR_TAIL]

        async def read_stdout() -> None:
            async for raw in proc.stdout:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                parsed = parser(event)
                if parsed.result is not None:
                    outcome.result = parsed.result
                if parsed.error:
                    outcome.error = parsed.error
                if parsed.activity and on_activity:
                    maybe = on_activity(parsed.activity)
                    if asyncio.iscoroutine(maybe):
                        await maybe

        try:
            await asyncio.wait_for(
                asyncio.gather(feed_stdin(), drain_stderr(), read_stdout(), proc.wait()),
                timeout=req.timeout,
            )
        except asyncio.TimeoutError:
            raise AgentError(f"{req.timeout:.0f} 秒以内に完了しませんでした") from None
        finally:
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()

        if outcome.result is not None and proc.returncode == 0:
            return outcome.result
        detail = outcome.error or stderr_tail.decode("utf-8", "replace").strip()
        raise AgentError(detail[-1000:] or f"エージェントが終了コード {proc.returncode} で終了しました")
