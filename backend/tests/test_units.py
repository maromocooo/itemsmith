import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from app.agents import claude as claude_mod
from app.agents.base import AgentError, AgentRequest, subscription_env
from app.agents.claude import ClaudeAdapter
from app.agents.codex import CodexAdapter, EventParser
from app.geometry import Tile, from_normalized, iou, make_tiles, merge_candidates, touches_inner_edge
from app import models_catalog
from app.models_catalog import catalog, validate_choice
from app.sam import accept_mask_box
from app.search import confirmed_name, sanitize_result, search_prompt


# --- catalog
def make_home(tmp_path: Path) -> Path:
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".codex" / "models_cache.json").write_text(json.dumps({"models": [
        {"slug": "b", "display_name": "B", "visibility": "list", "priority": 2},
        {"slug": "a", "display_name": "A", "visibility": "list", "priority": 1},
        {"slug": "hidden", "visibility": "hide", "priority": 0},
    ]}))
    (tmp_path / ".codex" / "config.toml").write_text('model = "b"\n')
    (tmp_path / ".claude.json").write_text(json.dumps({"additionalModelOptionsCache": [
        {"value": "claude-fable-5-1[1m]", "label": "Fable"},
    ]}))
    return tmp_path


def test_catalog_reads_local_config(tmp_path):
    claude, codex = catalog(make_home(tmp_path))
    assert [m["id"] for m in claude["models"]] == ["opus", "sonnet", "haiku", "claude-fable-5-1[1m]"]
    assert claude["default_model"] == "opus"
    assert [m["id"] for m in codex["models"]] == ["a", "b"]
    assert codex["default_model"] == "b"


def test_claude_labels_show_the_version_the_cli_reports(tmp_path, monkeypatch):
    resolved = {"opus": "claude-opus-5-5", "sonnet": None, "haiku": "claude-3-5-haiku-20241022",
                "claude-fable-5-1[1m]": "claude-fable-5-1"}
    calls = []

    def fake_resolve(model):
        calls.append(model)
        return resolved[model]

    monkeypatch.setattr(models_catalog, "resolve_claude_model", fake_resolve)
    monkeypatch.setattr(models_catalog.shutil, "which", lambda _: "/bin/claude")
    monkeypatch.setattr(models_catalog, "_resolved", {})
    models_catalog.refresh_claude_versions(list(resolved))
    labels = {m["id"]: m["label"] for m in catalog(make_home(tmp_path))[0]["models"]}
    assert labels == {"opus": "Opus 5.5", "sonnet": "Sonnet", "haiku": "claude-3-5-haiku-20241022",
                      "claude-fable-5-1[1m]": "Fable 5.1(1M)"}

    calls.clear()
    models_catalog.refresh_claude_versions(["opus", "--bad"])
    assert calls == []  # cached, and flag-like names never reach the CLI

    options = models_catalog.label_claude_options([{"id": "opus", "label": "Opus"}, {"id": "sonnet", "label": "Sonnet · fast"}])
    assert [o["label"] for o in options] == ["Opus 5.5", "Sonnet"]


def test_version_label():
    assert models_catalog.version_label("claude-sonnet-5-5") == "Sonnet 5.5"
    assert models_catalog.version_label("claude-opus-4-20250514") == "Opus 4"
    assert models_catalog.version_label("something-else") == "something-else"


def test_catalog_survives_missing_files(tmp_path):
    claude, codex = catalog(tmp_path)
    assert claude["default_model"] == "opus"
    assert codex["models"] == [] and codex["default_model"] is None


def test_validate_choice_rejects_unlisted_and_flag_like_models(tmp_path):
    home = make_home(tmp_path)
    assert validate_choice("nope", "a", home)
    assert validate_choice("codex", "--dangerously-bypass-approvals-and-sandbox", home)
    assert validate_choice("codex", "unlisted", home)


# --- environment
def test_subscription_env_strips_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("KEEP_ME", "1")
    env = subscription_env()
    assert "ANTHROPIC_API_KEY" not in env and "OPENAI_API_KEY" not in env and "CLAUDECODE" not in env
    assert env["KEEP_ME"] == "1"


# --- commands keep the agents restricted
def test_claude_command_limits_tools(tmp_path):
    req = AgentRequest(prompt="p", schema={}, model="opus", web=True)
    cmd = ClaudeAdapter().build_command(req, tmp_path)
    assert cmd[cmd.index("--tools") + 1] == "WebSearch,WebFetch"
    assert "--strict-mcp-config" in cmd and cmd[cmd.index("--setting-sources") + 1] == ""
    no_web = ClaudeAdapter().build_command(AgentRequest(prompt="p", schema={}, model="opus"), tmp_path)
    assert no_web[no_web.index("--tools") + 1] == "" and "--allowedTools" not in no_web


def test_codex_command_limits_tools(tmp_path):
    req = AgentRequest(prompt="--evil", schema={"type": "object"}, model="m", web=True, images=[Path("/i.jpg")])
    cmd = CodexAdapter().build_command(req, tmp_path)
    assert cmd[:3] == ["codex", "--search", "exec"]
    assert cmd[cmd.index("-s") + 1] == "read-only"
    disabled = {cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"}
    assert {"shell_tool", "unified_exec", "apps", "plugins", "view_image", "hooks"} <= disabled
    assert cmd[-2:] == ["--", "--evil"]  # prompts can never become flags
    assert "--search" not in CodexAdapter().build_command(AgentRequest(prompt="p", schema={}, model="m"), tmp_path)


# --- event parsing, on the shapes recorded from the real CLIs
def test_claude_events():
    parse = claude_mod.parse_event
    search = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "WebSearch", "input": {"query": "acme"}}]}}
    assert parse(search).activity == "検索: acme"
    ok = {"type": "result", "subtype": "success", "is_error": False, "structured_output": {"a": 1}, "result": ""}
    assert parse(ok).result == {"a": 1}
    bad = {"type": "result", "subtype": "success", "is_error": True, "result": "limit reached"}
    assert parse(bad).error == "limit reached"
    assert parse({"type": "system", "subtype": "thinking_tokens"}).result is None


def test_codex_uses_last_agent_message():
    parser = EventParser()
    parser({"type": "item.completed", "item": {"type": "agent_message", "text": '{"preamble": true}'}})
    item = {"id": "x", "type": "web_search", "query": "q", "action": {"type": "search", "query": "q"}}
    assert parser({"type": "item.completed", "item": item}).activity == "検索: q"
    parser({"type": "item.completed", "item": {"type": "agent_message", "text": '{"final": true}'}})
    assert parser({"type": "turn.completed"}).result == {"final": True}
    assert EventParser()({"type": "turn.failed", "error": {"message": "nope"}}).error == "nope"
    assert EventParser()({"type": "turn.completed"}).error


# --- process runner, with a scripted stand-in for the CLI
class ScriptAdapter(CodexAdapter):
    def __init__(self, script: str) -> None:
        self.script = script

    def build_command(self, req, workdir):
        return [sys.executable, "-c", self.script]


async def test_runner_success_streams_activity():
    lines = [
        {"type": "item.completed", "item": {"type": "web_search", "query": "q"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": '{"ok": 1}'}},
        {"type": "turn.completed"},
    ]
    script = "\n".join(f"print({json.dumps(json.dumps(l))})" for l in lines)
    seen = []
    result = await ScriptAdapter(script).run(AgentRequest(prompt="", schema={}, model="m"), seen.append)
    assert result == {"ok": 1} and seen == ["検索: q"]


async def test_runner_reports_failure_and_timeout():
    with pytest.raises(AgentError, match="boom"):
        await ScriptAdapter("import sys; sys.stderr.write('boom'); sys.exit(3)").run(AgentRequest(prompt="", schema={}, model="m"))
    with pytest.raises(AgentError, match="完了しませんでした"):
        await ScriptAdapter("import time; time.sleep(30)").run(AgentRequest(prompt="", schema={}, model="m", timeout=0.2))


# --- geometry
def test_tiles_cover_image_and_convert():
    tiles = make_tiles(1000, 800)
    assert len(tiles) == 4
    assert tiles[-1].x + tiles[-1].w == 1000 and tiles[-1].y + tiles[-1].h == 800
    assert from_normalized([0, 0, 500, 500], Tile(100, 200, 400, 400)) == [100, 200, 200, 200]
    assert from_normalized([5, 5, 5, 9], Tile(0, 0, 10, 10)) is None
    assert from_normalized([-50, 0, 2000, 500], Tile(0, 0, 10, 10)) == [0, 0, 10, 5]


def test_inner_edge_only():
    tile = Tile(0, 0, 500, 500)
    assert touches_inner_edge([400, 0, 100, 50], tile, 1000, 1000)
    assert not touches_inner_edge([0, 0, 50, 50], tile, 1000, 1000)


def cand(box, category="audio", rank=0, label="x"):
    return {"box": box, "category": category, "rank": rank, "label": label}


def test_merge_keeps_best_and_small_objects_on_large():
    full = cand([0, 0, 100, 100], rank=0)
    duplicate = cand([2, 2, 100, 100], rank=1)
    partial = cand([0, 0, 50, 100], rank=2)  # same object cut by a tile border
    mouse = cand([700, 700, 20, 20], category="input_device")
    pad = cand([600, 600, 300, 300], category="accessory")
    merged = merge_candidates([partial, duplicate, full, mouse, pad])
    boxes = [m["box"] for m in merged]
    assert full["box"] in boxes and duplicate["box"] not in boxes and partial["box"] not in boxes
    assert mouse["box"] in boxes and pad["box"] in boxes


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert iou([0, 0, 10, 10], [20, 20, 5, 5]) == 0


def test_mask_box_guards():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:50, 10:50] = True
    assert accept_mask_box([5, 5, 50, 50], mask) == [10, 10, 40, 40]
    assert accept_mask_box([60, 60, 30, 30], mask) is None  # latched onto something else
    assert accept_mask_box([5, 5, 50, 50], np.zeros((100, 100), dtype=bool)) is None
    big = np.ones((100, 100), dtype=bool)
    assert accept_mask_box([40, 40, 40, 40], big) is None  # bled across the image


# --- search helpers
def test_sanitize_drops_unsafe_urls():
    result = sanitize_result({
        "brand": " Acme ", "product_name": "Widget", "confidence": "bogus",
        "source_urls": ["javascript:alert(1)", "https://a.example/x", "http://[bad"],
    })
    assert result["brand"] == "Acme" and result["confidence"] == "low"
    assert result["source_urls"] == ["https://a.example/x"]


def test_prompt_carries_hint_and_rejections():
    obj = {
        "label": "モニター", "hint": "Acme 曲面",
        "history": [{"result": {"brand": "Acme", "product_name": "Wrong"}, "feedback": "大型"}],
    }
    prompt = search_prompt([obj])
    assert "Acme 曲面" in prompt and "Acme Wrong" in prompt and "大型" in prompt


def test_confirmed_name_keeps_model_number_once():
    assert confirmed_name({"brand": "Acme", "product_name": "Widget", "model_number": "X100-B"}) == "Acme Widget X100-B"
    assert confirmed_name({"brand": "Acme", "product_name": "X100-B", "model_number": "x100-b"}) == "Acme X100-B"
    assert confirmed_name({"brand": "", "product_name": "Widget Pro", "model_number": ""}) == "Widget Pro"
