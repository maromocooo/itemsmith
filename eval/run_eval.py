"""Score product identification against the known answers for the sample photo.

Copies a detected session, searches only the objects with a known answer and scores the
result. The answers are used for scoring only and never reach the agent prompt.

    cd backend && uv run python ../eval/run_eval.py <session_id> claude opus --hints none
    cd backend && uv run python ../eval/run_eval.py <session_id> codex gpt-6-astra --hints vague

`points` in ground_truth.json are on the photo scaled to 2000 px wide; each resolves to the
smallest detected box containing it. A key is correct when every point's answer matches.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "backend"))

from app.models_catalog import ADAPTERS  # noqa: E402
from app.search import SearchManager, display_name, representatives  # noqa: E402
from app.store import SessionStore  # noqa: E402


def object_at(objects: list[dict], x: float, y: float) -> dict | None:
    hits = [
        o for o in objects
        if o["bbox"][0] <= x <= o["bbox"][0] + o["bbox"][2] and o["bbox"][1] <= y <= o["bbox"][1] + o["bbox"][3]
    ]
    return min(hits, key=lambda o: o["bbox"][2] * o["bbox"][3], default=None)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id")
    parser.add_argument("provider", choices=["claude", "codex"])
    parser.add_argument("model")
    parser.add_argument("--hints", choices=["none", "vague"], default="none")
    args = parser.parse_args()

    truth = json.loads((ROOT / "eval" / "ground_truth.json").read_text())
    runs = ROOT / "data" / "eval_runs"
    source = ROOT / "data" / args.session_id
    session = json.loads((source / "session.json").read_text())
    new_id = SessionStore(runs).new_id()
    shutil.copytree(source, runs / new_id)

    scale = session["image"]["width"] / 2000
    targets: dict[str, list[dict | None]] = {}
    for entry in truth:
        targets[entry["key"]] = [
            object_at(session["objects"], px * scale, py * scale) for px, py in entry["points"]
        ]
    wanted = {o["id"]: e for e in truth for o in targets[e["key"]] if o}
    session.update(id=new_id, search={"provider": args.provider, "model": args.model})
    for obj in session["objects"]:
        obj["excluded"] = obj["id"] not in wanted
        if obj["id"] in wanted:
            obj["hint"] = wanted[obj["id"]]["vague_hint"] if args.hints == "vague" else ""
    (runs / new_id / "session.json").write_text(json.dumps(session, ensure_ascii=False))

    store = SessionStore(runs)
    live = store.get(new_id)
    manager = SearchManager(store, ADAPTERS)
    manager.start()
    pending = [o for o in live["objects"] if not o["excluded"]]
    for obj in representatives(pending):
        manager.enqueue(live, obj)
    while any(o["status"] in ("queued", "running") for o in pending):
        await asyncio.sleep(1)
    await manager.stop()

    correct = 0
    by_id = {o["id"]: o for o in live["objects"]}
    print(f"\n{args.provider}/{args.model} hints={args.hints} detected={len(live['objects'])}")
    for entry in truth:
        verdicts = []
        for found in targets[entry["key"]]:
            if found is None:
                verdicts.append((False, "(no box at this point)"))
                continue
            obj = by_id[found["id"]]
            name = (
                f"{display_name(obj['result'])} {obj['result']['model_number']} [{obj['result']['confidence']}]"
                if obj["result"] else f"FAILED: {obj['error']}"
            )
            ok = obj["result"] is not None and any(
                a in f"{display_name(obj['result'])} {obj['result']['model_number']}".lower() for a in entry["accept"]
            )
            verdicts.append((ok, f"#{obj['number']} {obj['label']} -> {name}"))
        passed = all(ok for ok, _ in verdicts)
        correct += passed
        print(f"{'OK ' if passed else 'NG '} {entry['key']} (truth: {entry['truth']})")
        for _, text in verdicts:
            print(f"     {text}")
    print(f"score: {correct}/{len(truth)}  run: {new_id}")


if __name__ == "__main__":
    asyncio.run(main())
