# itemsmith

**English** | [日本語](README.ja.md)

A local web app that boxes each product in a photo, identifies its name and size from your hints and an agent's web research, and turns them into true-to-size GLB models in one batch with [glbsmith](https://github.com/maromocooo/glbsmith).

## Use it on its own or with glbsmith and Room Setup

itemsmith works independently for product identification and research: provide a room photo and optional hints, confirm or correct the product matches, and review the reference photos and dimensions. You can stop there without installing glbsmith or Room Setup.

To continue into 3D, start [glbsmith](https://github.com/maromocooo/glbsmith) and use itemsmith's existing batch build flow to send the reviewed photos and dimensions to it. Download the resulting GLBs and use **Furniture → Import GLB** in [Room Setup](https://github.com/maromocooo/room-setup) to place them in a room.

Follow the **[end-to-end guide](https://github.com/maromocooo/room-setup/blob/main/docs/end-to-end.md)** for setup, the Japanese UI labels and the manual import step. These are three independent tools, not a fully automatic room reconstruction: product matches, collected references, dimensions and generated models still need your review.

## Quick start

```sh
cd frontend && npm install && npm run build
cd ../backend && uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. The app's interface is in Japanese. The `claude` / `codex` CLI must be logged in with a subscription. To build GLBs, start glbsmith as well (`uv run python -m glbsmith.app`, http://127.0.0.1:8021). The SAM 2.1 model (about 180 MB) downloads automatically on the first detection.

For frontend development, use `npm run dev` (http://localhost:5173, with the API proxied to port 8000).

## How it works

1. **Detect** (`app/detect.py`): The agent receives the whole photo and four overlapping tiles and proposes candidates. These are merged, SAM 2.1 tightens each box, and the agent reviews a numbered image.
2. **Add hints**: The list and the original photo are linked. Hints are optional.
3. **Search for products** (`app/search.py`): One agent per product, with progress streamed over SSE. Things bought as one product, such as left and right speakers or a desk's top and legs, are grouped during detection and take one row in the list. Before searching, you can regroup them with the checkboxes on the hints page (「別々に調べる」, search separately, and 「N を外す」, remove N, split a group).
4. **Confirm**: Mark each result as correct or wrong. For a wrong result, either search again with more detail or confirm the name you typed.
5. **Get sizes and product photos** (`app/sizes.py`, `app/product_photos.py`): For each confirmed product, an agent looks for the width, depth and height in sources such as official specs, and collects URLs of product photos from different angles. The server downloads the photos, and a separate agent compares them with the crop from your photo to check they show the same product. Sizes are chosen to match the state in the photo (for example, without a stand), and you can edit both the sizes and the photos to use.
6. **Build GLBs** (`app/builds.py`): The real photos (with masks) go first, followed by the selected product photos and the size, to glbsmith's `POST /api/jobs`. The finished GLB and image are copied to `data/<session>/models/`. A group is built either as parts of one thing (「部品を合わせて 1 つ」) or as several identical things (「同じ物が複数」, which builds one). The queue and concurrency follow glbsmith's settings.

Data is stored in `data/<session>/`. If the server stops during processing, queued and running searches resume on the next start.

## Agent permissions

What agents can do is limited, so that a web page steering them cannot cause harm.

- Claude Code: only `WebSearch` / `WebFetch`. MCP and user settings are not loaded.
- Codex: a read-only sandbox, with shell, connectors, plugins, image viewing and hooks disabled. The `apply_patch` and subagent tools cannot be disabled and remain, but the sandbox refuses writes.
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and similar variables are removed from child processes, so agents always run on your subscription.
- The server only accepts requests from localhost.
- Product photo URLs are found by an agent, so downloads are limited to http(s) on public IP addresses, including redirect targets. Images are decoded and re-encoded as JPEG before use.
- Size lookup agents run under the same limits as product search. The agents that build GLBs run under glbsmith's own limits (such as Landlock), and glbsmith refuses to run them where a sandbox is not available.

Environment variables: `SEARCH_CONCURRENCY` (concurrent agents, default 3), `SEARCH_TIMEOUT` (seconds, default 600), `DATA_DIR`, `FRONTEND_DIST`, `GLBSMITH_URL` (default http://127.0.0.1:8021), `GLBSMITH_POLL_SECONDS` (default 5).

## Image rights

Product photos come from the web, and their rights belong to their respective holders. Keep them to personal use (such as recreating your own room), and do not publish the photos or the models made from them. `data/` is not included in git.

## Tests

```sh
cd backend
uv run pytest
# Score against samples with known answers (the answers are never passed to the prompt)
uv run python ../eval/run_eval.py <session_id> claude opus --hints none
uv run python ../eval/run_eval.py <session_id> codex gpt-6-astra --hints vague
```

The answer file `eval/ground_truth.json` is not included in git, because it reveals what is in the photo. To score your own photo, create it in this format (`points` are coordinates on the photo scaled to a width of 2000 px):

```json
[{"key": "monitor", "truth": "Correct product name", "points": [[1000, 500]], "accept": ["text that counts as correct if found in the result"], "vague_hint": "A vague hint"}]
```

## License

itemsmith is [MIT](LICENSE) licensed. This covers the code only, not product photos fetched from the web or models made from them (see [Image rights](#image-rights)).
