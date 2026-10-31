# itemsmith

[English](README.md) | **日本語**

写真の製品を1つずつ枠で囲み、ヒントとエージェントの Web 調査で商品名とサイズを特定し、[glbsmith](https://github.com/maromocooo/glbsmith) で実寸の GLB にまとめて作るローカル Web アプリ。

## 単独利用と glbsmith / Room Setup との連携

itemsmith は製品の特定・調査だけでも単独で使えます。部屋の写真と任意のヒントを渡し、製品候補の正誤を確認して、参照画像と寸法を見直します。ここで終了する場合、glbsmith や Room Setup のインストールは不要です。

3D 化する場合は [glbsmith](https://github.com/maromocooo/glbsmith) を起動し、itemsmith の既存の一括作成機能から確認済みの写真と寸法を渡します。完成した GLB をダウンロードし、[Room Setup](https://github.com/maromocooo/room-setup) の **家具 → GLB を読み込む** で部屋に配置します。

セットアップ、画面の操作名、手動読み込みの手順は **[統合ガイド（英語）](https://github.com/maromocooo/room-setup/blob/main/docs/end-to-end.md)** を参照してください。3つは独立したツールであり、部屋を完全自動で再現するものではありません。製品候補・参照画像・寸法・生成モデルはユーザーによる確認が必要です。

## 起動

```sh
cd frontend && npm install && npm run build
cd ../backend && uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

http://127.0.0.1:8000 を開く。`claude` / `codex` CLI にサブスクリプションでログイン済みであること。GLB を作るときは glbsmith も起動しておく(`uv run python -m glbsmith.app`、http://127.0.0.1:8021)。SAM 2.1 のモデル(約 180MB)は初回検出時に自動でダウンロードされる。

フロントエンドを開発するときは `npm run dev`(http://localhost:5173、API はポート 8000 へプロキシ)。

## 仕組み

1. **検出** (`app/detect.py`): 全体 + 4 分割の画像をエージェントに渡して候補を集め、統合し、SAM 2.1 で枠を密着させ、番号付き画像でエージェントが点検する。
2. **ヒント入力**: リストと元写真が連動する。ヒントは任意。
3. **商品検索** (`app/search.py`): 1 製品につき 1 エージェント。進捗は SSE で配信する。左右のスピーカーや、デスクの天板と脚のように 1 つの商品として買う物は、検出時に 1 製品にまとめてリストも 1 行にする。検索前なら、ヒント入力画面のチェックボックスで人がまとめ直せる(「別々に調べる」「N を外す」で分割)。
4. **確定**: 「合っている」「間違っている」で判定。誤りなら追加情報で再検索するか、入力した名前で確定する。
5. **サイズと製品写真の取得** (`app/sizes.py`, `app/product_photos.py`): 確定した製品ごとにエージェントが公式仕様などから幅・奥行き・高さと出典を探し、角度の違う製品写真の URL も集める。サーバーが写真をダウンロードし、別のエージェントが実物の切り抜きと見比べて同じ製品か確認する。写真の状態(スタンドなしなど)に合う寸法を選び、寸法と使う写真は人が直せる。
6. **GLB 作成** (`app/builds.py`): 実物の写真(マスク付き)を先頭に、選んだ製品写真とサイズを glbsmith の `POST /api/jobs` に渡し、完成した GLB と画像を `data/<session>/models/` に取り込む。グループは「部品を合わせて 1 つ」か「同じ物が複数(1 つ分を作る)」を選べる。キューと同時実行数は glbsmith 側の設定に従う。

データは `data/<session>/` に保存する。処理中にサーバーを止めても、次回起動時に待機中・調査中の検索が再開される。

## エージェントの権限

Web ページの内容に誘導されても被害が出ないよう、できることを絞っている。

- Claude Code: `WebSearch` / `WebFetch` のみ。MCP とユーザー設定は読み込まない。
- Codex: read-only サンドボックスに加えて、シェル、コネクタ、プラグイン、画像読み込み、hooks を無効化。`apply_patch` とサブエージェントのツールは無効化できず残るが、書き込みはサンドボックスで拒否される。
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` などは子プロセスから除去し、必ずサブスクリプションで動かす。
- サーバーは localhost からのリクエストのみ受け付ける。
- 製品写真のダウンロード先はエージェントが見つけた URL なので、公開 IP アドレスの http(s) に限る(リダイレクト先も確認)。画像はデコードして JPEG に変換し直してから使う。
- サイズ取得のエージェントも商品検索と同じ制限で動く。GLB を作るエージェントは glbsmith 側の制限(Landlock など)で動き、サンドボックスが使えない環境では実行を許可しない。

環境変数: `SEARCH_CONCURRENCY`(同時実行数、既定 3)、`SEARCH_TIMEOUT`(秒、既定 600)、`DATA_DIR`、`FRONTEND_DIST`、`GLBSMITH_URL`(既定 http://127.0.0.1:8021)、`GLBSMITH_POLL_SECONDS`(既定 5)。

## 画像の権利

製品写真は Web から取得したもので、権利はそれぞれの権利者にあります。個人での利用(自分の部屋の再現など)にとどめ、取得した写真や、そこから作ったモデルを公開しないこと。`data/` は git に含めない。

## テスト

```sh
cd backend
uv run pytest
# 正解がわかっているサンプルでの採点(正解はプロンプトに渡さない)
uv run python ../eval/run_eval.py <session_id> claude opus --hints none
uv run python ../eval/run_eval.py <session_id> codex gpt-6-astra --hints vague
```

正解データの `eval/ground_truth.json` は、写真の持ち物がわかるので git に含めない。自分の写真で採点するときは、次の形式で作る(`points` は写真を幅 2000px にしたときの座標)。

```json
[{"key": "monitor", "truth": "正解の商品名", "points": [[1000, 500]], "accept": ["結果に含まれていれば正解とする文字列"], "vague_hint": "曖昧なヒント"}]
```

## ライセンス

itemsmith は [MIT](LICENSE) ライセンスです。対象はコードのみで、Web から取得した製品写真やそこから作ったモデルは含みません([画像の権利](#画像の権利)を参照)。
