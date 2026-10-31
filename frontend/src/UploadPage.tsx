import { DragEvent, useEffect, useRef, useState } from "react";
import { SessionSummary, api } from "./api";
import { AgentPicker, useAgents } from "./AgentPicker";

export function UploadPage() {
  const { providers, choice, setChoice, error: agentError } = useAgents();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const reload = () => api.sessions().then(setSessions, (e: Error) => setError(e.message));
  useEffect(() => {
    reload();
  }, []);

  async function upload(file: File) {
    if (!choice) return;
    setBusy(true);
    setError(null);
    try {
      const session = await api.create(file, choice.provider, choice.model);
      window.location.hash = `#/s/${session.id}`;
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) upload(file);
  }

  async function remove(session: SessionSummary) {
    if (!window.confirm(`「${session.filename}」のセッションを削除しますか?`)) return;
    try {
      await api.deleteSession(session.id);
      reload();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <main className="upload-page">
      <h1>写真の製品を特定する</h1>
      <p className="lead">写真の製品を1つずつ枠で囲み、ヒントをもとにエージェントが商品名を調べます。</p>

      <section className="field-block">
        <h2>検出に使うエージェント</h2>
        <AgentPicker providers={providers} choice={choice} onChange={setChoice} disabled={busy} />
      </section>

      <button
        type="button"
        className={`dropzone ${dragging ? "dragging" : ""}`}
        disabled={busy || !choice}
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <strong>{busy ? "アップロード中…" : "写真を選択、またはドロップ"}</strong>
        <span>JPEG / PNG / WebP、50MB まで</span>
      </button>
      <input
        ref={input}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (file) upload(file);
        }}
      />
      {(error || agentError) && <p className="error" role="alert">{error || agentError}</p>}

      {sessions.length > 0 && (
        <section className="session-list">
          <h2>これまでのセッション</h2>
          <ul>
            {sessions.map((s) => (
              <li key={s.id}>
                <a href={`#/s/${s.id}`}>
                  <span className="session-name">{s.filename}</span>
                  <span className="session-meta">
                    {new Date(s.created_at * 1000).toLocaleString("ja-JP")} · {describe(s)}
                  </span>
                </a>
                <button type="button" className="text-button" onClick={() => remove(s)}>削除</button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}

function describe(s: SessionSummary): string {
  if (s.status === "detecting") return "検出中";
  if (s.status === "detect_failed") return "検出失敗";
  return `${s.object_count} 件${s.search_started ? "・検索開始済み" : ""}`;
}
