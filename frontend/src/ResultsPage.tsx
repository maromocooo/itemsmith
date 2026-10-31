import { useEffect, useState } from "react";
import { DetectedObject, ObjectStatus, Session, SearchResult, api, groupProducts } from "./api";
import { PhotoPane, ProductImages, ProductTitle, useLinkedRow } from "./PhotoPane";

const STATUS_LABEL: Record<ObjectStatus, string> = {
  idle: "対象外",
  queued: "待機中",
  running: "調査中",
  done: "結果あり",
  rejected: "要修正",
  confirmed: "確定",
  failed: "失敗",
};

const CONFIDENCE_LABEL = { high: "確信度 高", medium: "確信度 中", low: "確信度 低" };

const resultName = (r: SearchResult) => [r.brand, r.product_name].filter(Boolean).join(" ");

export function ResultsPage({ session }: { session: Session }) {
  const [activeId, setActiveId] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [groupError, setGroupError] = useState<string | null>(null);
  const { register, jumpTo, flashId } = useLinkedRow("[data-row-focus]");
  const now = useNow(session.objects.some((o) => o.status === "running"));

  const targets = session.objects.filter((o) => !o.excluded);
  const products = groupProducts(targets);
  const queue = products
    .map(([o]) => o)
    .filter((o) => o.status === "queued")
    .sort((a, b) => (a.queued_at ?? 0) - (b.queued_at ?? 0))
    .map((o) => o.id);
  const count = (...statuses: ObjectStatus[]) => products.filter(([o]) => statuses.includes(o.status)).length;
  const finished = count("done", "rejected", "confirmed", "failed");
  const chosen = products.filter(([o]) => selected.has(o.id));

  function toggle(oid: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (!next.delete(oid)) next.add(oid);
      return next;
    });
  }

  async function regroup(action: () => Promise<unknown>) {
    setGroupError(null);
    try {
      await action();
      setSelected(new Set());
    } catch (e) {
      setGroupError((e as Error).message);
    }
  }
  const percent = products.length ? (finished / products.length) * 100 : 0;

  return (
    <div className="workspace">
      <main className="list-pane">
        <div className="pane-heading">
          <h1>検索結果</h1>
          <p className="muted">
            {session.search!.provider === "claude" ? "Claude Code" : "Codex"} / {session.search!.model}
          </p>
          <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={products.length} aria-valuenow={finished}>
            <span style={{ width: `${percent}%` }} />
          </div>
          <p className="summary">
            調査完了 <strong>{finished}</strong> / {products.length}
            <span>調査中 {count("running")}</span>
            <span>待機中 {count("queued")}</span>
            <span>確定 {count("confirmed")}</span>
            {count("failed") > 0 && <span className="failed-count">失敗 {count("failed")}</span>}
          </p>
          <a className="next-link" href={`#/s/${session.id}/build`}>サイズ取得と 3D 作成へ(確定 {count("confirmed")} 件)</a>
          {chosen.length > 0 && (
            <div className="group-bar results-group-bar">
              <span className="muted">{chosen.length} 件を選択中</span>
              <span className="group-actions">
                <button type="button" className="text-button" onClick={() => setSelected(new Set())}>選択を解除</button>
                <button
                  type="button"
                  disabled={chosen.length < 2}
                  onClick={() => regroup(() => api.createGroup(session.id, chosen.map(([o]) => o.id), {}))}
                >
                  同じ製品としてまとめる
                </button>
              </span>
            </div>
          )}
          {groupError && <p className="error" role="alert">{groupError}</p>}
        </div>
        <ul className="object-list">
          {products.map((members) => {
            const o = members[0];
            const has = (id: string | null) => members.some((m) => m.id === id);
            const hint = members.map((m) => m.hint).filter(Boolean).join(" / ");
            return (
              <li
                key={o.id}
                ref={register(members.map((m) => m.id))}
                className={`object-row result-row status-${o.status} ${has(activeId) ? "active" : ""} ${has(flashId) ? "flash" : ""}`}
                onMouseEnter={() => setActiveId(o.id)}
                onMouseLeave={() => setActiveId(null)}
                onFocus={() => setActiveId(o.id)}
              >
                <ProductImages session={session} members={members} />
                <div className="row-body">
                  <div className="row-title">
                    <input
                      type="checkbox"
                      className="row-check"
                      checked={selected.has(o.id)}
                      disabled={o.status === "queued" || o.status === "running" || Boolean(o.build)}
                      aria-label={`${o.label} をまとめる対象に選ぶ`}
                      onChange={() => toggle(o.id)}
                    />
                    <ProductTitle members={members} />
                    <span className={`status-chip status-${o.status}`}>{STATUS_LABEL[o.status]}</span>
                  </div>
                  {hint && <p className="hint-line">ヒント: {hint}</p>}
                  {members.length > 1 && !o.build && o.status !== "queued" && o.status !== "running" && (
                    <p className="group-note">
                      1 つの製品として扱います
                      <button type="button" className="text-button" onClick={() => regroup(() => api.patchObject(session.id, o.id, { ungroup: true }))}>
                        別々にする
                      </button>
                    </p>
                  )}
                  <RowContent session={session} obj={o} now={now} queuePosition={queue.indexOf(o.id)} />
                </div>
              </li>
            );
          })}
        </ul>
      </main>
      <aside className="photo-pane">
        <PhotoPane session={session} objects={targets} activeId={activeId} onHover={setActiveId} onSelect={jumpTo} />
      </aside>
    </div>
  );
}

function RowContent({ session, obj, now, queuePosition }: { session: Session; obj: DetectedObject; now: number; queuePosition: number }) {
  switch (obj.status) {
    case "queued":
      return <p className="muted" tabIndex={-1} data-row-focus>順番待ち({queuePosition + 1} 番目)</p>;
    case "running":
      return (
        <div tabIndex={-1} data-row-focus>
          <div className="progress indeterminate" role="progressbar" aria-label="調査中"><span /></div>
          <p className="activity">
            <span className="elapsed">{formatElapsed(now - (obj.started_at ?? now / 1000) * 1000)}</span>
            {obj.activity}
          </p>
        </div>
      );
    case "done":
      return <DoneContent session={session} obj={obj} />;
    case "rejected":
      return <CorrectionForm session={session} obj={obj} />;
    case "failed":
      return <FailedContent session={session} obj={obj} />;
    case "confirmed":
      return <ConfirmedContent session={session} obj={obj} />;
    default:
      return null;
  }
}

function useAction() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return { busy, error, run };
}

function ResultDetails({ result }: { result: SearchResult }) {
  return (
    <>
      <p className="product-name">
        {resultName(result)}
        {result.model_number && <span className="model-number">{result.model_number}</span>}
        <span className={`confidence ${result.confidence}`}>{CONFIDENCE_LABEL[result.confidence]}</span>
      </p>
      {result.reasoning && <p className="reasoning">{result.reasoning}</p>}
      {result.source_urls.length > 0 && (
        <ul className="sources">
          {result.source_urls.map((url) => (
            <li key={url}>
              <a href={url} target="_blank" rel="noreferrer noopener">{hostOf(url)}</a>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

function DoneContent({ session, obj }: { session: Session; obj: DetectedObject }) {
  const { busy, error, run } = useAction();
  return (
    <>
      <ResultDetails result={obj.result!} />
      <div className="verdict-buttons">
        <button type="button" className="confirm" data-row-focus disabled={busy} onClick={() => run(() => api.verdict(session.id, obj.id, "correct"))}>
          合っている
        </button>
        <button type="button" className="reject" disabled={busy} onClick={() => run(() => api.verdict(session.id, obj.id, "wrong"))}>
          間違っている
        </button>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
    </>
  );
}

function CorrectionForm({ session, obj }: { session: Session; obj: DetectedObject }) {
  const { busy, error, run } = useAction();
  const [text, setText] = useState("");
  const blank = !text.trim();
  return (
    <div className="correction">
      <p className="rejected-name">{resultName(obj.result!)}</p>
      <input
        type="text"
        value={text}
        maxLength={300}
        data-row-focus
        autoFocus
        aria-label="より詳しい特徴、または正しい商品名"
        placeholder="より詳しい特徴、または正しい商品名"
        onChange={(e) => setText(e.target.value)}
      />
      <div className="verdict-buttons">
        <button type="button" className="primary" disabled={busy || blank} onClick={() => run(() => api.retry(session.id, obj.id, text))}>
          再検索
        </button>
        <button type="button" disabled={busy || blank} onClick={() => run(() => api.verdict(session.id, obj.id, "manual", text))}>
          この名前で確定
        </button>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
    </div>
  );
}

function FailedContent({ session, obj }: { session: Session; obj: DetectedObject }) {
  const { busy, error, run } = useAction();
  const [text, setText] = useState("");
  return (
    <div className="correction">
      <p className="error">{obj.error}</p>
      <input
        type="text"
        value={text}
        maxLength={300}
        data-row-focus
        aria-label="追加のヒント、または商品名"
        placeholder="追加のヒント、または商品名(任意)"
        onChange={(e) => setText(e.target.value)}
      />
      <div className="verdict-buttons">
        <button type="button" className="primary" disabled={busy} onClick={() => run(() => api.retry(session.id, obj.id, text))}>
          再試行
        </button>
        <button type="button" disabled={busy || !text.trim()} onClick={() => run(() => api.verdict(session.id, obj.id, "manual", text))}>
          この名前で確定
        </button>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
    </div>
  );
}

function ConfirmedContent({ session, obj }: { session: Session; obj: DetectedObject }) {
  const { busy, error, run } = useAction();
  return (
    <>
      <p className="product-name confirmed">
        {obj.confirmed_name}
        {obj.confirmed_by === "user" && <span className="model-number">手入力</span>}
      </p>
      <button type="button" className="text-button" data-row-focus disabled={busy} onClick={() => run(() => api.verdict(session.id, obj.id, "reopen"))}>
        確定を取り消す
      </button>
      {error && <p className="error" role="alert">{error}</p>}
    </>
  );
}

function useNow(ticking: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!ticking) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [ticking]);
  return now;
}

function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}
