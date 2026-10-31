import { useEffect, useRef, useState } from "react";
import { DetectedObject, Session, api, groupProducts, imageUrl } from "./api";
import { AgentPicker, useAgents } from "./AgentPicker";
import { PhotoPane, ProductImages, ProductTitle, useLinkedRow } from "./PhotoPane";

const STAGES = [
  ["proposing", "候補を探す"],
  ["refining", "枠を調整"],
  ["reviewing", "点検"],
] as const;

export function HintsPage({ session }: { session: Session }) {
  if (session.status === "detecting") return <DetectingView session={session} />;
  if (session.status === "detect_failed") return <DetectFailedView session={session} />;
  return <HintEditor session={session} />;
}

function DetectingView({ session }: { session: Session }) {
  const stageIndex = STAGES.findIndex(([id]) => id === session.detect.stage);
  return (
    <main className="detecting">
      <img src={imageUrl(session.id)} alt="" />
      <div>
        <h1>オブジェクトを検出しています</h1>
        <p className="lead">{session.detect.message}</p>
        <div className="progress indeterminate" role="progressbar" aria-label="検出中"><span /></div>
        <ol className="stage-list">
          {STAGES.map(([id, label], i) => (
            <li key={id} className={i < stageIndex ? "done" : i === stageIndex ? "current" : ""}>{label}</li>
          ))}
        </ol>
        <p className="muted">写真によって数分かかります。この画面を閉じても処理は続きます。</p>
      </div>
    </main>
  );
}

function DetectFailedView({ session }: { session: Session }) {
  const [error, setError] = useState<string | null>(null);
  return (
    <main className="page-message">
      <h1>検出に失敗しました</h1>
      <p className="error">{session.detect.message}</p>
      {error && <p className="error">{error}</p>}
      <button type="button" className="primary" onClick={() => api.redetect(session.id).catch((e: Error) => setError(e.message))}>
        もう一度検出する
      </button>
    </main>
  );
}

function HintEditor({ session }: { session: Session }) {
  const { providers, choice, setChoice, error: agentError } = useAgents();
  const [hints, setHints] = useState<Record<string, string>>({});
  const [activeId, setActiveId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const timers = useRef(new Map<string, number>());
  const { register, jumpTo, flashId } = useLinkedRow("input[type=text]");

  useEffect(() => {
    const pending = timers.current;
    return () => pending.forEach((t) => window.clearTimeout(t));
  }, []);

  const hintOf = (o: DetectedObject) => hints[o.id] ?? o.hint;

  function changeHint(oid: string, value: string) {
    setHints((prev) => ({ ...prev, [oid]: value }));
    window.clearTimeout(timers.current.get(oid));
    timers.current.set(
      oid,
      window.setTimeout(() => {
        api.patchObject(session.id, oid, { hint: value }).catch(() => {
          // The latest text is sent again with the start request.
        });
      }, 500),
    );
  }

  async function patch(o: DetectedObject, change: { excluded?: boolean; ungroup?: boolean; leave_group?: boolean }) {
    try {
      await api.patchObject(session.id, o.id, change);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  function toggleSelected(oid: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (!next.delete(oid)) next.add(oid);
      return next;
    });
  }

  async function mergeSelected(ids: string[]) {
    setError(null);
    timers.current.forEach((t) => window.clearTimeout(t));
    try {
      await api.createGroup(session.id, ids, hints);
      setHints({}); // the server now holds every draft, merged
      setSelected(new Set());
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function start() {
    if (!choice) return;
    setBusy(true);
    setError(null);
    timers.current.forEach((t) => window.clearTimeout(t));
    try {
      await api.startSearch(session.id, choice.provider, choice.model, hints);
      window.location.hash = `#/s/${session.id}/results`;
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }

  const products = groupProducts(session.objects);
  const included = products.filter(([o]) => !o.excluded);
  const hinted = included.filter(([o]) => hintOf(o).trim()).length;
  const selectedProducts = included.filter(([o]) => selected.has(o.id));
  const selectedBoxes = new Set(selectedProducts.flat().map((o) => o.id));

  return (
    <div className="workspace">
      <main className="list-pane">
        <div className="pane-heading">
          <h1>{products.length} 件のオブジェクト</h1>
          <p className="muted">わかる範囲でヒントを入力してください。空欄のままでも検索できます。同じ製品の部品はチェックを付けてまとめられます。</p>
        </div>
        {session.detect.warnings.map((w) => <p key={w} className="warning">{w}</p>)}
        <ul className="object-list">
          {products.map((members) => {
            const o = members[0];
            const has = (id: string | null) => members.some((m) => m.id === id);
            return (
              <li
                key={o.id}
                ref={register(members.map((m) => m.id))}
                className={`object-row ${has(activeId) ? "active" : ""} ${has(flashId) ? "flash" : ""} ${o.excluded ? "excluded" : ""}`}
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
                      checked={selected.has(o.id) && !o.excluded}
                      disabled={o.excluded}
                      aria-label={`${o.label} をまとめる対象に選ぶ`}
                      onChange={() => toggleSelected(o.id)}
                    />
                    <ProductTitle members={members} />
                    <button type="button" className="text-button" onClick={() => patch(o, { excluded: !o.excluded })}>
                      {o.excluded ? "対象に戻す" : "除外"}
                    </button>
                  </div>
                  <input
                    type="text"
                    value={hintOf(o)}
                    maxLength={1000}
                    disabled={o.excluded}
                    aria-label={`${o.label} のヒント`}
                    placeholder={o.excluded ? "検索しません" : "ヒント(メーカー、サイズ、特徴など)"}
                    onChange={(e) => changeHint(o.id, e.target.value)}
                  />
                  {members.length > 1 && (
                    <p className="group-note">
                      同じ製品として 1 回で調べます
                      <button type="button" className="text-button" onClick={() => patch(o, { ungroup: true })}>
                        別々に調べる
                      </button>
                      {members.length > 2 && members.map((m) => (
                        <button
                          key={m.id}
                          type="button"
                          className="text-button"
                          aria-label={`${m.number} を外す(${m.label})`}
                          title={m.label}
                          onClick={() => patch(m, { leave_group: true })}
                        >
                          {m.number} を外す
                        </button>
                      ))}
                    </p>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      </main>

      <aside className="photo-pane">
        <PhotoPane
          session={session}
          objects={session.objects}
          activeId={activeId}
          selectedIds={selectedBoxes}
          onHover={setActiveId}
          onSelect={jumpTo}
        />
        <div className="start-panel">
          {selectedProducts.length > 0 && (
            <div className="start-row group-bar">
              <span className="muted">{selectedProducts.length} 件を選択中</span>
              <span className="group-actions">
                <button type="button" className="text-button" onClick={() => setSelected(new Set())}>選択を解除</button>
                <button
                  type="button"
                  disabled={selectedProducts.length < 2}
                  onClick={() => mergeSelected(selectedProducts.map(([o]) => o.id))}
                >
                  同じ製品としてまとめる
                </button>
              </span>
            </div>
          )}
          <AgentPicker providers={providers} choice={choice} onChange={setChoice} disabled={busy} />
          <div className="start-row">
            <span className="muted">{included.length} 件を検索(ヒント {hinted} 件)</span>
            <button type="button" className="primary" disabled={busy || !choice || included.length === 0} onClick={start}>
              {busy ? "開始しています…" : "商品検索を開始"}
            </button>
          </div>
          {(error || agentError) && <p className="error" role="alert">{error || agentError}</p>}
        </div>
      </aside>
    </div>
  );
}
