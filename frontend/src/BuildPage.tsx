import { useEffect, useRef, useState } from "react";
import { DetectedObject, GlbsmithModels, GlbsmithStatus, ProductBuild, ProductSize, Session, api, groupProducts, modelUrl, photoUrl, renderUrl } from "./api";
import { AgentPicker, useAgents } from "./AgentPicker";
import { PhotoPane, ProductImages, ProductTitle, useLinkedRow } from "./PhotoPane";

const SOURCE_LABEL = { official: "公式", retail: "販売ページ", other: "その他の資料", estimate: "写真から推定", manual: "手入力" };
const CONFIDENCE_LABEL = { high: "確信度 高", medium: "確信度 中", low: "確信度 低" };
const BUILD_LABEL: Record<ProductBuild["status"], string> = {
  preparing: "準備中",
  queued: "glbsmith の順番待ち",
  running: "作成中",
  done: "完成",
  failed: "失敗",
  cancelled: "取り消し",
};
const VIEW_LABEL: Record<string, string> = {
  front: "正面", back: "背面", left: "左側面", right: "右側面", top: "上", bottom: "下",
  angled: "斜め", detail: "拡大", in_use: "使用場面",
};
const AXES = [["width", "幅"], ["depth", "奥行き"], ["height", "高さ"]] as const;
type Axis = (typeof AXES)[number][0];
type Drafts = Record<string, Partial<Record<Axis, string>>>;

const isSizeBusy = (o: DetectedObject) => o.size?.status === "queued" || o.size?.status === "running";
const isBuildBusy = (o: DetectedObject) => ["preparing", "queued", "running"].includes(o.build?.status ?? "");

export function BuildPage({ session }: { session: Session }) {
  const { providers, choice, setChoice, error: agentError } = useAgents();
  const glb = useGlbsmith();
  const [activeId, setActiveId] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [drafts, setDrafts] = useState<Drafts>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timers = useRef(new Map<string, number>());
  const { register, jumpTo, flashId } = useLinkedRow("input[type=number]");

  useEffect(() => {
    const pending = timers.current;
    return () => pending.forEach((t) => window.clearTimeout(t));
  }, []);

  const targets = session.objects.filter((o) => !o.excluded);
  const products = groupProducts(targets);
  const confirmed = products.filter(([o]) => o.status === "confirmed");
  const unconfirmed = products.length - confirmed.length;
  const chosen = confirmed.filter(([o]) => selected.has(o.id)).map(([o]) => o);
  const forSize = chosen.filter((o) => !isSizeBusy(o) && !isBuildBusy(o));
  const value = (o: DetectedObject, axis: Axis) => drafts[o.id]?.[axis] ?? (o.size?.[axis] != null ? String(o.size[axis]) : "");
  const hasSize = (o: DetectedObject) => AXES.every(([axis]) => Number(value(o, axis)) > 0);
  const forBuild = chosen.filter((o) => hasSize(o) && !isSizeBusy(o) && !isBuildBusy(o));

  function savePatch(o: DetectedObject, axis: Axis, text: string) {
    const number = Number(text);
    if (number > 0) {
      api.patchSize(session.id, o.id, { [axis]: number }).catch((e: Error) => setError(e.message));
    }
  }

  function changeSize(o: DetectedObject, axis: Axis, text: string) {
    setDrafts((prev) => ({ ...prev, [o.id]: { ...prev[o.id], [axis]: text } }));
    const key = `${o.id}:${axis}`;
    window.clearTimeout(timers.current.get(key));
    timers.current.set(key, window.setTimeout(() => {
      timers.current.delete(key);
      savePatch(o, axis, text);
    }, 500));
  }

  /** Waits for sizes still being typed, so a build always uses what the row shows. */
  async function flushDrafts(objects: DetectedObject[]) {
    const writes: Promise<unknown>[] = [];
    for (const o of objects) {
      const patch: Partial<Record<Axis, number>> = {};
      for (const [axis] of AXES) {
        const key = `${o.id}:${axis}`;
        if (timers.current.has(key)) {
          window.clearTimeout(timers.current.get(key));
          timers.current.delete(key);
          patch[axis] = Number(drafts[o.id]?.[axis]);
        }
      }
      if (Object.keys(patch).length) writes.push(api.patchSize(session.id, o.id, patch));
    }
    await Promise.all(writes);
  }

  async function run(action: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await action();
      setSelected(new Set());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function toggle(oid: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (!next.delete(oid)) next.add(oid);
      return next;
    });
  }

  const startSizes = () => run(async () => {
    // A new lookup replaces what the row shows, so unsaved edits there go too.
    setDrafts((prev) => Object.fromEntries(Object.entries(prev).filter(([id]) => !forSize.some((o) => o.id === id))));
    forSize.forEach((o) => AXES.forEach(([axis]) => {
      window.clearTimeout(timers.current.get(`${o.id}:${axis}`));
      timers.current.delete(`${o.id}:${axis}`);
    }));
    await api.startSizes(session.id, choice!.provider, choice!.model, forSize.map((o) => o.id));
  });

  const startBuilds = () => run(async () => {
    await flushDrafts(forBuild);
    await api.startBuilds(session.id, glb.agent!, glb.model, forBuild.map((o) => o.id));
  });

  const allIds = confirmed.map(([o]) => o.id);
  const missingSize = confirmed.filter(([o]) => !hasSize(o) && !isSizeBusy(o) && !isBuildBusy(o)).map(([o]) => o.id);

  return (
    <div className="workspace build-page">
      <main className="list-pane">
        <div className="pane-heading">
          <h1>サイズ取得と 3D 作成</h1>
          <p className="muted">
            確定済みの {confirmed.length} 製品が対象です。
            {unconfirmed > 0 && <>未確定の {unconfirmed} 製品は<a href={`#/s/${session.id}/results`}>結果確認</a>で判定すると追加されます。</>}
          </p>
        </div>
        <div className="select-all">
          <button type="button" className="text-button" onClick={() => setSelected(new Set(missingSize))}>サイズ未取得を選択</button>
          <button type="button" className="text-button" onClick={() => setSelected(new Set(allIds))}>すべて選択</button>
          <button type="button" className="text-button" onClick={() => setSelected(new Set())}>選択を解除</button>
        </div>
        <ul className="object-list">
          {confirmed.map((members) => {
            const o = members[0];
            const has = (id: string | null) => members.some((m) => m.id === id);
            return (
              <li
                key={o.id}
                ref={register(members.map((m) => m.id))}
                className={`object-row ${has(activeId) ? "active" : ""} ${has(flashId) ? "flash" : ""}`}
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
                      disabled={isBuildBusy(o)}
                      aria-label={`${o.confirmed_name} を選ぶ`}
                      onChange={() => toggle(o.id)}
                    />
                    <ProductTitle members={members} />
                  </div>
                  <p className="product-name confirmed">{o.confirmed_name}</p>
                  <SizeEditor
                    session={session}
                    obj={o}
                    grouped={members.length > 1}
                    value={(axis) => value(o, axis)}
                    onChange={(axis, text) => changeSize(o, axis, text)}
                    onError={setError}
                    locked={isBuildBusy(o)}
                  />
                  {o.build && <BuildStatus session={session} obj={o} build={o.build} onError={setError} />}
                </div>
              </li>
            );
          })}
        </ul>
        {confirmed.length === 0 && <p className="muted">まだ確定した製品がありません。</p>}
      </main>
      <aside className="photo-pane">
        <PhotoPane
          session={session}
          objects={confirmed.flat()}
          activeId={activeId}
          selectedIds={new Set(chosen.flatMap((o) => confirmed.find(([lead]) => lead.id === o.id) ?? []).map((m) => m.id))}
          onHover={setActiveId}
          onSelect={jumpTo}
        />
        <div className="start-panel">
          <section className="panel-section">
            <h2>サイズを調べる</h2>
            <AgentPicker providers={providers} choice={choice} onChange={setChoice} disabled={busy} />
            <div className="start-row">
              <span className="muted">{forSize.length} 製品を選択中</span>
              <button type="button" disabled={busy || !choice || forSize.length === 0} onClick={startSizes}>
                サイズを調べる
              </button>
            </div>
          </section>
          <section className="panel-section">
            <h2>3D モデルを作る(glbsmith)</h2>
            <GlbsmithPicker glb={glb} disabled={busy} />
            <div className="start-row">
              <span className="muted">
                {forBuild.length} 製品を作成
                {glb.status?.settings && `(同時 ${glb.status.settings.concurrency} 件ずつ)`}
              </span>
              <button
                type="button"
                className="primary"
                disabled={busy || !glb.status?.available || !glb.agent || forBuild.length === 0}
                onClick={startBuilds}
              >
                3D モデルを作成
              </button>
            </div>
            <p className="muted">1 製品あたり数十万トークンを使うことがあります。</p>
          </section>
          {(error || agentError) && <p className="error" role="alert">{error || agentError}</p>}
        </div>
      </aside>
    </div>
  );
}

function SizeEditor({
  session,
  obj,
  grouped,
  value,
  onChange,
  onError,
  locked,
}: {
  session: Session;
  obj: DetectedObject;
  grouped: boolean;
  value: (axis: Axis) => string;
  onChange: (axis: Axis, text: string) => void;
  onError: (message: string) => void;
  locked: boolean;
}) {
  const size: ProductSize | undefined = obj.size;
  if (size?.status === "queued") return <p className="muted">サイズ取得の順番待ち</p>;
  if (size?.status === "running") {
    return (
      <div>
        <div className="progress indeterminate" role="progressbar" aria-label="サイズを調査中"><span /></div>
        <p className="activity">{size.activity}</p>
      </div>
    );
  }
  return (
    <>
      {size?.status === "failed" && <p className="error">サイズを取得できませんでした: {size.error}</p>}
      <div className="size-fields">
        {AXES.map(([axis, label]) => (
          <label key={axis}>
            {label}(cm)
            <input
              type="number"
              min="0.1"
              max="2000"
              step="0.1"
              value={value(axis)}
              disabled={locked}
              onChange={(e) => onChange(axis, e.target.value)}
            />
          </label>
        ))}
        {grouped && (
          <label>
            グループの作り方
            <select
              className="kind-select"
              value={size?.kind === "identical" ? "identical" : "parts"}
              disabled={locked}
              onChange={(e) => {
                api.patchSize(session.id, obj.id, { kind: e.target.value as "parts" | "identical" }).catch((err: Error) => onError(err.message));
              }}
            >
              <option value="parts">部品を合わせて 1 つ</option>
              <option value="identical">同じ物が複数(1 つ分を作る)</option>
            </select>
          </label>
        )}
      </div>
      {size?.status === "done" && (
        <>
          <p className="size-meta">
            {size.edited ? <span>手で修正済み</span> : <span>{SOURCE_LABEL[size.source_kind ?? "manual"]}</span>}
            {size.confidence && !size.edited && <span>{CONFIDENCE_LABEL[size.confidence]}</span>}
            <SourceLink size={size} productUrl={obj.result?.source_urls?.[0]} />
          </p>
          {size.source_url && size.source_quotes?.length ? (
            <p className="size-note">
              ページの記載: {size.source_quotes.map((q) => `「${q}」`).join(" ")}(色が付かないときは、この文字をページ内検索)
            </p>
          ) : null}
        </>
      )}
      {size?.note && <p className="size-note">{size.note}</p>}
      {size?.photo_note && <p className="size-note">{size.photo_note}</p>}
      {size?.photos && size.photos.length > 0 && (
        <div className="photo-picker">
          <p className="size-meta">
            製品写真 {size.photos.filter((p) => p.selected).length} / {size.photos.length} 枚を 3D に使う(クリックで切替)
          </p>
          <ul>
            {size.photos.map((p) => (
              <li key={p.id}>
                <button
                  type="button"
                  className={p.selected ? "selected" : ""}
                  aria-pressed={p.selected}
                  aria-label={`${VIEW_LABEL[p.view] ?? p.view}の写真を${p.selected ? "使わない" : "使う"}`}
                  disabled={locked}
                  onClick={() => api.patchPhoto(session.id, obj.id, p.id, !p.selected).catch((e: Error) => onError(e.message))}
                >
                  <img src={photoUrl(session.id, obj.id, p.id)} alt="" loading="lazy" />
                  <span>{VIEW_LABEL[p.view] ?? p.view}</span>
                </button>
                {p.page_url && <a href={p.page_url} target="_blank" rel="noreferrer noopener">{hostOf(p.page_url)}</a>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </>
  );
}

/** Text fragments scroll to and highlight the quote; a browser or page that cannot simply opens the page. */
function sourceHref(url: string, quotes: string[] = []): string {
  if (!quotes.length || url.includes(":~:")) return url;
  const directives = quotes.map((q) => `text=${encodeURIComponent(q).replace(/-/g, "%2D")}`).join("&");
  return `${url}${url.includes("#") ? ":~:" : "#:~:"}${directives}`;
}

function SourceLink({ size, productUrl }: { size: ProductSize; productUrl?: string }) {
  const link = (href: string, label: string) => <a href={href} target="_blank" rel="noreferrer noopener">{label}</a>;
  if (!size.source_url) return productUrl ? <>{link(productUrl, `製品の参考ページ(${hostOf(productUrl)})`)}<span>寸法の記載は確認できず</span></> : null;
  const host = hostOf(size.source_url);
  return link(sourceHref(size.source_url, size.source_quotes), size.source_quotes?.length ? `寸法の記載を開く(${host})` : `寸法の出典を開く(${host})`);
}

function BuildStatus({ session, obj, build, onError }: { session: Session; obj: DetectedObject; build: ProductBuild; onError: (message: string) => void }) {
  const active = ["preparing", "queued", "running"].includes(build.status);
  return (
    <div className="build-box">
      <p className="size-meta">
        <strong>3D: {BUILD_LABEL[build.status]}</strong>
        <span>{build.width} × {build.depth} × {build.height} cm</span>
        {build.product_photos != null && <span>製品写真 {build.product_photos} 枚</span>}
        {build.seconds != null && <span>{formatSeconds(build.seconds)}</span>}
        {build.tokens && <span>{Math.round(build.tokens.total / 10000).toLocaleString()} 万トークン</span>}
        {active && (
          <button type="button" className="text-button" onClick={() => api.cancelBuild(session.id, obj.id).catch((e: Error) => onError(e.message))}>
            取り消す
          </button>
        )}
      </p>
      {active && <div className="progress indeterminate" role="progressbar" aria-label="3D モデルを作成中"><span /></div>}
      {build.error && <p className="error">{build.error}</p>}
      {build.has_model && (
        <div className="build-result">
          {build.has_render ? <img src={renderUrl(session.id, obj.id)} alt={`${obj.confirmed_name} の 3D モデル`} loading="lazy" /> : <span />}
          <div>
            <a href={modelUrl(session.id, obj.id)} download>GLB をダウンロード</a>
            {build.warnings.length > 0 && (
              <ul className="build-warnings">{build.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function useGlbsmith() {
  const [status, setStatus] = useState<GlbsmithStatus | null>(null);
  const [agent, setAgent] = useState<string | null>(null);
  const [models, setModels] = useState<GlbsmithModels | null>(null);
  const [model, setModel] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    const load = () => api.glbsmith().then((next) => {
      if (!live) return;
      setStatus(next);
      setAgent((current) => current ?? next.agents?.find((a) => a.available)?.id ?? null);
    }, () => live && setStatus({ available: false, url: "" }));
    load();
    const timer = window.setInterval(load, 10000);
    return () => {
      live = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!agent) return;
    setModels(null);
    setModel(null);
    let live = true;
    api.glbsmithModels(agent).then((next) => {
      if (live) setModels(next);
    }, () => undefined);
    return () => {
      live = false;
    };
  }, [agent]);

  return { status, agent, setAgent, models, model, setModel };
}

function GlbsmithPicker({ glb, disabled }: { glb: ReturnType<typeof useGlbsmith>; disabled: boolean }) {
  const { status, agent, setAgent, models, model, setModel } = glb;
  if (!status) return <p className="muted">glbsmith を確認中…</p>;
  if (!status.available) {
    return <p className="warning">glbsmith に接続できません({status.url})。glbsmith を起動すると自動で再接続します。</p>;
  }
  const agents = status.agents ?? [];
  const defaultLabel = models?.options.find((m) => m.id === models.default)?.label;
  return (
    <div className="agent-picker">
      <div className="segmented" role="radiogroup" aria-label="3D 作成のエージェント">
        {agents.map((a) => (
          <button
            key={a.id}
            type="button"
            role="radio"
            aria-checked={a.id === agent}
            className={a.id === agent ? "active" : ""}
            disabled={disabled || !a.available}
            onClick={() => setAgent(a.id)}
          >
            {a.label}
          </button>
        ))}
      </div>
      <label className="model-select">
        <span>モデル</span>
        <select value={model ?? ""} disabled={disabled || !models} onChange={(e) => setModel(e.target.value || null)}>
          <option value="">既定{defaultLabel && `(${defaultLabel})`}</option>
          {models?.options.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
        </select>
      </label>
    </div>
  );
}

function formatSeconds(total: number): string {
  return total < 60 ? `${total} 秒` : `${Math.floor(total / 60)} 分 ${total % 60} 秒`;
}

function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}
