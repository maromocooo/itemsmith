import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Session, api } from "./api";
import { UploadPage } from "./UploadPage";
import { HintsPage } from "./HintsPage";
import { ResultsPage } from "./ResultsPage";
import { BuildPage } from "./BuildPage";
import "./style.css";

function useHash(): string {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const onChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return hash;
}

/** Live session state: SSE pushes the whole document on every change. */
function useSession(sid: string | null) {
  const [session, setSession] = useState<Session | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setSession(null);
    setError(null);
    if (!sid) return;
    api.session(sid).then(setSession, (e: Error) => setError(e.message));
    const source = new EventSource(`/api/sessions/${sid}/events`);
    source.addEventListener("session", (e) => setSession(JSON.parse((e as MessageEvent).data)));
    return () => source.close();
  }, [sid]);
  return { session, error };
}

function App() {
  const hash = useHash();
  const match = hash.match(/^#\/s\/([0-9a-f]{32})(\/results|\/build)?$/);
  const sid = match?.[1] ?? null;
  const page = match?.[2] ?? "";
  const { session, error } = useSession(sid);

  useEffect(() => {
    if (!session) return;
    const started = session.search !== null;
    if (started !== Boolean(page)) {
      window.location.replace(`#/s/${session.id}${started ? "/results" : ""}`);
    }
  }, [session, page]);

  let body;
  if (!sid) body = <UploadPage />;
  else if (error) body = <div className="page-message">{error}</div>;
  else if (!session) body = <div className="page-message">読み込み中…</div>;
  else if (session.search && page === "/build") body = <BuildPage session={session} />;
  else if (session.search) body = <ResultsPage session={session} />;
  else body = <HintsPage session={session} />;

  return (
    <>
      <header className="topbar">
        <a className="brand" href="#/">itemsmith</a>
        {session && <span className="topbar-file">{session.image.filename}</span>}
        {session && (
          <ol className="steps" aria-label="進行状況">
            <li className={session.search ? "" : "current"}>ヒント入力</li>
            <li className={session.search && page !== "/build" ? "current" : ""}>
              {session.search ? <a href={`#/s/${session.id}/results`}>結果確認</a> : "結果確認"}
            </li>
            <li className={page === "/build" ? "current" : ""}>
              {session.search ? <a href={`#/s/${session.id}/build`}>サイズ・3D</a> : "サイズ・3D"}
            </li>
          </ol>
        )}
      </header>
      {body}
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
