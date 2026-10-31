export type ObjectStatus = "idle" | "queued" | "running" | "done" | "rejected" | "confirmed" | "failed";

export interface SearchResult {
  brand: string;
  product_name: string;
  model_number: string;
  confidence: "high" | "medium" | "low";
  reasoning: string;
  source_urls: string[];
}

export interface ProductPhoto {
  id: string;
  url: string;
  page_url: string;
  width: number;
  height: number;
  view: string;
  product_only: boolean;
  selected: boolean;
}

export interface ProductSize {
  status: "queued" | "running" | "done" | "failed";
  width?: number;
  depth?: number;
  height?: number;
  source_url?: string;
  source_quotes?: string[];
  source_kind?: "official" | "retail" | "other" | "estimate" | "manual";
  kind?: "single" | "parts" | "identical";
  confidence?: "high" | "medium" | "low" | null;
  note?: string;
  activity?: string | null;
  started_at?: number | null;
  error?: string | null;
  edited?: boolean;
  photos?: ProductPhoto[];
  photo_note?: string | null;
}

export interface ProductBuild {
  status: "preparing" | "queued" | "running" | "done" | "failed" | "cancelled";
  job_id: string | null;
  agent: string;
  model: string | null;
  width: number;
  depth: number;
  height: number;
  kind: string;
  requested_at: number;
  error: string | null;
  message: string | null;
  seconds: number | null;
  tokens: { total: number } | null;
  warnings: string[];
  has_model: boolean;
  has_render: boolean;
  product_photos?: number;
}

export interface DetectedObject {
  id: string;
  number: number;
  label: string;
  category: string;
  bbox: [number, number, number, number];
  group?: string | null;
  hint: string;
  excluded: boolean;
  status: ObjectStatus;
  activity: string | null;
  started_at: number | null;
  queued_at: number | null;
  result: SearchResult | null;
  history: { result: SearchResult; feedback: string }[];
  confirmed_name: string | null;
  confirmed_by: "agent" | "user" | null;
  error: string | null;
  size?: ProductSize;
  build?: ProductBuild;
}

export interface Session {
  id: string;
  created_at: number;
  status: "detecting" | "ready" | "detect_failed";
  image: { filename: string; width: number; height: number };
  detect: {
    provider: string;
    model: string;
    stage: string;
    message: string;
    warnings: string[];
    started_at: number;
  };
  search: { provider: string; model: string; started_at: number } | null;
  objects: DetectedObject[];
}

export interface GlbsmithStatus {
  available: boolean;
  url: string;
  error?: string;
  blender?: boolean;
  sandbox?: boolean;
  agents?: { id: string; label: string; available: boolean; billing: string }[];
  settings?: { concurrency: number; max: number; running: number; queued: number };
}

export interface GlbsmithModels {
  default: string | null;
  options: { id: string; label: string }[];
}

export interface SessionSummary {
  id: string;
  filename: string;
  created_at: number;
  status: Session["status"];
  object_count: number;
  search_started: boolean;
}

export interface Provider {
  id: string;
  label: string;
  available: boolean;
  default_model: string | null;
  models: { id: string; label: string }[];
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : `リクエストに失敗しました (${res.status})`;
    throw new Error(detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

function json(method: string, body: unknown): RequestInit {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

export const api = {
  agents: () => request<Provider[]>("/api/agents"),
  sessions: () => request<SessionSummary[]>("/api/sessions"),
  session: (sid: string) => request<Session>(`/api/sessions/${sid}`),
  deleteSession: (sid: string) => request<void>(`/api/sessions/${sid}`, { method: "DELETE" }),
  redetect: (sid: string) => request<Session>(`/api/sessions/${sid}/redetect`, { method: "POST" }),
  create: (file: File, provider: string, model: string) => {
    const form = new FormData();
    form.append("file", file);
    form.append("provider", provider);
    form.append("model", model);
    return request<Session>("/api/sessions", { method: "POST", body: form });
  },
  patchObject: (sid: string, oid: string, patch: { hint?: string; excluded?: boolean; ungroup?: boolean; leave_group?: boolean }) =>
    request<DetectedObject>(`/api/sessions/${sid}/objects/${oid}`, json("PATCH", patch)),
  createGroup: (sid: string, objectIds: string[], hints: Record<string, string>) =>
    request<Session>(`/api/sessions/${sid}/groups`, json("POST", { object_ids: objectIds, hints })),
  startSizes: (sid: string, provider: string, model: string, objectIds: string[]) =>
    request<Session>(`/api/sessions/${sid}/sizes`, json("POST", { provider, model, object_ids: objectIds })),
  patchSize: (sid: string, oid: string, patch: { width?: number; depth?: number; height?: number; kind?: "parts" | "identical" }) =>
    request<DetectedObject>(`/api/sessions/${sid}/objects/${oid}/size`, json("PATCH", patch)),
  patchPhoto: (sid: string, oid: string, photoId: string, selected: boolean) =>
    request<DetectedObject>(`/api/sessions/${sid}/objects/${oid}/photos/${photoId}`, json("PATCH", { selected })),
  glbsmith: () => request<GlbsmithStatus>("/api/glbsmith"),
  glbsmithModels: (agent: string) => request<GlbsmithModels>(`/api/glbsmith/agents/${agent}/models`),
  startBuilds: (sid: string, agent: string, model: string | null, objectIds: string[]) =>
    request<Session>(`/api/sessions/${sid}/builds`, json("POST", { agent, model, object_ids: objectIds })),
  cancelBuild: (sid: string, oid: string) =>
    request<DetectedObject>(`/api/sessions/${sid}/objects/${oid}/build/cancel`, { method: "POST" }),
  startSearch: (sid: string, provider: string, model: string, hints: Record<string, string>) =>
    request<Session>(`/api/sessions/${sid}/search`, json("POST", { provider, model, hints })),
  verdict: (sid: string, oid: string, verdict: "correct" | "wrong" | "manual" | "reopen", name?: string) =>
    request<DetectedObject>(`/api/sessions/${sid}/objects/${oid}/verdict`, json("POST", { verdict, name })),
  retry: (sid: string, oid: string, feedback: string) =>
    request<DetectedObject>(`/api/sessions/${sid}/objects/${oid}/retry`, json("POST", { feedback })),
};

/** One list row per product: left/right parts of a product share a row, in detection order. */
export function groupProducts(objects: DetectedObject[]): DetectedObject[][] {
  const groups = new Map<string, DetectedObject[]>();
  const products: DetectedObject[][] = [];
  for (const o of objects) {
    const members = o.group ? groups.get(o.group) : undefined;
    if (members) {
      members.push(o);
      continue;
    }
    const product = [o];
    products.push(product);
    if (o.group) groups.set(o.group, product);
  }
  return products;
}

export const imageUrl = (sid: string) => `/api/sessions/${sid}/image`;
export const photoUrl = (sid: string, oid: string, photoId: string) => `/api/sessions/${sid}/objects/${oid}/photos/${photoId}.jpg`;
export const modelUrl = (sid: string, oid: string) => `/api/sessions/${sid}/objects/${oid}/model.glb`;
export const renderUrl = (sid: string, oid: string) => `/api/sessions/${sid}/objects/${oid}/render.png`;
export const cropUrl = (sid: string, oid: string) => `/api/sessions/${sid}/objects/${oid}/crop`;
