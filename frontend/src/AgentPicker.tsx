import { useEffect, useState } from "react";
import { Provider, api } from "./api";

export interface AgentChoice {
  provider: string;
  model: string;
}

export function useAgents() {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [choice, setChoice] = useState<AgentChoice | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.agents().then((list) => {
      setProviders(list);
      const first = list.find((p) => p.available && p.default_model);
      if (first) setChoice({ provider: first.id, model: first.default_model! });
      else setError("Claude Code / Codex の CLI が見つかりません");
    }, (e: Error) => setError(e.message));
  }, []);
  return { providers, choice, setChoice, error };
}

export function AgentPicker({
  providers,
  choice,
  onChange,
  disabled,
}: {
  providers: Provider[];
  choice: AgentChoice | null;
  onChange: (choice: AgentChoice) => void;
  disabled?: boolean;
}) {
  const current = providers.find((p) => p.id === choice?.provider);
  return (
    <div className="agent-picker">
      <div className="segmented" role="radiogroup" aria-label="エージェント">
        {providers.map((p) => (
          <button
            key={p.id}
            type="button"
            role="radio"
            aria-checked={p.id === choice?.provider}
            className={p.id === choice?.provider ? "active" : ""}
            disabled={disabled || !p.available || !p.default_model}
            title={p.available ? undefined : "CLI が見つかりません"}
            onClick={() => onChange({ provider: p.id, model: p.default_model! })}
          >
            {p.label}
          </button>
        ))}
      </div>
      <label className="model-select">
        <span>モデル</span>
        <select
          value={choice?.model ?? ""}
          disabled={disabled || !current}
          onChange={(e) => onChange({ provider: choice!.provider, model: e.target.value })}
        >
          {current?.models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
              {m.id === current.default_model ? "(既定)" : ""}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
