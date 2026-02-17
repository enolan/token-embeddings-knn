import { useState, useEffect, useCallback } from "react";
import { ModelSelector } from "./components/ModelSelector";
import { EmbeddingToggle } from "./components/EmbeddingToggle";
import { TokenSearch } from "./components/TokenSearch";
import { NeighborResults } from "./components/NeighborResults";
import {
  useModelData,
  availableModels,
  type EmbeddingType,
} from "./hooks/useModelData";
import "./App.css";

function parseParams(): {
  model: string;
  embedding: EmbeddingType;
  token: number | null;
} {
  const params = new URLSearchParams(window.location.search);
  const model = params.get("model") || availableModels()[0];
  const embRaw = params.get("embedding");
  const embedding: EmbeddingType =
    embRaw === "input" || embRaw === "output" ? embRaw : "input";
  const tokenStr = params.get("token");
  const token = tokenStr !== null ? Number(tokenStr) : null;
  return {
    model,
    embedding,
    token: token !== null && !isNaN(token) ? token : null,
  };
}

export function App() {
  const initial = parseParams();
  const [modelId, setModelId] = useState(initial.model);
  const [embeddingType, setEmbeddingType] = useState<EmbeddingType>(
    initial.embedding
  );
  const [selectedToken, setSelectedToken] = useState<number | null>(
    initial.token
  );
  const { data, loading, error, search, getToken } = useModelData(
    modelId,
    embeddingType
  );

  // Sync URL with state
  useEffect(() => {
    const params = new URLSearchParams();
    params.set("model", modelId);
    params.set("embedding", embeddingType);
    if (selectedToken !== null) params.set("token", String(selectedToken));
    const url = `${window.location.pathname}?${params}`;
    window.history.replaceState(null, "", url);
  }, [modelId, embeddingType, selectedToken]);

  // Handle browser back/forward
  useEffect(() => {
    function onPopState() {
      const p = parseParams();
      setModelId(p.model);
      setEmbeddingType(p.embedding);
      setSelectedToken(p.token);
    }
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = useCallback((tokenId: number) => {
    const params = new URLSearchParams(window.location.search);
    params.set("token", String(tokenId));
    window.history.pushState(null, "", `${window.location.pathname}?${params}`);
    setSelectedToken(tokenId);
  }, []);

  const handleModelChange = useCallback((id: string) => {
    setModelId(id);
    setSelectedToken(null);
  }, []);

  const token = selectedToken !== null ? getToken(selectedToken) : undefined;

  return (
    <div className="app">
      <header>
        <h1>Token Embeddings KNN Explorer</h1>
        <div className="header-controls">
          <ModelSelector value={modelId} onChange={handleModelChange} />
          <EmbeddingToggle value={embeddingType} onChange={setEmbeddingType} />
        </div>
      </header>

      <div className="legend">
        <span><span className="token-text">{"\u00B7"}</span> space</span>
        <span><span className="token-text">{"\u21B5"}</span> newline</span>
        <span><span className="token-text">{"\u21E5"}</span> tab</span>
        <span><span className="token-text">{"\u2581"}</span> word boundary (tokenizer)</span>
      </div>

      {loading && <div className="status">Loading model data...</div>}
      {error && <div className="status error">Error: {error}</div>}

      {data && (
        <>
          <TokenSearch onSelect={navigate} search={search} />
          {token && selectedToken !== null && (
            <NeighborResults
              tokenId={selectedToken}
              token={token}
              getToken={getToken}
              onNavigate={navigate}
            />
          )}
        </>
      )}
    </div>
  );
}
