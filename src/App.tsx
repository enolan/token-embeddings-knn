import { useState, useEffect, useCallback, useRef } from "react";
import { ModelSelector } from "./components/ModelSelector";
import { EmbeddingToggle } from "./components/EmbeddingToggle";
import { TokenSearch } from "./components/TokenSearch";
import { NeighborResults } from "./components/NeighborResults";
import {
  useModelData,
  availableModels,
  availableEmbeddingTypes,
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
  const pendingTokenStr = useRef<string | null>(null);
  const [searchPrefill, setSearchPrefill] = useState<string | null>(null);
  const [showLegend, setShowLegend] = useState(false);
  const legendRef = useRef<HTMLDivElement>(null);

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

  // Close legend on outside click
  useEffect(() => {
    if (!showLegend) return;
    function handleClick(e: MouseEvent) {
      if (legendRef.current && !legendRef.current.contains(e.target as Node)) {
        setShowLegend(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [showLegend]);

  const navigate = useCallback((tokenId: number) => {
    const params = new URLSearchParams(window.location.search);
    params.set("token", String(tokenId));
    window.history.pushState(null, "", `${window.location.pathname}?${params}`);
    setSelectedToken(tokenId);
    setSearchPrefill(null);
  }, []);

  const handleModelChange = useCallback((id: string) => {
    if (selectedToken !== null) {
      const entry = getToken(selectedToken);
      pendingTokenStr.current = entry ? entry.s : null;
    } else {
      pendingTokenStr.current = null;
    }
    setSearchPrefill(null);
    setModelId(id);
    setSelectedToken(null);
    const types = availableEmbeddingTypes(id);
    setEmbeddingType((prev) => (types.includes(prev) ? prev : types[0]));
  }, [selectedToken, getToken]);

  // Resolve pending token string after model switch — only fires when data changes
  useEffect(() => {
    const pending = pendingTokenStr.current;
    if (!pending || !data) return;
    pendingTokenStr.current = null;
    const results = search(pending);
    const exact = results.find((r) => r.text === pending);
    if (exact) {
      navigate(exact.id);
    } else {
      setSearchPrefill(pending);
    }
  }, [data, search, navigate]);

  const token = selectedToken !== null ? getToken(selectedToken) : undefined;

  return (
    <div className="app">
      <header>
        <h1>Token Embeddings KNN Explorer</h1>
        <div className="header-controls">
          <ModelSelector value={modelId} onChange={handleModelChange} />
          <EmbeddingToggle value={embeddingType} onChange={setEmbeddingType} availableTypes={availableEmbeddingTypes(modelId)} />
          <div className="legend-wrapper" ref={legendRef}>
            <button
              className="legend-button"
              onClick={() => setShowLegend((v) => !v)}
              title="Symbol legend"
            >
              ?
            </button>
            {showLegend && (
              <div className="legend-popover">
                <span>
                  <span className="token-text">{"\u00B7"}</span> space
                </span>
                <span>
                  <span className="token-text">{"\u21B5"}</span> newline
                </span>
                <span>
                  <span className="token-text">{"\u21E5"}</span> tab
                </span>
                <span>
                  <span className="token-text">{"\u2581"}</span> word boundary
                </span>
              </div>
            )}
          </div>
        </div>
      </header>

      {loading && (
        <div className="loading-skeleton">
          <div className="skeleton-bar skeleton-search" />
          <div className="skeleton-bar skeleton-row" />
          <div className="skeleton-bar skeleton-row" />
          <div className="skeleton-bar skeleton-row" />
          <div className="skeleton-bar skeleton-row" />
          <div className="skeleton-bar skeleton-row" />
        </div>
      )}
      {error && <div className="status error">Error: {error}</div>}

      {data && (
        <>
          <TokenSearch onSelect={navigate} search={search} prefillQuery={searchPrefill} />
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

      <footer className="site-footer">
        Made by Claude and <a href="https://x.com/enolan" target="_blank" rel="noopener noreferrer">Echo Nolan</a>
      </footer>
    </div>
  );
}
