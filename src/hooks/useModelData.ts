import { useState, useEffect, useCallback, useRef } from "react";
import brotliPromise from "brotli-dec-wasm";
import type { ModelData, SearchResult, TokenEntry } from "../types";

export type EmbeddingType = "input" | "output";

const MODEL_FILES: Record<string, Partial<Record<EmbeddingType, string>>> = {
  "qwen3-30b-a3b": {
    input: "/data/qwen3-30b-a3b-input.json.br",
    output: "/data/qwen3-30b-a3b-output.json.br",
  },
  "llama-3.1-8b": {
    input: "/data/llama-3.1-8b-input.json.br",
    output: "/data/llama-3.1-8b-output.json.br",
  },
  "gemma-3-4b": {
    input: "/data/gemma-3-4b-input.json.br",
  },
};

export function availableModels(): string[] {
  return Object.keys(MODEL_FILES);
}

export function availableEmbeddingTypes(modelId: string): EmbeddingType[] {
  const files = MODEL_FILES[modelId];
  if (!files) return ["input"];
  return (Object.keys(files) as EmbeddingType[]);
}

interface UseModelDataReturn {
  data: ModelData | null;
  loading: boolean;
  error: string | null;
  search: (query: string, limit?: number) => SearchResult[];
  getToken: (id: number) => TokenEntry | undefined;
}

export function useModelData(
  modelId: string,
  embeddingType: EmbeddingType
): UseModelDataReturn {
  const [data, setData] = useState<ModelData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Index: lowercase token string → token ids, built once on load
  const searchIndex = useRef<Map<string, number[]>>(new Map());

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);
    searchIndex.current = new Map();

    const files = MODEL_FILES[modelId];
    if (!files) {
      setError(`Unknown model: ${modelId}`);
      setLoading(false);
      return;
    }
    const url = files[embeddingType];
    if (!url) {
      setError(`${modelId} does not have ${embeddingType} embeddings (tied weights)`);
      setLoading(false);
      return;
    }

    (async () => {
      try {
        const [resp, brotli] = await Promise.all([fetch(url), brotliPromise]);
        if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);

        if (cancelled) return;
        const bytes = new Uint8Array(await resp.arrayBuffer());
        // If the server set Content-Encoding: br, the browser already
        // decompressed and we have raw JSON (starts with '{'). Otherwise
        // we have raw brotli bytes that need manual decompression.
        const json =
          bytes[0] === 0x7b
            ? new TextDecoder().decode(bytes)
            : new TextDecoder().decode(brotli.decompress(bytes));
        const parsed: ModelData = JSON.parse(json);

        // Build search index
        const idx = new Map<string, number[]>();
        for (const [idStr, entry] of Object.entries(parsed.tokens)) {
          const key = entry.s.toLowerCase();
          const ids = idx.get(key);
          if (ids) {
            ids.push(Number(idStr));
          } else {
            idx.set(key, [Number(idStr)]);
          }
        }

        if (cancelled) return;
        searchIndex.current = idx;
        setData(parsed);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [modelId, embeddingType]);

  const search = useCallback(
    (query: string, limit = 50): SearchResult[] => {
      if (!data || !query) return [];
      const lower = query.toLowerCase();
      const results: SearchResult[] = [];

      // Check if query is a numeric token ID
      if (/^\d+$/.test(query)) {
        const id = Number(query);
        const entry = data.tokens[String(id)];
        if (entry) {
          results.push({ id, text: entry.s });
        }
      }

      // Search by token string
      for (const [tokenStr, tokenIds] of searchIndex.current) {
        if (results.length >= limit) break;
        if (tokenStr.includes(lower)) {
          for (const tokenId of tokenIds) {
            if (results.length >= limit) break;
            if (!results.some((r) => r.id === tokenId)) {
              results.push({ id: tokenId, text: data.tokens[String(tokenId)].s });
            }
          }
        }
      }
      return results;
    },
    [data]
  );

  const getToken = useCallback(
    (id: number): TokenEntry | undefined => {
      if (!data) return undefined;
      return data.tokens[String(id)];
    },
    [data]
  );

  return { data, loading, error, search, getToken };
}
