import { useState, useEffect, useCallback, useRef } from "react";
import type { ModelData, SearchResult, TokenEntry } from "../types";

export type EmbeddingType = "input" | "output";

const MODEL_FILES: Record<string, Record<EmbeddingType, string>> = {
  "qwen3-30b-a3b": {
    input: "/data/qwen3-30b-a3b-input.json.gz",
    output: "/data/qwen3-30b-a3b-output.json.gz",
  },
};

export function availableModels(): string[] {
  return Object.keys(MODEL_FILES);
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
  // Index: lowercase token string → token id, built once on load
  const searchIndex = useRef<Map<string, number>>(new Map());

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

    (async () => {
      try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);

        // Try to decompress gzip. The browser may do it automatically via
        // Content-Encoding, or we may need to do it manually.
        let json: string;
        const contentType = resp.headers.get("content-type") || "";
        if (
          contentType.includes("application/json") ||
          contentType.includes("text/")
        ) {
          // Browser already decompressed it
          json = await resp.text();
        } else {
          // Manually decompress
          const buf = await resp.arrayBuffer();
          const ds = new DecompressionStream("gzip");
          const writer = ds.writable.getWriter();
          writer.write(new Uint8Array(buf));
          writer.close();
          const reader = ds.readable.getReader();
          const chunks: Uint8Array[] = [];
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
          }
          const totalLength = chunks.reduce((s, c) => s + c.length, 0);
          const merged = new Uint8Array(totalLength);
          let offset = 0;
          for (const chunk of chunks) {
            merged.set(chunk, offset);
            offset += chunk.length;
          }
          json = new TextDecoder().decode(merged);
        }

        if (cancelled) return;
        const parsed: ModelData = JSON.parse(json);

        // Build search index
        const idx = new Map<string, number>();
        for (const [idStr, entry] of Object.entries(parsed.tokens)) {
          idx.set(entry.s.toLowerCase(), Number(idStr));
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
      for (const [tokenStr, tokenId] of searchIndex.current) {
        if (results.length >= limit) break;
        if (tokenStr.includes(lower)) {
          // Avoid duplicate if we already added via numeric match
          if (!results.some((r) => r.id === tokenId)) {
            results.push({ id: tokenId, text: data.tokens[String(tokenId)].s });
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
