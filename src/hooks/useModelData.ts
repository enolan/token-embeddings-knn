import { useState, useEffect, useCallback, useRef } from "react";
import brotliPromise from "brotli-dec-wasm";
import type { ShardManifest, KnnShard, SearchResult, TokenEntry } from "../types";

export type EmbeddingType = "input" | "output";

/** URL prefix for each model/embedding pair (without file suffix) */
const MODEL_PREFIXES: Record<string, Partial<Record<EmbeddingType, string>>> = {
  "qwen3-30b-a3b": {
    input: "/data/qwen3-30b-a3b-input",
    output: "/data/qwen3-30b-a3b-output",
  },
  "llama-3.1-8b": {
    input: "/data/llama-3.1-8b-input",
    output: "/data/llama-3.1-8b-output",
  },
  "gemma-3-4b": {
    input: "/data/gemma-3-4b-input",
  },
};

export function availableModels(): string[] {
  return Object.keys(MODEL_PREFIXES);
}

export function availableEmbeddingTypes(modelId: string): EmbeddingType[] {
  const files = MODEL_PREFIXES[modelId];
  if (!files) return ["input"];
  return Object.keys(files) as EmbeddingType[];
}

interface UseModelDataReturn {
  loading: boolean;
  error: string | null;
  search: (query: string, limit?: number) => SearchResult[];
  getToken: (id: number) => TokenEntry | undefined;
  neighborsLoading: boolean;
}

async function fetchAndDecompress<T>(url: string): Promise<T> {
  const [resp, brotli] = await Promise.all([fetch(url), brotliPromise]);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);
  const bytes = new Uint8Array(await resp.arrayBuffer());
  // If the server set Content-Encoding: br, the browser already
  // decompressed and we have raw JSON (starts with '[' or '{'). Otherwise
  // we have raw brotli bytes that need manual decompression.
  const firstByte = bytes[0];
  const json =
    firstByte === 0x7b || firstByte === 0x5b
      ? new TextDecoder().decode(bytes)
      : new TextDecoder().decode(brotli.decompress(bytes));
  return JSON.parse(json);
}

export function useModelData(
  modelId: string,
  embeddingType: EmbeddingType,
  selectedToken: number | null
): UseModelDataReturn {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tokenStrings, setTokenStrings] = useState<string[] | null>(null);
  const [manifest, setManifest] = useState<ShardManifest | null>(null);
  const [neighborsLoading, setNeighborsLoading] = useState(false);
  // Incremented when a shard loads, to trigger re-renders for getToken/search consumers
  const [, setShardVersion] = useState(0);

  const searchIndex = useRef<Map<string, number[]>>(new Map());
  const shardCache = useRef<Map<number, KnnShard>>(new Map());
  const prefixRef = useRef<string | null>(null);

  // Phase 1: Load manifest + tokens on model/embedding switch
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setTokenStrings(null);
    setManifest(null);
    setNeighborsLoading(false);
    searchIndex.current = new Map();
    shardCache.current = new Map();
    setShardVersion(0);

    const files = MODEL_PREFIXES[modelId];
    if (!files) {
      setError(`Unknown model: ${modelId}`);
      setLoading(false);
      return;
    }
    const prefix = files[embeddingType];
    if (!prefix) {
      setError(`${modelId} does not have ${embeddingType} embeddings (tied weights)`);
      setLoading(false);
      return;
    }
    prefixRef.current = prefix;

    (async () => {
      try {
        // Fetch manifest first (small, uncompressed)
        const manifestData = await fetchAndDecompress<ShardManifest>(
          `${prefix}-manifest.json`
        );
        if (cancelled) return;

        // Fetch tokens file
        const tokens = await fetchAndDecompress<string[]>(
          `${prefix}-tokens.json.br`
        );
        if (cancelled) return;

        // Build search index
        const idx = new Map<string, number[]>();
        for (let id = 0; id < tokens.length; id++) {
          const key = tokens[id].toLowerCase();
          const ids = idx.get(key);
          if (ids) {
            ids.push(id);
          } else {
            idx.set(key, [id]);
          }
        }

        if (cancelled) return;
        searchIndex.current = idx;
        setManifest(manifestData);
        setTokenStrings(tokens);
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

  // Phase 2: Load shard for selected token
  useEffect(() => {
    if (!manifest || !tokenStrings || selectedToken === null) return;
    if (selectedToken < 0 || selectedToken >= manifest.vocabSize) return;

    const shardIndex = Math.floor(selectedToken / manifest.shardSize);
    if (shardCache.current.has(shardIndex)) return;

    const prefix = prefixRef.current;
    if (!prefix) return;

    let cancelled = false;
    setNeighborsLoading(true);

    (async () => {
      try {
        const shard = await fetchAndDecompress<KnnShard>(
          `${prefix}-knn-${shardIndex}.json.br`
        );
        if (cancelled) return;
        shardCache.current.set(shardIndex, shard);
        setShardVersion((v) => v + 1);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        if (!cancelled) setNeighborsLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [selectedToken, manifest, tokenStrings]);

  // Phase 3: Background prefetch remaining shards
  useEffect(() => {
    if (!manifest || !tokenStrings) return;

    const prefix = prefixRef.current;
    if (!prefix) return;

    let cancelled = false;
    const cache = shardCache.current;

    // Wait a moment before starting prefetch
    const timeout = setTimeout(() => {
      if (cancelled) return;

      let shardQueue = Array.from(
        { length: manifest.numShards },
        (_, i) => i
      ).filter((i) => !cache.has(i));

      function prefetchNext() {
        if (cancelled || shardQueue.length === 0) return;
        const idx = shardQueue.shift()!;
        // Skip if already loaded (might have been loaded by Phase 2)
        if (cache.has(idx)) {
          prefetchNext();
          return;
        }

        if ("requestIdleCallback" in window) {
          requestIdleCallback(() => {
            if (cancelled) return;
            fetchAndDecompress<KnnShard>(`${prefix}-knn-${idx}.json.br`)
              .then((shard) => {
                if (cancelled) return;
                cache.set(idx, shard);
                setShardVersion((v) => v + 1);
                prefetchNext();
              })
              .catch(() => {
                // Prefetch failures are non-critical
                if (!cancelled) prefetchNext();
              });
          });
        } else {
          fetchAndDecompress<KnnShard>(`${prefix}-knn-${idx}.json.br`)
            .then((shard) => {
              if (cancelled) return;
              cache.set(idx, shard);
              setShardVersion((v) => v + 1);
              prefetchNext();
            })
            .catch(() => {
              if (!cancelled) prefetchNext();
            });
        }
      }

      prefetchNext();
    }, 500);

    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  }, [manifest, tokenStrings]);

  const search = useCallback(
    (query: string, limit = 50): SearchResult[] => {
      if (!tokenStrings || !query) return [];
      const lower = query.toLowerCase();
      const results: SearchResult[] = [];

      // Check if query is a numeric token ID
      if (/^\d+$/.test(query)) {
        const id = Number(query);
        if (id >= 0 && id < tokenStrings.length) {
          results.push({ id, text: tokenStrings[id] });
        }
      }

      // Search by token string
      for (const [tokenStr, tokenIds] of searchIndex.current) {
        if (results.length >= limit) break;
        if (tokenStr.includes(lower)) {
          for (const tokenId of tokenIds) {
            if (results.length >= limit) break;
            if (!results.some((r) => r.id === tokenId)) {
              results.push({ id: tokenId, text: tokenStrings[tokenId] });
            }
          }
        }
      }
      return results;
    },
    [tokenStrings]
  );

  const getToken = useCallback(
    (id: number): TokenEntry | undefined => {
      if (!tokenStrings || id < 0 || id >= tokenStrings.length) return undefined;
      if (!manifest) return undefined;
      const shardIndex = Math.floor(id / manifest.shardSize);
      const shard = shardCache.current.get(shardIndex);
      const neighbors = shard
        ? shard[id - shardIndex * manifest.shardSize]
        : [];
      return { s: tokenStrings[id], n: neighbors };
    },
    [tokenStrings, manifest]
  );

  return { loading, error, search, getToken, neighborsLoading };
}
