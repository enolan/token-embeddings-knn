export interface TokenEntry {
  /** Token string representation */
  s: string;
  /** Neighbors as [token_id, similarity] pairs */
  n: [number, number][];
}

export interface ModelData {
  model: string;
  k: number;
  metric: string;
  tokens: Record<string, TokenEntry>;
}

export interface SearchResult {
  id: number;
  text: string;
}

export interface ShardManifest {
  model: string;
  k: number;
  metric: string;
  vocabSize: number;
  shardSize: number;
  numShards: number;
}

/** Flat array where index = token ID */
export type TokenStrings = string[];

/** Array of neighbor-lists. Token ID = shardIndex * shardSize + arrayIndex */
export type KnnShard = [number, number][][];
