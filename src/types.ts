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
