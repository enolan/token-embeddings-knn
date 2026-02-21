# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
bun run dev       # Start Vite dev server
bun run build     # Type-check (tsc) then bundle (vite build)
bun run preview   # Serve production build locally
```

No test framework is configured. `tsc --noEmit` is the primary correctness check.

## Architecture

Single-page React 19 + TypeScript app (Vite 7) for exploring K-nearest-neighbor relationships in LLM token embeddings. Users search for a token, see its nearest neighbors by cosine similarity, and click through to explore.

### Data flow

1. `useModelData` hook fetches a manifest, then a tokens file, then KNN shards on demand from `public/data/`
2. Data is split into three file types per model/embedding pair:
   - `{prefix}-manifest.json` — metadata (vocabSize, shardSize, numShards, k)
   - `{prefix}-tokens.json.br` — flat array of token strings (brotli-compressed)
   - `{prefix}-knn-{i}.json.br` — neighbor lists for a chunk of tokens (brotli-compressed, ~16K tokens each)
3. Tokens load first so search is interactive immediately; KNN shards load on demand when a token is selected, with background prefetch for the rest
4. Builds an in-memory `Map<string, number[]>` search index (lowercased token string → token IDs) in a `useRef`
5. Exposes `search()`, `getToken()`, `loading`, `neighborsLoading`, `error` to `App.tsx`
6. `App.tsx` owns all state (`modelId`, `embeddingType`, `selectedToken`) and syncs it bidirectionally with URL query params via `replaceState`/`pushState`/`popstate`

### Generating data

Requires a GPU (or pass `--cpu` to `compute_knn.py`). All model IDs, slugs, and parameters are defined in the script:

```bash
cd build_data
./regenerate_all.sh
```

To add a new model, add it to `regenerate_all.sh` and to `MODEL_PREFIXES` in `src/hooks/useModelData.ts`.

### Python / uv

Always use `uv run` to run Python scripts in `build_data/` — never use the system Python or `uv pip`. The `uv run` command automatically uses the project's virtualenv and dependencies from `pyproject.toml`.

### Design system

CSS variables defined in `:root` in `App.css`. Fonts: Syne (display), DM Sans (body), JetBrains Mono (code/tokens), loaded from Google Fonts in `index.html`. Token text uses amber (`--accent-token`), similarity scores use slate blue (`--accent-similarity`).

### Key patterns

- No routing library — manual URL state management in `App.tsx`
- No state management library — plain `useState`/`useRef`/`useCallback`
- Similarity bars in `NeighborResults` are normalized to 15%–100% relative to the min/max in the current result set
- `visualizeToken()` in `util.ts` replaces whitespace with visible Unicode glyphs (space→`·`, newline→`↵`, tab→`⇥`)
