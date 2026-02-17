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

1. `useModelData` hook fetches `/data/{model}-{type}.json.gz` from `public/data/`, decompresses (browser-transparent or manual `DecompressionStream`), parses JSON
2. Builds an in-memory `Map<string, number>` search index (lowercased token string → token ID) in a `useRef`
3. Exposes `search()`, `getToken()`, `loading`, `error` to `App.tsx`
4. `App.tsx` owns all state (`modelId`, `embeddingType`, `selectedToken`) and syncs it bidirectionally with URL query params via `replaceState`/`pushState`/`popstate`

### Adding a new model

1. Run `build_data/compute_knn.py` to generate `{slug}-{type}.json.gz` files (requires GPU, uses FAISS)
2. Place the `.json.gz` files in `public/data/`
3. Add the model to `MODEL_FILES` in `src/hooks/useModelData.ts`

### Design system

CSS variables defined in `:root` in `App.css`. Fonts: Syne (display), DM Sans (body), JetBrains Mono (code/tokens), loaded from Google Fonts in `index.html`. Token text uses amber (`--accent-token`), similarity scores use slate blue (`--accent-similarity`).

### Key patterns

- No routing library — manual URL state management in `App.tsx`
- No state management library — plain `useState`/`useRef`/`useCallback`
- Similarity bars in `NeighborResults` are normalized to 15%–100% relative to the min/max in the current result set
- `visualizeToken()` in `util.ts` replaces whitespace with visible Unicode glyphs (space→`·`, newline→`↵`, tab→`⇥`)
