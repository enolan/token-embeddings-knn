# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
bun run dev       # Start Vite dev server
bun run build     # Type-check (tsc) then bundle (vite build)
bun run preview   # Serve production build locally (wrangler pages dev, runs Functions)
bun run deploy    # Build + deploy to Cloudflare Pages (production)
```

Canary deploys: `bun run build && bunx wrangler pages deploy dist/ --branch canary`

Type-checking the Cloudflare Functions (separate tsconfig): `tsc --noEmit -p functions/tsconfig.json`

No test framework is configured. `tsc --noEmit` is the primary correctness check (run both the root and `functions/` tsconfigs).

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

To add a new model, add it to `regenerate_all.sh`, `MODEL_PREFIXES` in `src/hooks/useModelData.ts`, and the duplicated `MODEL_PREFIXES`/`MODEL_DISPLAY_NAMES` in `functions/og.ts` and `functions/_middleware.ts` (marked with "keep in sync" comments).

### Python / uv

Always use `uv run` to run Python scripts in `build_data/` — never use the system Python or `uv pip`. The `uv run` command automatically uses the project's virtualenv and dependencies from `pyproject.toml`.

### Design system

CSS variables defined in `:root` in `App.css`. Fonts: Syne (display), DM Sans (body), JetBrains Mono (code/tokens), loaded from Google Fonts in `index.html`. Token text uses amber (`--accent-token`), similarity scores use slate blue (`--accent-similarity`).

### Hosting & OpenGraph

Deployed on Cloudflare Pages. Configuration in `wrangler.toml`.

- `functions/_middleware.ts` — HTMLRewriter middleware that rewrites OG meta tags (title, description, image, url) when the URL has `?model=` and `?token=` params
- `functions/og.ts` — `/og` endpoint that generates dynamic OG images using workers-og/satori, loading token data from static assets via `env.ASSETS.fetch()`
- Static fonts for OG rendering live in `public/fonts/`; non-Latin fallback fonts (CJK, Arabic, etc.) are loaded on-demand from Google Fonts

### Key patterns

- No routing library — manual URL state management in `App.tsx`
- No state management library — plain `useState`/`useRef`/`useCallback`
- Similarity bars in `NeighborResults` are normalized to 15%–100% relative to the min/max in the current result set
- `visualizeToken()` in `util.ts` replaces whitespace with visible Unicode glyphs (space→`·`, newline→`↵`, tab→`⇥`)
