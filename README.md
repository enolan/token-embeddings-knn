# Token Embeddings KNN Explorer

Interactive tool for exploring K-nearest-neighbor relationships in LLM token embeddings. Search for a token, see its nearest neighbors by cosine similarity, and click through to explore.

Live at [token-embeddings-knn.pages.dev](https://token-embeddings-knn.pages.dev/)

## Models

- Qwen3-30B-A3B (input + output embeddings)
- Llama 3.1-8B (input + output embeddings)
- Gemma 3-4B (input embeddings only — tied weights)

## Development

```bash
bun run dev       # Start Vite dev server
bun run build     # Type-check then bundle
bun run deploy    # Build and deploy to Cloudflare Pages
```

## Generating data

KNN data is precomputed with FAISS and stored as sharded brotli-compressed JSON in `public/data/` (gitignored due to size). To regenerate all models:

```bash
cd build_data
./regenerate_all.sh
```

Requires a GPU. Model IDs, slugs, and parameters (k, shard size) are defined in `regenerate_all.sh`.
