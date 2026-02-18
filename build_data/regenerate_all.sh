#!/usr/bin/env bash
set -euo pipefail

# Regenerate all KNN data files for the app.
# Run from the build_data/ directory: ./regenerate_all.sh

K=20
SHARD_SIZE=16384
OUTPUT_DIR="../public/data"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

declare -A MODELS=(
  # slug -> "hf_model_id embedding_types"
  [qwen3-30b-a3b]="Qwen/Qwen3-30B-A3B both"
  [llama-3.1-8b]="meta-llama/Llama-3.1-8B both"
  [gemma-3-4b]="google/gemma-3-4b-pt input"
)

for slug in "${!MODELS[@]}"; do
  read -r model_id embedding <<< "${MODELS[$slug]}"
  echo ""
  echo "========================================"
  echo "  $slug ($model_id, --embedding $embedding)"
  echo "========================================"
  uv run compute_knn.py \
    --model "$model_id" \
    --slug "$slug" \
    --output-dir "$OUTPUT_DIR" \
    --embedding "$embedding" \
    --k "$K" \
    --shard-size "$SHARD_SIZE"
done

echo ""
echo "All models regenerated (k=$K, shard_size=$SHARD_SIZE)."
