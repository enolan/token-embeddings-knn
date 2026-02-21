"""Compute k-nearest neighbors for all tokens in a model's embedding space."""

import argparse
import json
import os

import brotli
import faiss
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoTokenizer


INPUT_TENSOR_CANDIDATES = [
    "model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
]

OUTPUT_TENSOR_CANDIDATES = [
    "lm_head.weight",
    "language_model.lm_head.weight",
]

TENSOR_CANDIDATES = {
    "input": INPUT_TENSOR_CANDIDATES,
    "output": OUTPUT_TENSOR_CANDIDATES,
}


def _resolve_tensor_name(available_names: list[str], candidates: list[str], label: str) -> str:
    """Find the first matching tensor name from candidates."""
    for candidate in candidates:
        if candidate in available_names:
            return candidate
    embed_head = [k for k in available_names if "embed" in k.lower() or "head" in k.lower()]
    raise KeyError(
        f"No known {label} tensor found. Tried: {candidates}. "
        f"Available tensors containing 'embed' or 'head': {embed_head}"
    )


def load_embedding_weight(model_id: str, candidates: list[str], label: str) -> np.ndarray:
    """Load only a single tensor from a safetensors model (sharded or single-file)."""
    try:
        # Try sharded model first
        index_path = hf_hub_download(model_id, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        tensor_name = _resolve_tensor_name(list(index["weight_map"].keys()), candidates, label)
        shard_file = index["weight_map"][tensor_name]
        print(f"  {tensor_name} is in shard: {shard_file}")
        shard_path = hf_hub_download(model_id, shard_file)
    except Exception as e:
        if "model.safetensors.index.json" in str(e) or "EntryNotFoundError" in type(e).__name__:
            # Single-file model
            print(f"  No shard index found, trying single-file model.safetensors...")
            shard_path = hf_hub_download(model_id, "model.safetensors")
            with safe_open(shard_path, framework="pt") as f:
                tensor_name = _resolve_tensor_name(list(f.keys()), candidates, label)
        else:
            raise

    print(f"  Loading tensor: {tensor_name}")
    # Use torch framework since the weights are bfloat16 (unsupported by numpy).
    with safe_open(shard_path, framework="pt") as f:
        embeddings = f.get_tensor(tensor_name)

    return embeddings.float().numpy()


def _write_brotli(path: str, data: bytes, quality: int = 9) -> int:
    """Write brotli-compressed data and return compressed size."""
    compressed = brotli.compress(data, quality=quality)
    with open(path, "wb") as f:
        f.write(compressed)
    return len(compressed)


def compute_and_write(
    embeddings: np.ndarray,
    tokenizer: "AutoTokenizer",
    model_name: str,
    embedding_type: str,
    k: int,
    output_dir: str,
    prefix: str,
    shard_size: int,
    use_gpu: bool = True,
):
    vocab_size, dim = embeddings.shape

    # Normalize for cosine similarity
    print("Normalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings_normalized = (embeddings / norms).astype(np.float32)
    del embeddings

    # Build FAISS index (inner product on normalized vectors = cosine similarity)
    cpu_index = faiss.IndexFlatIP(dim)

    if use_gpu:
        print("Building FAISS GPU index...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        print("Building FAISS CPU index...")
        index = cpu_index

    index.add(embeddings_normalized)

    # Search in batches to show progress
    k_query = k + 1  # +1 because each token's nearest neighbor is itself
    batch_size = 8192
    all_similarities = np.empty((vocab_size, k_query), dtype=np.float32)
    all_indices = np.empty((vocab_size, k_query), dtype=np.int64)

    print(f"Computing {k} nearest neighbors for {vocab_size} tokens...")
    for start in tqdm(range(0, vocab_size, batch_size), desc="KNN search"):
        end = min(start + batch_size, vocab_size)
        sims, idxs = index.search(embeddings_normalized[start:end], k_query)
        all_similarities[start:end] = sims
        all_indices[start:end] = idxs

    del embeddings_normalized, index, cpu_index

    # Build token strings array
    print("Building token strings...")
    token_strings = []
    for token_id in tqdm(range(vocab_size), desc="Decoding tokens"):
        token_strings.append(tokenizer.decode([token_id]))

    # Build KNN neighbor lists
    print("Building neighbor lists...")
    all_neighbors = []
    for token_id in tqdm(range(vocab_size), desc="Formatting neighbors"):
        neighbors = []
        for j in range(k_query):
            neighbor_id = int(all_indices[token_id, j])
            sim = float(all_similarities[token_id, j])
            if neighbor_id == token_id:
                continue
            neighbors.append([neighbor_id, round(sim, 4)])
            if len(neighbors) == k:
                break
        all_neighbors.append(neighbors)

    del all_similarities, all_indices

    num_shards = (vocab_size + shard_size - 1) // shard_size

    # 1. Write manifest
    manifest = {
        "model": model_name,
        "k": k,
        "metric": "cosine_similarity",
        "vocabSize": vocab_size,
        "shardSize": shard_size,
        "numShards": num_shards,
    }
    manifest_path = os.path.join(output_dir, f"{prefix}-manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"  Manifest: {os.path.getsize(manifest_path)} bytes -> {manifest_path}")

    # 2. Write tokens file (brotli-compressed)
    tokens_path = os.path.join(output_dir, f"{prefix}-tokens.json.br")
    tokens_json = json.dumps(token_strings, ensure_ascii=False).encode("utf-8")
    tokens_compressed = _write_brotli(tokens_path, tokens_json)
    print(f"  Tokens: {len(tokens_json) / 1024 / 1024:.1f} MB JSON -> {tokens_compressed / 1024 / 1024:.1f} MB brotli -> {tokens_path}")

    # 3. Write KNN shards (brotli-compressed)
    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, vocab_size)
        shard_data = all_neighbors[start:end]
        shard_path = os.path.join(output_dir, f"{prefix}-knn-{shard_idx}.json.br")
        shard_json = json.dumps(shard_data).encode("utf-8")
        shard_compressed = _write_brotli(shard_path, shard_json)
        print(f"  Shard {shard_idx}: tokens {start}-{end-1}, {len(shard_json) / 1024 / 1024:.1f} MB JSON -> {shard_compressed / 1024 / 1024:.1f} MB brotli -> {shard_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute KNN for token embeddings"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g. Qwen/Qwen3-30B-A3B)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for sharded data files",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        choices=["input", "output", "both"],
        default="both",
        help="Which embeddings to compute (default: both)",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors")
    parser.add_argument(
        "--shard-size",
        type=int,
        default=16384,
        help="Number of tokens per KNN shard (default: 16384)",
    )
    parser.add_argument(
        "--input-tensor",
        type=str,
        default=None,
        help="Override tensor name for input embeddings (default: model.embed_tokens.weight)",
    )
    parser.add_argument(
        "--output-tensor",
        type=str,
        default=None,
        help="Override tensor name for output embeddings (default: lm_head.weight)",
    )
    parser.add_argument(
        "--slug",
        type=str,
        default=None,
        help="Override output filename slug (default: lowercased model name)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU for FAISS",
    )
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    slug = args.slug or model_name.lower()
    os.makedirs(args.output_dir, exist_ok=True)

    tensor_overrides = {}
    if args.input_tensor:
        tensor_overrides["input"] = [args.input_tensor]
    if args.output_tensor:
        tensor_overrides["output"] = [args.output_tensor]

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    types = ["input", "output"] if args.embedding == "both" else [args.embedding]

    for emb_type in types:
        candidates = tensor_overrides.get(emb_type, TENSOR_CANDIDATES[emb_type])
        print(f"\n{'='*60}")
        print(f"Processing {emb_type} embeddings")
        print(f"{'='*60}")

        print(f"Loading {emb_type} embedding weights...")
        embeddings = load_embedding_weight(args.model, candidates, emb_type)
        print(f"Embedding matrix shape: {embeddings.shape}")

        prefix = f"{slug}-{emb_type}"
        compute_and_write(
            embeddings, tokenizer, model_name, emb_type, args.k,
            args.output_dir, prefix, args.shard_size,
            use_gpu=not args.cpu,
        )
        del embeddings

    print("\nAll done!")


if __name__ == "__main__":
    main()
