"""Compute k-nearest neighbors for all tokens in a model's embedding space."""

import argparse
import gzip
import json
import os

import faiss
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoTokenizer


TENSOR_NAMES = {
    "input": "model.embed_tokens.weight",
    "output": "lm_head.weight",
}


def load_embedding_weight(model_id: str, tensor_name: str) -> np.ndarray:
    """Load only a single tensor from a safetensors model."""
    index_path = hf_hub_download(model_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    shard_file = index["weight_map"][tensor_name]
    print(f"  {tensor_name} is in shard: {shard_file}")

    shard_path = hf_hub_download(model_id, shard_file)

    # Use torch framework since the weights are bfloat16 (unsupported by numpy).
    with safe_open(shard_path, framework="pt") as f:
        embeddings = f.get_tensor(tensor_name)

    return embeddings.float().numpy()


def compute_and_write(
    embeddings: np.ndarray,
    tokenizer: "AutoTokenizer",
    model_name: str,
    embedding_type: str,
    k: int,
    output_path: str,
):
    vocab_size, dim = embeddings.shape

    # Normalize for cosine similarity
    print("Normalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings_normalized = (embeddings / norms).astype(np.float32)
    del embeddings

    # Build GPU FAISS index (inner product on normalized vectors = cosine similarity)
    print("Building FAISS GPU index...")
    res = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
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

    # Build output data
    print("Building output data...")
    tokens = {}
    for token_id in tqdm(range(vocab_size), desc="Formatting"):
        token_str = tokenizer.decode([token_id])
        neighbors = []
        for j in range(k_query):
            neighbor_id = int(all_indices[token_id, j])
            sim = float(all_similarities[token_id, j])
            if neighbor_id == token_id:
                continue
            neighbors.append([neighbor_id, round(sim, 4)])
            if len(neighbors) == k:
                break
        tokens[str(token_id)] = {"s": token_str, "n": neighbors}

    data = {
        "model": model_name,
        "embedding": embedding_type,
        "k": k,
        "metric": "cosine_similarity",
        "tokens": tokens,
    }

    print(f"Writing output to {output_path}...")
    json_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
    print(f"  JSON size: {len(json_bytes) / 1024 / 1024:.1f} MB")
    with gzip.open(output_path, "wb", compresslevel=9) as f:
        f.write(json_bytes)
    print(f"  Gzipped size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


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
        help="Output directory for gzipped JSON files",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        choices=["input", "output", "both"],
        default="both",
        help="Which embeddings to compute (default: both)",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    slug = model_name.lower()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    types = ["input", "output"] if args.embedding == "both" else [args.embedding]

    for emb_type in types:
        tensor_name = TENSOR_NAMES[emb_type]
        print(f"\n{'='*60}")
        print(f"Processing {emb_type} embeddings ({tensor_name})")
        print(f"{'='*60}")

        print(f"Loading {emb_type} embedding weights...")
        embeddings = load_embedding_weight(args.model, tensor_name)
        print(f"Embedding matrix shape: {embeddings.shape}")

        output_path = os.path.join(args.output_dir, f"{slug}-{emb_type}.json.gz")
        compute_and_write(embeddings, tokenizer, model_name, emb_type, args.k, output_path)
        del embeddings

    print("\nAll done!")


if __name__ == "__main__":
    main()
