"""
build_embeddings.py

Helper script to build and store player embeddings for FPL Graph Retrieval.

Usage:
    python build_embeddings.py

This script will:
1. Build embeddings with model 1 (all-MiniLM-L6-v2) using cloud API (M1 compatible)
2. Build embeddings with model 2 (paraphrase-MiniLM-L3-v2) using cloud API (M1 compatible)
3. Create separate vector indexes for each model

You can then experiment with both models in graph_retrieval.py
"""

from graph_retrieval import build_and_store_player_embeddings
from input_processing import load_hf_token

print("=" * 70)
print("FPL Player Embedding Builder (M1 Compatible - Cloud Mode)")
print("=" * 70)
print("\nThis script will build embeddings for all players using")
print("two different SentenceTransformer models via HuggingFace API.")
print("\n" + "=" * 70)

# Load HuggingFace token
print("\nLoading HuggingFace token from hf.txt...")
hf_token = load_hf_token()
if not hf_token:
    print("\n" + "!" * 70)
    print("ERROR: hf.txt not found!")
    print("!" * 70)
    print("\nTo use cloud mode (M1 compatible), create a file named 'hf.txt'")
    print("containing your HuggingFace API token.")
    print("\nGet your token from: https://huggingface.co/settings/tokens")
    print("\nAlternatively, set use_cloud=False in the function calls below")
    print("to use local models (may cause OpenMP lock issues on M1 Macs).")
    print("!" * 70)
    import sys
    sys.exit(1)

print(f"✓ Token loaded: {hf_token[:10]}...")

# Model 1: all-MiniLM-L6-v2 (384 dimensions, good general purpose)
print("\n### Building Embeddings with Model 1 (Cloud Mode) ###")
print("Model: sentence-transformers/all-MiniLM-L6-v2")
print("Description: General-purpose semantic similarity model")
print("-" * 70)

try:
    build_and_store_player_embeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_name="player_embedding_index_minilm",
        use_cloud=True,  # M1 compatible
        hf_token=hf_token,
    )
    print("\n✓ Model 1 embeddings created successfully!")
except Exception as e:
    print(f"\n✗ Error building Model 1 embeddings: {e}")
    import traceback
    traceback.print_exc()

# Model 2: paraphrase-MiniLM-L3-v2 (384 dimensions, optimized for paraphrase detection)
print("\n\n### Building Embeddings with Model 2 (Cloud Mode) ###")
print("Model: sentence-transformers/paraphrase-MiniLM-L3-v2")
print("Description: Optimized for paraphrase detection and semantic similarity")
print("-" * 70)

try:
    build_and_store_player_embeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        index_name="player_embedding_index_para",
        use_cloud=True,  # M1 compatible
        hf_token=hf_token,
    )
    print("\n✓ Model 2 embeddings created successfully!")
except Exception as e:
    print(f"\n✗ Error building Model 2 embeddings: {e}")
    import traceback
    traceback.print_exc()

print("\n\n" + "=" * 70)
print("EMBEDDING BUILD COMPLETE!")
print("=" * 70)
print("\nTwo indexes created using cloud API (M1 compatible):")
print("  1. player_embedding_index_minilm  (all-MiniLM-L6-v2)")
print("  2. player_embedding_index_para    (paraphrase-MiniLM-L3-v2)")
print("\nYou can now test both models in graph_retrieval.py")
print("Use mode 2 (embedding) or mode 3 (hybrid) to try them out")
print("=" * 70)
