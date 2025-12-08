"""
build_embeddings.py

Helper script to build and store player embeddings for FPL Graph Retrieval.

Usage:
    python build_embeddings.py

This script will:
1. Build embeddings with model 1 (all-MiniLM-L6-v2)
2. Build embeddings with model 2 (paraphrase-MiniLM-L3-v2)
3. Create separate vector indexes for each model

You can then experiment with both models in graph_retrieval.py
"""

from graph_retrieval import build_and_store_player_embeddings

print("=" * 70)
print("FPL Player Embedding Builder")
print("=" * 70)
print("\nThis script will build embeddings for all players using")
print("two different SentenceTransformer models for comparison.")
print("\n" + "=" * 70)

# Model 1: all-MiniLM-L6-v2 (384 dimensions, good general purpose)
print("\n### Building Embeddings with Model 1 ###")
print("Model: sentence-transformers/all-MiniLM-L6-v2")
print("Description: General-purpose semantic similarity model")
print("-" * 70)

try:
    build_and_store_player_embeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_name="player_embedding_index_minilm",
    )
    print("\n✓ Model 1 embeddings created successfully!")
except Exception as e:
    print(f"\n✗ Error building Model 1 embeddings: {e}")
    import traceback
    traceback.print_exc()

# Model 2: paraphrase-MiniLM-L3-v2 (384 dimensions, optimized for paraphrase detection)
print("\n\n### Building Embeddings with Model 2 ###")
print("Model: sentence-transformers/paraphrase-MiniLM-L3-v2")
print("Description: Optimized for paraphrase detection and semantic similarity")
print("-" * 70)

try:
    build_and_store_player_embeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        index_name="player_embedding_index_para",
    )
    print("\n✓ Model 2 embeddings created successfully!")
except Exception as e:
    print(f"\n✗ Error building Model 2 embeddings: {e}")
    import traceback
    traceback.print_exc()

print("\n\n" + "=" * 70)
print("EMBEDDING BUILD COMPLETE!")
print("=" * 70)
print("\nTwo indexes created:")
print("  1. player_embedding_index_minilm  (all-MiniLM-L6-v2)")
print("  2. player_embedding_index_para    (paraphrase-MiniLM-L3-v2)")
print("\nYou can now test both models in graph_retrieval.py")
print("Use mode 2 (embedding) or mode 3 (hybrid) to try them out")
print("=" * 70)
