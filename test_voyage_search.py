#!/usr/bin/env python3
"""Test Voyage collection with a simple search query"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import chromadb
from src.config import config
from src.indexing.hybrid_search import HybridSearchEngine

def test_voyage_search():
    """Test search on Voyage collection"""
    print("=" * 70)
    print("Testing Voyage Collection Search")
    print("=" * 70)

    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(config.chroma_path))

    # Load Voyage collection
    collection = chroma_client.get_collection(name="scientific_papers_voyage")

    print(f"\nCollection stats:")
    print(f"  - Name: {collection.name}")
    print(f"  - Document count: {collection.count()}")
    print(f"  - Metadata: {collection.metadata}")

    # Initialize search engine with Voyage
    config.use_voyage_api = True
    config.use_jina_api = False

    search_engine = HybridSearchEngine(
        chroma_collection=collection,
        embedding_model="voyage"
    )

    # Test query
    test_query = "glacier albedo reduction from wildfire aerosols"
    print(f"\nTest query: '{test_query}'")
    print("-" * 70)

    # Search with hybrid mode (returns tuple: ids, scores, metadatas, documents)
    ids, scores, metadatas, documents = search_engine.search(
        query=test_query,
        top_k=5,
        alpha=0.7  # 70% semantic, 30% keyword
    )

    print(f"\nFound {len(ids)} results:\n")

    for i, (doc_id, score, metadata, text) in enumerate(zip(ids, scores, metadatas, documents), 1):
        print(f"{i}. ID: {doc_id}")
        print(f"   Score: {score:.4f}")
        print(f"   Source: {metadata.get('source', 'N/A')}")
        print(f"   Page: {metadata.get('page', 'N/A')}")
        print(f"   Text preview: {text[:200]}...")
        print()

    print("=" * 70)
    print("[OK] Voyage search test completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    test_voyage_search()
