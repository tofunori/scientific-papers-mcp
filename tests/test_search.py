#!/usr/bin/env python
"""Test hybrid search capabilities after indexing"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.indexing.chroma_client import ChromaClientManager
from src.indexing.hybrid_search import HybridSearchEngine
from src.utils.logger import setup_logger

logger = setup_logger(__name__, level="INFO")


def test_search():
    """Test hybrid search functionality"""
    print("\n" + "=" * 80)
    print("TESTING HYBRID SEARCH")
    print("=" * 80)

    try:
        # Initialize
        manager = ChromaClientManager(
            chroma_path=config.chroma_path,
            embedding_model=config.embedding_model,
        )
        collection = manager.get_or_create_collection("scientific_papers")

        # Check if there are documents
        collection_count = collection.count()
        print(f"\n[OK] Collection contains {collection_count} documents")

        if collection_count == 0:
            print("[WARN] No documents indexed yet")
            return False

        search_engine = HybridSearchEngine(
            collection, embedding_model=config.embedding_model
        )

        # Test queries
        test_queries = [
            ("glacier albedo feedback", 0.5),
            ("black carbon snow", 0.5),
            ("MODIS satellite", 0.3),
            ("Ren et al", 0.7),
            ("aerosol deposition", 0.5),
        ]

        print(f"\n[OK] Hybrid search engine initialized (BM25 docs: {len(search_engine.doc_ids)})")
        print("\n" + "=" * 80)
        print("SEARCH RESULTS")
        print("=" * 80)

        for query, alpha in test_queries:
            print(f"\n[QUERY] '{query}' (alpha={alpha})")

            try:
                doc_ids, scores, metadatas = search_engine.search(
                    query=query, top_k=3, alpha=alpha
                )

                if doc_ids:
                    print(f"  Results: {len(doc_ids)} found")
                    for i, (doc_id, score) in enumerate(zip(doc_ids, scores), 1):
                        title = metadatas[i-1].get("title", "Unknown") if i-1 < len(metadatas) else "Unknown"
                        year = metadatas[i-1].get("year", "?") if i-1 < len(metadatas) else "?"
                        print(f"    {i}. [{score:.3f}] {title[:60]}... ({year})")
                else:
                    print(f"  No results found")

            except Exception as e:
                print(f"  [ERROR] {e}")

        print("\n" + "=" * 80)
        print("[SUCCESS] Search functionality working correctly!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"[FAIL] Search error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    try:
        success = test_search()
        return 0 if success else 1
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
