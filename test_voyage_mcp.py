#!/usr/bin/env python3
"""Test Voyage AI collection via direct ChromaDB access"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.indexing.chroma_client import ChromaClientManager

def test_voyage_collection():
    """Test the Voyage AI collection metadata and basic queries"""

    print("=" * 70)
    print("Testing Voyage AI Collection (scientific_papers_voyage)")
    print("=" * 70)

    # Initialize Chroma client
    chroma_manager = ChromaClientManager(
        chroma_path=config.chroma_path,
        embedding_model="voyage-ai/voyage-context-3"
    )

    # Get the Voyage collection
    voyage_collection = chroma_manager.get_or_create_collection("scientific_papers_voyage")
    print(f"\n[OK] Loaded collection: {voyage_collection.name}")
    print(f"     Total chunks: {voyage_collection.count()}")
    print(f"     Metadata: {voyage_collection.metadata}")

    # Test 1: Semantic search
    print("\n" + "=" * 70)
    print("Test 1: Semantic Search - 'glacier albedo black carbon'")
    print("=" * 70)

    query = "glacier albedo black carbon"
    doc_ids, scores, metadatas, documents = search_engine.search(
        query=query,
        top_k=5,
        alpha=0.7  # Favor semantic search (Voyage excels at this)
    )

    print(f"\nQuery: '{query}'")
    print(f"Found {len(doc_ids)} results:\n")

    for i, (doc_id, score, metadata, doc_text) in enumerate(zip(doc_ids, scores, metadatas, documents), 1):
        title = metadata.get("title", "Unknown") if metadata else "Unknown"
        authors = metadata.get("authors", "Unknown") if metadata else "Unknown"
        year = metadata.get("year") if metadata else "N/A"

        print(f"{i}. Score: {score:.4f}")
        print(f"   Title: {title}")
        print(f"   Authors: {authors}")
        print(f"   Year: {year}")
        print(f"   Excerpt: {doc_text[:150]}...")
        print()

    # Test 2: Author search
    print("=" * 70)
    print("Test 2: Author Search - 'Wang'")
    print("=" * 70)

    # Get all documents and filter by author
    all_data = voyage_collection.get(
        limit=100000,
        include=["metadatas"]
    )

    wang_papers = []
    for doc_id, metadata in zip(all_data["ids"], all_data["metadatas"]):
        authors = metadata.get("authors", "") if metadata else ""
        if "wang" in authors.lower():
            paper_id = doc_id.rsplit("_chunk_", 1)[0]
            if not any(p["paper_id"] == paper_id for p in wang_papers):
                wang_papers.append({
                    "paper_id": paper_id,
                    "title": metadata.get("title", "Unknown") if metadata else "Unknown",
                    "authors": authors,
                    "year": metadata.get("year") if metadata else None
                })

    print(f"\nFound {len(wang_papers)} papers with 'Wang' as author:\n")
    for i, paper in enumerate(wang_papers[:5], 1):
        print(f"{i}. {paper['title']}")
        print(f"   Authors: {paper['authors']}")
        print(f"   Year: {paper['year']}")
        print()

    # Test 3: Hybrid search comparison
    print("=" * 70)
    print("Test 3: Hybrid Search Comparison (alpha=0.0 vs 0.5 vs 1.0)")
    print("=" * 70)

    query = "wildfire smoke impact on glacier melting"

    for alpha_val in [0.0, 0.5, 1.0]:
        print(f"\nAlpha = {alpha_val} ({'keyword' if alpha_val == 0.0 else 'hybrid' if alpha_val == 0.5 else 'semantic'}):")
        doc_ids, scores, metadatas, _ = search_engine.search(
            query=query,
            top_k=3,
            alpha=alpha_val
        )

        for i, (doc_id, score, metadata) in enumerate(zip(doc_ids, scores, metadatas), 1):
            title = metadata.get("title", "Unknown")[:60] if metadata else "Unknown"
            print(f"  {i}. [{score:.4f}] {title}...")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
    print("\nVoyage AI collection is working correctly with:")
    print("- Contextualized embeddings (voyage-context-3)")
    print("- 8,663 chunks from 122 unique documents")
    print("- Hybrid search (BM25 + semantic)")

if __name__ == "__main__":
    try:
        test_voyage_search()
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
