#!/usr/bin/env python3
"""Simple test of Voyage AI collection"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.indexing.chroma_client import ChromaClientManager

def main():
    print("=" * 70)
    print("Testing Voyage AI Collection")
    print("=" * 70)

    # Initialize Chroma
    manager = ChromaClientManager(chroma_path=config.chroma_path)

    # List all collections
    print("\n1. Available Collections:")
    collections = manager.list_collections()
    for coll in collections:
        print(f"   - {coll.name}: {coll.count()} documents")

    # Get Voyage collection
    print("\n2. Loading Voyage AI collection...")
    voyage_coll = manager.get_or_create_collection("scientific_papers_voyage")
    print(f"   [OK] Collection: {voyage_coll.name}")
    print(f"   [OK] Total chunks: {voyage_coll.count()}")

    # Sample documents
    print("\n3. Sampling 5 documents...")
    sample = voyage_coll.get(limit=5, include=["metadatas", "documents"])

    for i, (doc_id, metadata, document) in enumerate(zip(
        sample["ids"], sample["metadatas"], sample["documents"]
    ), 1):
        print(f"\n   Document {i}:")
        print(f"   ID: {doc_id}")
        print(f"   Title: {metadata.get('title', 'N/A')[:60]}...")
        print(f"   Authors: {metadata.get('authors', 'N/A')[:60]}...")
        print(f"   Year: {metadata.get('year', 'N/A')}")
        print(f"   Text: {document[:100]}...")

    # Count unique papers
    print("\n4. Counting unique papers...")
    all_ids = voyage_coll.get(include=[])["ids"]
    unique_papers = set(doc_id.rsplit("_chunk_", 1)[0] for doc_id in all_ids)
    print(f"   Total chunks: {len(all_ids)}")
    print(f"   Unique papers: {len(unique_papers)}")
    print(f"   Avg chunks/paper: {len(all_ids) / len(unique_papers):.1f}")

    print("\n" + "=" * 70)
    print("Voyage AI collection is ready for use!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
