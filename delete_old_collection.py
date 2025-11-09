#!/usr/bin/env python3
"""Delete the old scientific_papers collection"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.indexing.chroma_client import ChromaClientManager

def main():
    print("=" * 70)
    print("Deleting old 'scientific_papers' collection")
    print("=" * 70)

    manager = ChromaClientManager(chroma_path=config.chroma_path)

    # List collections before
    print("\nCollections BEFORE deletion:")
    collections = manager.list_collections()
    for coll in collections:
        print(f"  - {coll.name}: {coll.count()} documents")

    # Delete the old collection
    print("\nDeleting 'scientific_papers' collection...")
    try:
        manager.delete_collection("scientific_papers")
        print("✓ Collection deleted successfully")
    except Exception as e:
        print(f"[ERROR] Failed to delete: {e}")
        sys.exit(1)

    # List collections after
    print("\nCollections AFTER deletion:")
    collections = manager.list_collections()
    for coll in collections:
        print(f"  - {coll.name}: {coll.count()} documents")

    print("\n" + "=" * 70)
    print("Old collection successfully removed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
