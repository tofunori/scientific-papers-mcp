#!/usr/bin/env python3
"""Delete the incomplete Voyage collection"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.indexing.chroma_client import ChromaClientManager

def main():
    print("Deleting incomplete Voyage collection...")

    manager = ChromaClientManager(chroma_path=config.chroma_path)

    try:
        manager.delete_collection("scientific_papers_voyage")
        print("[OK] Collection deleted successfully")
    except Exception as e:
        print(f"[ERROR] Failed to delete: {e}")
        sys.exit(1)

    # List collections after
    print("\nCollections after deletion:")
    collections = manager.list_collections()
    for coll in collections:
        print(f"  - {coll.name}: {coll.count()} documents")

    print("\nReady to reindex with fixed client!")

if __name__ == "__main__":
    main()
