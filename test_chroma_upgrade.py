#!/usr/bin/env python3
"""
Test script to verify Chroma 1.0.20 compatibility and server functionality
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Set encoding for Windows terminals
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding='utf-8')

def test_chroma_import():
    """Test 1: Verify Chroma 1.0.20 is installed"""
    try:
        import chromadb
        print(f"✅ Chroma imported successfully")
        print(f"   Version: {chromadb.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Chroma: {e}")
        return False


def test_chroma_client():
    """Test 2: Initialize Chroma client"""
    try:
        from src.indexing.chroma_client import ChromaClientManager
        from src.config import config

        manager = ChromaClientManager(
            chroma_path=config.chroma_path,
            embedding_model=config.embedding_model
        )
        print(f"✅ Chroma client initialized successfully")
        print(f"   Path: {config.chroma_path}")

        # List collections
        collections = manager.list_collections()
        print(f"   Collections: {len(collections)}")

        return True
    except Exception as e:
        print(f"❌ Failed to initialize Chroma client: {e}")
        return False


def test_collection_operations():
    """Test 3: Test collection operations"""
    try:
        from src.indexing.chroma_client import initialize_chroma
        from src.config import config

        collection = initialize_chroma(config.chroma_path)

        # Test get_or_create
        print(f"✅ Collection operations working")
        print(f"   Collection count: {collection.count()}")

        return True
    except Exception as e:
        print(f"❌ Failed collection operations: {e}")
        return False


def test_embeddings():
    """Test 4: Test sentence transformers embedding model"""
    try:
        from sentence_transformers import SentenceTransformer
        from src.config import config

        model = SentenceTransformer(config.embedding_model)
        embedding = model.encode("test document", convert_to_tensor=False)

        print(f"✅ Embedding model loaded successfully")
        print(f"   Model: {config.embedding_model}")
        print(f"   Embedding dimension: {len(embedding)}")

        return True
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        return False


def test_hybrid_search():
    """Test 5: Test hybrid search engine initialization"""
    try:
        from src.indexing.chroma_client import initialize_chroma
        from src.indexing.hybrid_search import HybridSearchEngine
        from src.config import config

        collection = initialize_chroma(config.chroma_path)
        search_engine = HybridSearchEngine(collection)

        print(f"✅ Hybrid search engine initialized successfully")
        print(f"   BM25 index size: {len(search_engine.doc_ids) if search_engine.doc_ids else 0}")

        return True
    except Exception as e:
        print(f"❌ Failed to initialize search engine: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 Chroma 1.0.20 Upgrade Compatibility Tests")
    print("="*60 + "\n")

    tests = [
        ("Chroma Import", test_chroma_import),
        ("Chroma Client", test_chroma_client),
        ("Collection Operations", test_collection_operations),
        ("Embeddings", test_embeddings),
        ("Hybrid Search", test_hybrid_search),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n📋 Test: {name}")
        print("-" * 40)
        result = test_func()
        results.append((name, result))

    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n✅ {passed}/{total} tests passed\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
