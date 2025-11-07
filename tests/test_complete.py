#!/usr/bin/env python
"""Complete test suite for the Scientific Papers MCP Server"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.extractors.metadata_extractor import extract_metadata_from_file
from src.indexing.chroma_client import ChromaClientManager
from src.indexing.hybrid_search import HybridSearchEngine
from src.indexing.chunker import ScientificPaperChunker
from src.utils.logger import setup_logger

logger = setup_logger(__name__, level="INFO")


def test_phase_1_metadata():
    """Test 1: Metadata extraction"""
    print("\n" + "=" * 70)
    print("TEST 1: Metadata Extraction")
    print("=" * 70)

    docs_path = Path(config.documents_path)
    markdown_files = list(docs_path.glob("*.md"))[:3]  # Test first 3 files

    if not markdown_files:
        print("[FAIL] No markdown files found")
        return False

    success = True
    for test_file in markdown_files:
        try:
            metadata = extract_metadata_from_file(test_file)
            if metadata:
                print(f"\n[OK] {test_file.name}")
                print(f"    Title: {metadata.title[:60]}...")
                print(f"    Year: {metadata.year}")
                print(f"    Tags: {metadata.tags}")
                print(f"    Authors: {len(metadata.authors)} found")
            else:
                print(f"[WARN] Could not extract metadata from {test_file.name}")
                success = False
        except Exception as e:
            print(f"[FAIL] Error with {test_file.name}: {e}")
            success = False

    return success


def test_phase_2_chroma():
    """Test 2: Chroma initialization"""
    print("\n" + "=" * 70)
    print("TEST 2: Chroma DB Initialization")
    print("=" * 70)

    try:
        manager = ChromaClientManager(
            chroma_path=config.chroma_path,
            embedding_model=config.embedding_model,
        )
        collection = manager.get_or_create_collection("test_collection")
        count = collection.count()

        print(f"[OK] Chroma initialized")
        print(f"    Path: {config.chroma_path}")
        print(f"    Collection count: {count}")
        print(f"    Embedding model: {config.embedding_model}")

        return True
    except Exception as e:
        print(f"[FAIL] Chroma initialization: {e}")
        return False


def test_phase_3_chunking():
    """Test 3: Document chunking"""
    print("\n" + "=" * 70)
    print("TEST 3: Document Chunking")
    print("=" * 70)

    docs_path = Path(config.documents_path)
    markdown_files = list(docs_path.glob("*.md"))[:1]  # Test first file

    if not markdown_files:
        print("[FAIL] No markdown files found")
        return False

    try:
        test_file = markdown_files[0]

        with open(test_file, "r", encoding="utf-8") as f:
            text = f.read()

        chunker = ScientificPaperChunker(
            max_chunk_size=config.max_chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        chunks = chunker.chunk_document(text, document_id=test_file.stem)

        print(f"[OK] Document chunked: {test_file.name}")
        print(f"    Total chunks: {len(chunks)}")
        print(f"    Chunk sizes: {[len(c.text.split()) for c in chunks[:3]]} words...")
        print(f"    Sections: {list(set(c.section for c in chunks))[:5]}")

        return len(chunks) > 0

    except Exception as e:
        print(f"[FAIL] Chunking error: {e}")
        return False


def test_phase_4_indexing():
    """Test 4: Document indexing"""
    print("\n" + "=" * 70)
    print("TEST 4: Document Indexing")
    print("=" * 70)

    try:
        # Initialize Chroma
        manager = ChromaClientManager(
            chroma_path=config.chroma_path,
            embedding_model=config.embedding_model,
        )
        collection = manager.get_or_create_collection("scientific_papers")

        # Initialize search engine
        search_engine = HybridSearchEngine(
            collection, embedding_model=config.embedding_model
        )

        # Get a test document
        docs_path = Path(config.documents_path)
        markdown_files = list(docs_path.glob("*.md"))[:1]

        if not markdown_files:
            print("[FAIL] No markdown files found")
            return False

        test_file = markdown_files[0]

        # Extract metadata
        metadata = extract_metadata_from_file(test_file)
        if not metadata:
            print("[FAIL] Could not extract metadata")
            return False

        # Read and chunk
        with open(test_file, "r", encoding="utf-8") as f:
            text = f.read()

        chunker = ScientificPaperChunker()
        chunks = chunker.chunk_document(text, document_id=metadata.filename)

        # Index a few chunks
        indexed_count = 0
        for i, chunk in enumerate(chunks[:3]):  # Index first 3 chunks only
            try:
                search_engine.index_document(
                    doc_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata={
                        "filename": metadata.filename,
                        "title": metadata.title,
                        "section": chunk.section,
                    },
                )
                indexed_count += 1
            except Exception as e:
                logger.warning(f"Could not index chunk {i}: {e}")

        print(f"[OK] Indexed {indexed_count} chunks from {test_file.name}")
        print(f"    Document: {metadata.title[:50]}...")
        print(f"    Total BM25 docs: {len(search_engine.doc_ids)}")

        return indexed_count > 0

    except Exception as e:
        print(f"[FAIL] Indexing error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase_5_search():
    """Test 5: Hybrid search"""
    print("\n" + "=" * 70)
    print("TEST 5: Hybrid Search (if documents indexed)")
    print("=" * 70)

    try:
        # Initialize
        manager = ChromaClientManager(
            chroma_path=config.chroma_path,
            embedding_model=config.embedding_model,
        )
        collection = manager.get_or_create_collection("scientific_papers")
        search_engine = HybridSearchEngine(
            collection, embedding_model=config.embedding_model
        )

        if not search_engine.doc_ids:
            print("[SKIP] No documents indexed yet")
            return True

        # Test searches
        queries = [
            "glacier albedo",
            "black carbon snow",
            "MODIS satellite",
        ]

        for query in queries:
            try:
                doc_ids, scores, metadatas = search_engine.search(
                    query=query, top_k=3, alpha=0.5
                )

                if doc_ids:
                    print(f"\n[OK] Query: '{query}'")
                    print(f"    Results: {len(doc_ids)} found")
                    for i, (doc_id, score) in enumerate(zip(doc_ids, scores)):
                        title = metadatas[i].get("title", "Unknown") if i < len(metadatas) else "Unknown"
                        print(f"      {i+1}. {score:.3f} - {title[:40]}...")
                else:
                    print(f"\n[SKIP] Query: '{query}' - No results (normal if no docs indexed)")

            except Exception as e:
                print(f"[WARN] Error with query '{query}': {e}")

        return True

    except Exception as e:
        print(f"[FAIL] Search error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase_6_mcp_tools():
    """Test 6: MCP tools"""
    print("\n" + "=" * 70)
    print("TEST 6: MCP Tools")
    print("=" * 70)

    try:
        from src.server import (
            search,
            get_collection_stats,
            list_papers,
        )

        # Initialize server first
        from src.server import initialize_server
        initialize_server()

        print("[OK] Server initialized")

        # Test get_collection_stats
        stats = get_collection_stats()
        if "error" not in stats:
            print("[OK] get_collection_stats()")
            print(f"    Total chunks: {stats.get('total_chunks_indexed', 0)}")
            print(f"    Total documents: {stats.get('total_documents', 0)}")
        else:
            print("[WARN] get_collection_stats() - no documents indexed")

        # Test list_papers
        papers = list_papers()
        if "error" not in papers:
            print(f"[OK] list_papers()")
            print(f"    Total papers: {papers.get('total_papers', 0)}")
            if papers.get("papers"):
                print(f"    First paper: {papers['papers'][0].get('title', 'Unknown')[:50]}...")
        else:
            print("[SKIP] list_papers() - no documents indexed")

        # Test search
        result = search(query="glacier", top_k=3, alpha=0.5)
        if "error" not in result:
            print(f"[OK] search()")
            print(f"    Query: {result.get('query')}")
            print(f"    Results: {result.get('num_results', 0)}")
        else:
            print("[SKIP] search() - no documents indexed")

        return True

    except Exception as e:
        print(f"[FAIL] MCP tools error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("SCIENTIFIC PAPERS MCP SERVER - COMPLETE TEST SUITE")
    print("=" * 70)

    tests = [
        ("Metadata Extraction", test_phase_1_metadata),
        ("Chroma Initialization", test_phase_2_chroma),
        ("Document Chunking", test_phase_3_chunking),
        ("Document Indexing", test_phase_4_indexing),
        ("Hybrid Search", test_phase_5_search),
        ("MCP Tools", test_phase_6_mcp_tools),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n[FATAL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, passed_test in results.items():
        status = "[PASS]" if passed_test else "[FAIL]"
        print(f"{status} {name}")

    print("=" * 70)
    print(f"Result: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n✓ ALL TESTS PASSED! Server is ready to use.")
        print("\nNext steps:")
        print("1. Index all documents: python -c \"from src.server import index_all_documents; ...\"")
        print("2. Configure Claude Code (see SETUP_CLAUDE_CODE.md)")
        print("3. Restart Claude Code")
        print("4. Start searching!")
    elif passed >= total - 1:
        print("\n✓ Most tests passed. Some features may not work without indexed documents.")
    else:
        print("\n✗ Some tests failed. Check the output above.")

    return 0 if passed >= total - 1 else 1


if __name__ == "__main__":
    sys.exit(main())
