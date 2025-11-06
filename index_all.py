#!/usr/bin/env python
"""
Index all scientific papers into the MCP server

This script will:
1. Load all markdown files from the documents directory
2. Extract metadata for each document
3. Chunk documents intelligently
4. Generate embeddings and index in Chroma + BM25
5. Report statistics
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.extractors.metadata_extractor import extract_metadata_from_file
from src.indexing.chroma_client import ChromaClientManager
from src.indexing.hybrid_search import HybridSearchEngine
from src.indexing.chunker import ScientificPaperChunker
from src.utils.logger import setup_logger

logger = setup_logger(__name__, level="INFO")


def sanitize_metadata(metadata, section=""):
    """Convert None values to defaults for Chroma compatibility"""
    return {
        "filename": metadata.filename or "unknown",
        "title": metadata.title or "Unknown Title",
        "authors": ", ".join(metadata.authors) if metadata.authors else "Unknown",
        "year": str(metadata.year) if metadata.year else "0",
        "journal": metadata.journal or "Unknown",
        "tags": ", ".join(metadata.tags) if metadata.tags else "",
        "instruments": ", ".join(metadata.instruments) if metadata.instruments else "",
        "section": section or "unknown",
    }


def index_all_documents():
    """Index all markdown documents"""

    print("\n" + "=" * 80)
    print("INDEXING ALL SCIENTIFIC PAPERS")
    print("=" * 80)

    # Initialize components
    print("\nInitializing components...")
    try:
        manager = ChromaClientManager(
            chroma_path=config.chroma_path,
            embedding_model=config.embedding_model,
        )
        collection = manager.get_or_create_collection("scientific_papers")

        search_engine = HybridSearchEngine(
            collection, embedding_model=config.embedding_model
        )

        chunker = ScientificPaperChunker(
            max_chunk_size=config.max_chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        print("[OK] Components initialized")
        sys.stdout.flush()
    except Exception as e:
        print(f"[FAIL] Initialization error: {e}")
        sys.stdout.flush()
        return False

    # Get all markdown files
    print("\nSearching for markdown files...")
    sys.stdout.flush()
    docs_path = Path(config.documents_path)
    markdown_files = sorted(docs_path.glob("*.md"))

    if not markdown_files:
        print(f"[FAIL] No markdown files found in {docs_path}")
        sys.stdout.flush()
        return False

    print(f"\n[OK] Found {len(markdown_files)} markdown files to index")
    print("=" * 80)
    sys.stdout.flush()

    # Index statistics
    total_files = 0
    total_chunks = 0
    total_errors = 0
    indexed_files = []
    failed_files = []

    # Index each document
    for idx, file_path in enumerate(markdown_files, 1):
        print(f"\n[{idx}/{len(markdown_files)}] Processing: {file_path.name}")

        try:
            # Extract metadata
            metadata = extract_metadata_from_file(file_path)
            if not metadata:
                print(f"    [WARN] Could not extract metadata, skipping")
                failed_files.append(file_path.name)
                total_errors += 1
                continue

            # Read document
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            # Chunk document
            chunks = chunker.chunk_document(
                text, document_id=metadata.filename, keep_abstract=True
            )

            if not chunks:
                print(f"    [WARN] No chunks generated, skipping")
                failed_files.append(file_path.name)
                total_errors += 1
                continue

            # Index chunks
            indexed_chunks = 0
            for chunk in chunks:
                try:
                    search_engine.index_document(
                        doc_id=chunk.chunk_id,
                        text=chunk.text,
                        metadata=sanitize_metadata(metadata, chunk.section),
                    )
                    indexed_chunks += 1
                except Exception as e:
                    logger.warning(f"Failed to index chunk {chunk.chunk_id}: {e}")
                    continue

            if indexed_chunks > 0:
                print(f"    [OK] Indexed {indexed_chunks} chunks")
                print(f"         Title: {metadata.title[:60]}...")
                print(f"         Year: {metadata.year}")
                print(f"         Authors: {len(metadata.authors)}")
                print(f"         Tags: {', '.join(metadata.tags) if metadata.tags else 'None'}")

                total_files += 1
                total_chunks += indexed_chunks
                indexed_files.append(file_path.name)
            else:
                print(f"    [FAIL] No chunks were indexed")
                failed_files.append(file_path.name)
                total_errors += 1

        except UnicodeDecodeError:
            print(f"    [FAIL] Encoding error, skipping")
            failed_files.append(file_path.name)
            total_errors += 1
        except Exception as e:
            print(f"    [FAIL] Error: {e}")
            failed_files.append(file_path.name)
            total_errors += 1

    # Final report
    print("\n" + "=" * 80)
    print("INDEXING COMPLETE - SUMMARY REPORT")
    print("=" * 80)

    print(f"\nResults:")
    print(f"  Total markdown files: {len(markdown_files)}")
    print(f"  Successfully indexed: {total_files}")
    print(f"  Failed: {total_errors}")
    print(f"  Total chunks created: {total_chunks}")

    # Collection stats
    collection_count = collection.count()
    print(f"\nCollection Statistics:")
    print(f"  Total documents in Chroma: {collection_count}")
    print(f"  BM25 index size: {len(search_engine.doc_ids)}")

    if failed_files:
        print(f"\nFailed files:")
        for fname in failed_files:
            print(f"  - {fname}")

    if total_files > 0:
        print(f"\nIndexed files (first 5):")
        for fname in indexed_files[:5]:
            print(f"  - {fname}")
        if len(indexed_files) > 5:
            print(f"  ... and {len(indexed_files) - 5} more")

    print("\n" + "=" * 80)

    if total_files > 0:
        print(f"\n[SUCCESS] Indexed {total_files} documents with {total_chunks} chunks")
        print("\nYour MCP server is now ready to use:")
        print("  1. Configure Claude Code (see SETUP_CLAUDE_CODE.md)")
        print("  2. Restart Claude Code")
        print("  3. Start searching!")
        print("\nExample queries:")
        print("  - 'Find articles about glacier albedo feedback'")
        print("  - 'Search for black carbon on glaciers'")
        print("  - 'Papers by Ren et al'")
        return True
    else:
        print(f"\n[FAILED] No documents were indexed.")
        print("Check the errors above and try again.")
        return False


def main():
    """Main entry point"""
    try:
        success = index_all_documents()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nIndexing interrupted by user")
        return 1
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
