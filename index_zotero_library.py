#!/usr/bin/env python3
"""
Index Zotero Library - Main indexing script

Usage:
    python index_zotero_library.py [options]

Options:
    --force-rebuild      Force complete reindexing (ignore state)
    --limit N            Limit to N documents (for testing)
    --no-dedup           Disable deduplication
    --clear-state        Clear all indexing state before starting

Examples:
    # Full indexing (incremental)
    python index_zotero_library.py

    # Force complete reindex
    python index_zotero_library.py --force-rebuild

    # Test with first 10 documents
    python index_zotero_library.py --limit 10

    # Fresh start (clear state and reindex)
    python index_zotero_library.py --clear-state --force-rebuild
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import chromadb
from src.config import config
from src.indexing.hybrid_search import HybridSearchEngine
from src.indexing.zotero_indexer import ZoteroLibraryIndexer
from src.indexing.indexing_state import IndexingStateManager
from src.indexing.deduplicator import DocumentDeduplicator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Index Zotero library with incremental updates and deduplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force complete reindexing (ignore state)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N documents (for testing)",
    )

    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication",
    )

    parser.add_argument(
        "--clear-state",
        action="store_true",
        help="Clear all indexing state before starting",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.batch_indexing_size,
        help=f"Batch size for indexing (default: {config.batch_indexing_size})",
    )

    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="ChromaDB collection name (default: auto-detect from embedding model)",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        choices=["jina", "voyage", "local"],
        default=None,
        help="Force specific embedding model (overrides config)",
    )

    return parser.parse_args()


def main():
    """Main indexing workflow"""
    args = parse_args()

    # Override config based on --embedding-model argument
    if args.embedding_model:
        if args.embedding_model == "voyage":
            config.use_voyage_api = True
            config.use_jina_api = False
        elif args.embedding_model == "jina":
            config.use_jina_api = True
            config.use_voyage_api = False
        else:  # local
            config.use_jina_api = False
            config.use_voyage_api = False

    # Determine collection name
    if args.collection_name:
        collection_name = args.collection_name
    else:
        # Auto-detect from embedding model
        if config.use_voyage_api:
            collection_name = "scientific_papers_voyage"
        elif config.use_jina_api:
            collection_name = "scientific_papers_jina"
        else:
            collection_name = "scientific_papers_local"

    logger.info("=" * 70)
    logger.info("Zotero Library Indexer")
    logger.info("=" * 70)
    logger.info(f"Library path: {config.documents_path}")
    logger.info(f"ChromaDB path: {config.chroma_path}")
    logger.info(f"Collection name: {collection_name}")
    logger.info(f"Embedding model: {config.embedding_model}")
    logger.info(f"Incremental indexing: {config.enable_incremental_indexing}")
    logger.info(f"Deduplication: {config.enable_deduplication and not args.no_dedup}")
    logger.info(f"Force rebuild: {args.force_rebuild}")
    logger.info(f"Batch size: {args.batch_size}")
    if args.limit:
        logger.info(f"Limit: {args.limit} documents")
    logger.info("=" * 70)

    try:
        # Validate paths
        config.validate_paths()

        # Initialize ChromaDB
        logger.info("Initializing ChromaDB...")
        chroma_client = chromadb.PersistentClient(path=str(config.chroma_path))

        # Get or create collection
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Zotero library indexed papers",
                "embedding_provider": "voyage" if config.use_voyage_api else "jina" if config.use_jina_api else "local",
            },
        )

        logger.info(f"ChromaDB collection loaded (existing docs: {collection.count()})")

        # Initialize search engine
        logger.info(f"Initializing hybrid search engine (model: {config.embedding_model})...")
        search_engine = HybridSearchEngine(
            chroma_collection=collection, embedding_model=config.embedding_model
        )

        # Initialize state manager
        state_manager = IndexingStateManager()

        if args.clear_state:
            logger.warning("Clearing all indexing state...")
            state_manager.clear_state()

        # Initialize deduplicator
        deduplicator = None if args.no_dedup else DocumentDeduplicator()

        # Initialize indexer
        indexer = ZoteroLibraryIndexer(
            search_engine=search_engine,
            library_path=config.documents_path,
            state_manager=state_manager,
            deduplicator=deduplicator,
        )

        # Run indexing
        logger.info("")
        logger.info("Starting indexing process...")
        logger.info("")

        statistics = indexer.index_library(
            force_rebuild=args.force_rebuild,
            limit=args.limit,
            save_state=True,
        )

        # Print final statistics
        logger.info("")
        logger.info("=" * 70)
        logger.info("Indexing Statistics")
        logger.info("=" * 70)
        logger.info(f"Total scanned:        {statistics['total_scanned']}")
        logger.info(f"Added:                {statistics['added']}")
        logger.info(f"Updated:              {statistics['updated']}")
        logger.info(f"Skipped (unchanged):  {statistics['skipped']}")
        logger.info(f"Duplicates removed:   {statistics['duplicates']}")
        logger.info(f"Errors:               {statistics['errors']}")

        if statistics['error_files']:
            logger.info("")
            logger.info(f"First {len(statistics['error_files'])} error files:")
            for error_file in statistics['error_files']:
                logger.info(f"  - {error_file}")

        logger.info("=" * 70)

        # Show state statistics
        state_stats = state_manager.get_statistics()
        logger.info("")
        logger.info("State Information")
        logger.info("-" * 70)
        logger.info(f"Total indexed files:  {state_stats['total_indexed']}")
        if state_stats['last_full_reindex']:
            logger.info(f"Last full reindex:    {state_stats['last_full_reindex']}")
        if state_stats['last_incremental_update']:
            logger.info(f"Last update:          {state_stats['last_incremental_update']}")
        logger.info("=" * 70)

        logger.info("")
        logger.info("[OK] Indexing completed successfully!")
        logger.info("")

        return 0

    except KeyboardInterrupt:
        logger.warning("\n\nIndexing interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"\n\nError during indexing: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
