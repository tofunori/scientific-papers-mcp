#!/usr/bin/env python3
"""
Update Zotero Index - Fast incremental update script

This script only processes new/modified documents since the last indexing.
Perfect for regular updates to keep your index fresh without full reindexing.

Expected performance:
- No changes: ~5-10 seconds (just scanning)
- Few changes (1-10 docs): ~30 seconds - 2 minutes
- Many changes (50+ docs): ~5-10 minutes

Usage:
    python update_zotero_index.py [options]

Options:
    --limit N            Limit to N documents (for testing)
    --no-dedup           Disable deduplication
    --verbose            Show detailed logging

Examples:
    # Quick update (default)
    python update_zotero_index.py

    # Update with verbose logging
    python update_zotero_index.py --verbose

    # Test update with first 20 documents
    python update_zotero_index.py --limit 20
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
        description="Fast incremental update for Zotero library index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        "--verbose",
        action="store_true",
        help="Show detailed logging",
    )

    return parser.parse_args()


def main():
    """Main update workflow"""
    args = parse_args()

    # Set logging level
    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("Zotero Library - Quick Update")
    logger.info("=" * 70)
    logger.info(f"Library path: {config.documents_path}")
    logger.info(f"Incremental mode: ENABLED (only processes changed files)")
    logger.info(f"Deduplication: {config.enable_deduplication and not args.no_dedup}")
    if args.limit:
        logger.info(f"Limit: {args.limit} documents")
    logger.info("=" * 70)

    try:
        # Validate paths
        config.validate_paths()

        # Check if state exists
        state_manager = IndexingStateManager()
        state_stats = state_manager.get_statistics()

        if state_stats["total_indexed"] == 0:
            logger.warning("")
            logger.warning("⚠ No previous indexing state found!")
            logger.warning("Please run a full index first:")
            logger.warning("  python index_zotero_library.py")
            logger.warning("")
            return 1

        logger.info(f"\nPrevious state: {state_stats['total_indexed']} indexed files")
        if state_stats["last_incremental_update"]:
            logger.info(f"Last update: {state_stats['last_incremental_update']}")
        logger.info("")

        # Initialize ChromaDB
        logger.info("Loading ChromaDB...")
        chroma_client = chromadb.PersistentClient(path=str(config.chroma_path))

        collection = chroma_client.get_or_create_collection(
            name="scientific_papers",
            metadata={"description": "Zotero library indexed papers"},
        )

        logger.info(f"ChromaDB loaded ({collection.count()} documents)")

        # Initialize search engine
        search_engine = HybridSearchEngine(
            chroma_collection=collection, embedding_model=config.embedding_model
        )

        # Initialize deduplicator
        deduplicator = None if args.no_dedup else DocumentDeduplicator()

        # Initialize indexer
        indexer = ZoteroLibraryIndexer(
            search_engine=search_engine,
            library_path=config.documents_path,
            state_manager=state_manager,
            deduplicator=deduplicator,
        )

        # Run incremental update
        logger.info("Scanning for new/modified documents...")
        logger.info("")

        statistics = indexer.index_library(
            force_rebuild=False,  # ALWAYS incremental for update script
            limit=args.limit,
            save_state=True,
        )

        # Print summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("Update Summary")
        logger.info("=" * 70)

        total_processed = statistics["added"] + statistics["updated"]
        if total_processed == 0:
            logger.info("[OK] No changes detected - index is up to date!")
        else:
            logger.info(f"Added:                {statistics['added']}")
            logger.info(f"Updated:              {statistics['updated']}")
            if statistics["duplicates"] > 0:
                logger.info(f"Duplicates removed:   {statistics['duplicates']}")
            if statistics["errors"] > 0:
                logger.info(f"Errors:               {statistics['errors']}")

        logger.info(f"Skipped (unchanged):  {statistics['skipped']}")
        logger.info("=" * 70)

        if total_processed > 0:
            logger.info(f"\n[OK] Updated {total_processed} documents successfully!\n")
        else:
            logger.info("\n[OK] Index is fresh - no updates needed!\n")

        return 0

    except KeyboardInterrupt:
        logger.warning("\n\nUpdate interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"\n\nError during update: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
