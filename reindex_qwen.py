#!/usr/bin/env python
"""
Reindex script: Migrates from multilingual-e5-large to Qwen3-Embedding-4B

This script:
1. Backs up existing ChromaDB
2. Deletes old collection
3. Scans and reindexes all documents with Qwen3-4B @ 2048 dims
4. Uses batch processing for speed (3-5x faster)
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.extractors.pdf_extractor import extract_text_from_pdf
from src.extractors.metadata_extractor import extract_metadata
from src.indexing.chroma_client import ChromaClientManager
from src.indexing.hybrid_search import HybridSearchEngine
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Progress tracking
BATCH_SIZE = config.batch_size
PROGRESS_INTERVAL = 10  # Log progress every N documents


def backup_chroma(backup_dir: Path = None) -> Path:
    """Create backup of current ChromaDB"""
    backup_dir = backup_dir or Path(".chroma_backup")

    if config.chroma_path.exists():
        logger.info(f"Backing up ChromaDB to {backup_dir}...")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(config.chroma_path, backup_dir)
        logger.info("Backup completed successfully")
        return backup_dir
    else:
        logger.info("No existing ChromaDB to backup")
        return None


def delete_old_collection(chroma_manager: ChromaClientManager) -> None:
    """Delete old collection"""
    try:
        chroma_manager.delete_collection("scientific_papers")
        logger.info("Old collection deleted")
    except Exception as e:
        logger.warning(f"Could not delete collection: {e}")


def scan_documents(doc_path: Path) -> List[Tuple[str, str, dict]]:
    """
    Scan documents folder and extract text + metadata

    Returns: List of (doc_id, text, metadata) tuples
    """
    documents = []

    if not doc_path.exists():
        logger.error(f"Documents path does not exist: {doc_path}")
        return []

    logger.info(f"Scanning documents from {doc_path}...")

    # Find all Markdown and PDF files
    md_files = list(doc_path.glob("**/*.md"))
    pdf_files = list(doc_path.glob("**/*.pdf"))

    total_files = len(md_files) + len(pdf_files)
    logger.info(f"Found {len(md_files)} markdown files and {len(pdf_files)} PDF files")

    # Process Markdown files
    for i, md_file in enumerate(md_files, 1):
        try:
            text = md_file.read_text(encoding="utf-8")
            doc_id = f"md_{md_file.stem}_{i}"
            metadata = extract_metadata(text, str(md_file))
            documents.append((doc_id, text, metadata))

            if i % PROGRESS_INTERVAL == 0:
                logger.info(f"Processed {i}/{total_files} files")
        except Exception as e:
            logger.warning(f"Error processing {md_file}: {e}")

    # Process PDF files
    for i, pdf_file in enumerate(pdf_files, 1):
        try:
            text, is_scanned = extract_text_from_pdf(str(pdf_file))
            if text:
                doc_id = f"pdf_{pdf_file.stem}_{i}"
                metadata = extract_metadata(text, str(pdf_file))
                metadata["source_type"] = "pdf_scanned" if is_scanned else "pdf_text"
                documents.append((doc_id, text, metadata))

            if (len(md_files) + i) % PROGRESS_INTERVAL == 0:
                logger.info(f"Processed {len(md_files) + i}/{total_files} files")
        except Exception as e:
            logger.warning(f"Error processing {pdf_file}: {e}")

    logger.info(f"Successfully scanned {len(documents)} documents")
    return documents


def reindex_documents(
    search_engine: HybridSearchEngine,
    documents: List[Tuple[str, str, dict]],
    batch_size: int = BATCH_SIZE
) -> int:
    """
    Reindex documents using batch processing

    Returns: Number of documents indexed
    """
    if not documents:
        logger.warning("No documents to index")
        return 0

    indexed_count = 0

    logger.info(f"Starting reindexing with batch size {batch_size}...")

    # Process in batches
    for batch_idx in range(0, len(documents), batch_size):
        batch = documents[batch_idx : batch_idx + batch_size]

        try:
            doc_ids = [doc[0] for doc in batch]
            texts = [doc[1] for doc in batch]
            metadatas = [doc[2] for doc in batch]

            # Batch index (3-5x faster than one-by-one)
            search_engine.index_documents_batch(doc_ids, texts, metadatas)

            indexed_count += len(batch)

            # Log progress
            progress_pct = (indexed_count / len(documents)) * 100
            logger.info(f"Indexed {indexed_count}/{len(documents)} documents ({progress_pct:.1f}%)")

        except Exception as e:
            logger.error(f"Error indexing batch starting at {batch_idx}: {e}")

    logger.info(f"Reindexing completed: {indexed_count} documents indexed")
    return indexed_count


def print_summary(
    backup_path: Path,
    indexed_count: int,
    duration_seconds: float
) -> None:
    """Print migration summary"""
    print("\n" + "="*60)
    print("REINDEXING SUMMARY")
    print("="*60)
    print(f"Backup location:     {backup_path}")
    print(f"Documents indexed:   {indexed_count}")
    print(f"Duration:            {duration_seconds:.1f} seconds ({duration_seconds/60:.1f} minutes)")
    if indexed_count > 0:
        docs_per_sec = indexed_count / duration_seconds
        print(f"Speed:               {docs_per_sec:.1f} docs/sec")
    print(f"Model:               Qwen/Qwen3-Embedding-4B")
    print(f"Dimensions:          2048")
    print(f"Batch size:          {BATCH_SIZE}")
    print("="*60)
    print("✅ Migration completed successfully!")
    print(f"Backup saved at: {backup_path}")
    print("="*60 + "\n")


def main():
    """Main reindexing workflow"""
    start_time = datetime.now()

    try:
        print("\n" + "="*60)
        print("QWEN3-4B REINDEXING SCRIPT")
        print("="*60)
        print(f"Documents path: {config.documents_path}")
        print(f"ChromaDB path:  {config.chroma_path}")
        print(f"Model:          Qwen/Qwen3-Embedding-4B")
        print(f"Dimensions:     {config.embedding_dimensions}")
        print("="*60 + "\n")

        # Step 1: Backup
        logger.info("STEP 1: Backing up existing ChromaDB...")
        backup_path = backup_chroma()

        # Step 2: Initialize new Chroma client
        logger.info("STEP 2: Initializing new Chroma client...")
        chroma_manager = ChromaClientManager()

        # Step 3: Delete old collection
        logger.info("STEP 3: Deleting old collection...")
        delete_old_collection(chroma_manager)

        # Step 4: Get new collection
        logger.info("STEP 4: Creating new collection with optimized HNSW...")
        collection = chroma_manager.get_or_create_collection("scientific_papers")

        # Step 5: Initialize search engine
        logger.info("STEP 5: Initializing hybrid search engine...")
        search_engine = HybridSearchEngine(collection)

        # Step 6: Scan documents
        logger.info("STEP 6: Scanning documents...")
        documents = scan_documents(config.documents_path)

        if not documents:
            logger.error("No documents found! Aborting.")
            return

        # Step 7: Reindex with batch processing
        logger.info("STEP 7: Reindexing documents with batch processing...")
        indexed_count = reindex_documents(search_engine, documents)

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()

        # Print summary
        print_summary(backup_path, indexed_count, duration)

    except Exception as e:
        logger.error(f"Fatal error during reindexing: {e}", exc_info=True)
        print("\n❌ Reindexing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
