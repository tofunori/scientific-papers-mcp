"""Quick test script for the MCP server"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.extractors.metadata_extractor import extract_metadata_from_file
from src.indexing.chroma_client import initialize_chroma
from src.indexing.hybrid_search import HybridSearchEngine
from src.indexing.chunker import ScientificPaperChunker
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_metadata_extraction():
    """Test metadata extraction from a markdown file"""
    logger.info("\n=== Testing Metadata Extraction ===")

    docs_path = Path(config.documents_path)
    markdown_files = list(docs_path.glob("*.md"))

    if not markdown_files:
        logger.warning(f"No markdown files found in {docs_path}")
        return

    # Test first file
    test_file = markdown_files[0]
    logger.info(f"Testing extraction from: {test_file.name}")

    metadata = extract_metadata_from_file(test_file)
    if metadata:
        logger.info(f"✓ Title: {metadata.title}")
        logger.info(f"✓ Authors: {metadata.authors}")
        logger.info(f"✓ Year: {metadata.year}")
        logger.info(f"✓ Tags: {metadata.tags}")
        logger.info(f"✓ Instruments: {metadata.instruments}")
        logger.info("✓ Metadata extraction working!")
    else:
        logger.error("✗ Failed to extract metadata")


def test_chroma_initialization():
    """Test Chroma initialization"""
    logger.info("\n=== Testing Chroma Initialization ===")

    try:
        collection = initialize_chroma(
            chroma_path=config.chroma_path,
            embedding_model=config.embedding_model,
        )
        logger.info(f"✓ Chroma collection initialized")
        logger.info(f"✓ Collection count: {collection.count()}")
    except Exception as e:
        logger.error(f"✗ Chroma initialization failed: {e}")


def test_chunking():
    """Test document chunking"""
    logger.info("\n=== Testing Document Chunking ===")

    docs_path = Path(config.documents_path)
    markdown_files = list(docs_path.glob("*.md"))

    if not markdown_files:
        logger.warning(f"No markdown files found in {docs_path}")
        return

    test_file = markdown_files[0]

    try:
        with open(test_file, "r", encoding="utf-8") as f:
            text = f.read()

        chunker = ScientificPaperChunker(
            max_chunk_size=config.max_chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        chunks = chunker.chunk_document(text, document_id=test_file.stem)
        logger.info(f"✓ Chunked document into {len(chunks)} chunks")
        logger.info(f"✓ First chunk size: {len(chunks[0].text)} characters")
    except Exception as e:
        logger.error(f"✗ Chunking failed: {e}")


def test_hybrid_search():
    """Test hybrid search engine"""
    logger.info("\n=== Testing Hybrid Search ===")

    try:
        # Initialize
        collection = initialize_chroma(
            chroma_path=config.chroma_path,
            embedding_model=config.embedding_model,
        )

        search_engine = HybridSearchEngine(
            collection, embedding_model=config.embedding_model
        )

        # Check if we have documents indexed
        if not search_engine.doc_ids:
            logger.warning("No documents indexed yet. Run index_documents first.")
            logger.info("To index documents, use: search_engine.index_document(...)")
            return

        # Test search
        query = "glacier albedo"
        logger.info(f"Testing search with query: '{query}'")

        doc_ids, scores, metadatas = search_engine.search(
            query=query, top_k=5, alpha=0.5
        )

        if doc_ids:
            logger.info(f"✓ Found {len(doc_ids)} results")
            for i, (doc_id, score) in enumerate(zip(doc_ids, scores)):
                logger.info(f"  {i+1}. {doc_id}: {score:.4f}")
        else:
            logger.info("No results found (collection may be empty)")

    except Exception as e:
        logger.error(f"✗ Hybrid search failed: {e}")


def main():
    """Run all tests"""
    logger.info("=" * 50)
    logger.info("Scientific Papers MCP - Quick Test")
    logger.info("=" * 50)

    test_metadata_extraction()
    test_chroma_initialization()
    test_chunking()
    test_hybrid_search()

    logger.info("\n" + "=" * 50)
    logger.info("Tests completed!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
