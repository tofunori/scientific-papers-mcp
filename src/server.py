"""Scientific Papers MCP Server - Main entry point"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging
import json

from fastmcp import FastMCP

from .config import config
from .utils.logger import setup_logger
from .extractors.metadata_extractor import extract_metadata_from_file
from .indexing.chroma_client import initialize_chroma
from .indexing.hybrid_search import HybridSearchEngine
from .indexing.chunker import ScientificPaperChunker

# Setup logging
logger = setup_logger(__name__, level=config.log_level)

# Initialize FastMCP
mcp = FastMCP(
    name="Scientific Papers Search",
    instructions="""
    Search and retrieve information from a collection of glacier albedo research papers.

    Hybrid search combining semantic understanding with precise keyword matching.

    Available operations:
    - search: Perform hybrid semantic + keyword search
    - search_by_author: Find papers by author name
    - search_by_year: Find papers by publication year
    - get_metadata: Get detailed metadata for a paper
    - list_papers: List all indexed papers
    - index_documents: Index markdown documents from a directory
    """,
)

# Global search engine instance
search_engine: Optional[HybridSearchEngine] = None
chunker: Optional[ScientificPaperChunker] = None


def initialize_server() -> None:
    """Initialize the MCP server and search engine"""
    global search_engine, chunker

    try:
        logger.info("Initializing Scientific Papers MCP Server")

        # Validate config
        config.validate_paths()

        # Initialize Chroma
        logger.info(f"Initializing Chroma from {config.chroma_path}")
        collection = initialize_chroma(
            chroma_path=config.chroma_path,
            embedding_model=config.embedding_model,
        )

        # Initialize search engine
        search_engine = HybridSearchEngine(collection, config.embedding_model)

        # Initialize chunker
        chunker = ScientificPaperChunker(
            max_chunk_size=config.max_chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        logger.info("Server initialized successfully")

        # Auto-index if enabled
        if config.auto_index_on_start:
            logger.info("Auto-indexing documents...")
            index_all_documents(config.documents_path)

    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise


def index_all_documents(documents_path: Path | str) -> Dict:
    """Index all markdown documents in a directory"""
    documents_path = Path(documents_path)

    if not documents_path.exists():
        logger.error(f"Documents path does not exist: {documents_path}")
        return {"status": "error", "message": f"Path not found: {documents_path}"}

    try:
        markdown_files = list(documents_path.glob("*.md"))
        logger.info(f"Found {len(markdown_files)} markdown files")

        indexed_count = 0
        for file_path in markdown_files:
            try:
                # Extract metadata
                metadata = extract_metadata_from_file(file_path)
                if not metadata:
                    logger.warning(f"Failed to extract metadata from {file_path.name}")
                    continue

                # Read document
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                # Chunk document
                chunks = chunker.chunk_document(
                    text, document_id=metadata.filename, keep_abstract=True
                )

                # Index chunks
                for chunk in chunks:
                    search_engine.index_document(
                        doc_id=chunk.chunk_id,
                        text=chunk.text,
                        metadata={
                            "filename": metadata.filename,
                            "title": metadata.title,
                            "authors": ", ".join(metadata.authors),
                            "year": metadata.year,
                            "tags": ", ".join(metadata.tags),
                            "instruments": ", ".join(metadata.instruments),
                            "section": chunk.section,
                        },
                    )

                indexed_count += 1
                logger.info(f"Indexed {metadata.filename} ({len(chunks)} chunks)")

            except Exception as e:
                logger.error(f"Error indexing {file_path.name}: {e}")
                continue

        return {
            "status": "success",
            "indexed_files": indexed_count,
            "total_files": len(markdown_files),
        }

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool()
def search(
    query: str,
    top_k: int = 10,
    alpha: float = 0.5,
) -> Dict:
    """
    Perform hybrid semantic + keyword search across research papers.

    Args:
        query: Search query (e.g., "glacier albedo feedback mechanisms")
        top_k: Number of results to return (default: 10)
        alpha: Balance between semantic (1.0) and keyword (0.0) search
               (default: 0.5 for balanced search)

    Returns:
        List of matching papers with relevance scores and metadata
    """
    if not search_engine:
        return {"error": "Search engine not initialized"}

    try:
        logger.info(f"Search: '{query}' (top_k={top_k}, alpha={alpha})")

        doc_ids, scores, metadatas, documents = search_engine.search(
            query=query, top_k=top_k, alpha=alpha
        )

        results = []
        for doc_id, score, metadata, document in zip(doc_ids, scores, metadatas, documents):
            results.append(
                {
                    "doc_id": doc_id,
                    "relevance_score": float(score),
                    "title": metadata.get("title", "Unknown"),
                    "authors": metadata.get("authors", "Unknown"),
                    "year": metadata.get("year"),
                    "section": metadata.get("section", "unknown"),
                    "tags": metadata.get("tags", "").split(", ") if metadata.get("tags") else [],
                    "instruments": metadata.get("instruments", "").split(", ") if metadata.get("instruments") else [],
                    "text": document,
                }
            )

        return {
            "query": query,
            "num_results": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"error": str(e)}


@mcp.tool()
def search_by_author(author_name: str) -> Dict:
    """
    Find all papers by a specific author.

    Args:
        author_name: Name of the author to search for

    Returns:
        List of papers by that author
    """
    if not search_engine:
        return {"error": "Search engine not initialized"}

    try:
        logger.info(f"Search by author: {author_name}")

        # Search using the author name
        doc_ids, scores, metadatas, documents = search_engine.search(
            query=f"author:{author_name}",
            top_k=100,
            alpha=0.2,  # Favor keyword search for exact names
        )

        # Filter by author name in metadata
        results = []
        for doc_id, score, metadata, document in zip(doc_ids, scores, metadatas, documents):
            authors = metadata.get("authors", "")
            if author_name.lower() in authors.lower():
                results.append(
                    {
                        "doc_id": doc_id,
                        "title": metadata.get("title"),
                        "authors": authors,
                        "year": metadata.get("year"),
                    }
                )

        return {
            "author": author_name,
            "num_results": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Author search error: {e}")
        return {"error": str(e)}


@mcp.tool()
def search_by_year(year: int, start_year: Optional[int] = None) -> Dict:
    """
    Find papers by publication year or year range.

    Args:
        year: Specific year or end year for range
        start_year: Optional start year for range search

    Returns:
        List of papers from specified year(s)
    """
    if not search_engine:
        return {"error": "Search engine not initialized"}

    try:
        # Get all documents and filter by year
        all_ids = search_engine.doc_ids
        results = []

        for doc_id in all_ids:
            metadata = search_engine.get_metadata(doc_id)
            doc_year = metadata.get("year")

            if start_year:
                # Range search
                if doc_year and start_year <= doc_year <= year:
                    results.append(
                        {
                            "doc_id": doc_id,
                            "title": metadata.get("title"),
                            "authors": metadata.get("authors"),
                            "year": doc_year,
                        }
                    )
            else:
                # Exact year
                if doc_year == year:
                    results.append(
                        {
                            "doc_id": doc_id,
                            "title": metadata.get("title"),
                            "authors": metadata.get("authors"),
                            "year": doc_year,
                        }
                    )

        return {
            "year_range": f"{start_year}-{year}" if start_year else str(year),
            "num_results": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Year search error: {e}")
        return {"error": str(e)}


@mcp.tool()
def get_metadata(doc_id: str) -> Dict:
    """
    Get complete metadata for a specific paper.

    Args:
        doc_id: Document identifier

    Returns:
        Complete metadata for the document
    """
    if not search_engine:
        return {"error": "Search engine not initialized"}

    try:
        metadata = search_engine.get_metadata(doc_id)
        if not metadata:
            return {"error": f"Document not found: {doc_id}"}

        return {
            "doc_id": doc_id,
            "metadata": metadata,
        }
    except Exception as e:
        logger.error(f"Metadata retrieval error: {e}")
        return {"error": str(e)}


@mcp.tool()
def list_papers() -> Dict:
    """
    List all indexed papers with their metadata.

    Returns:
        List of all papers in the collection
    """
    if not search_engine:
        return {"error": "Search engine not initialized"}

    try:
        papers = []
        unique_docs = set()

        for doc_id in search_engine.doc_ids:
            # Extract filename from doc_id (before first dash if chunked)
            filename = doc_id.split("-")[0] if "-" in doc_id else doc_id

            if filename not in unique_docs:
                unique_docs.add(filename)
                metadata = search_engine.get_metadata(doc_id)
                papers.append(
                    {
                        "filename": metadata.get("filename", filename),
                        "title": metadata.get("title"),
                        "authors": metadata.get("authors"),
                        "year": metadata.get("year"),
                        "tags": metadata.get("tags", "").split(", ") if metadata.get("tags") else [],
                        "instruments": metadata.get("instruments", "").split(", ") if metadata.get("instruments") else [],
                    }
                )

        return {
            "total_papers": len(papers),
            "papers": papers,
        }

    except Exception as e:
        logger.error(f"List papers error: {e}")
        return {"error": str(e)}


@mcp.tool()
def index_documents(directory_path: str) -> Dict:
    """
    Index all markdown documents from a directory.

    Args:
        directory_path: Path to directory containing markdown files

    Returns:
        Status of indexing operation
    """
    if not search_engine or not chunker:
        return {"error": "Search engine not initialized"}

    try:
        return index_all_documents(directory_path)
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        return {"error": str(e)}


@mcp.tool()
def get_collection_stats() -> Dict:
    """
    Get statistics about the indexed document collection.

    Returns:
        Statistics about indexed documents and chunks
    """
    if not search_engine:
        return {"error": "Search engine not initialized"}

    try:
        total_chunks = len(search_engine.doc_ids)
        unique_docs = len(set(doc_id.split("-")[0] for doc_id in search_engine.doc_ids))

        return {
            "total_chunks_indexed": total_chunks,
            "total_documents": unique_docs,
            "chroma_path": str(config.chroma_path),
            "embedding_model": config.embedding_model,
            "documents_path": str(config.documents_path),
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {"error": str(e)}


# ============================================================================
# Main entry point
# ============================================================================


def main():
    """Main entry point for the MCP server"""
    try:
        # Initialize server
        initialize_server()

        # Run MCP server (STDIO interface for Claude Desktop)
        logger.info("Starting MCP server...")
        mcp.run()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
