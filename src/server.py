"""Scientific Papers MCP Server - Main entry point"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging
import json
import re

from fastmcp import FastMCP

from .config import config
from .utils.logger import setup_logger
from .extractors.metadata_extractor import extract_metadata_from_file
from .extractors.pdf_extractor import extract_text_from_pdf, extract_metadata_from_pdf
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


def index_all_documents(documents_path: Path | str, recursive: bool = False) -> Dict:
    """
    Index all markdown and PDF documents in a directory.

    Args:
        documents_path: Path to directory containing documents
        recursive: If True, search in subdirectories recursively (for Zotero storage structure)
    """
    documents_path = Path(documents_path)

    if not documents_path.exists():
        logger.error(f"Documents path does not exist: {documents_path}")
        return {"status": "error", "message": f"Path not found: {documents_path}"}

    try:
        # Use recursive glob pattern if recursive=True
        if recursive:
            logger.info("Searching for documents recursively (including subdirectories)")
            markdown_files = list(documents_path.rglob("*.md"))
            pdf_files = list(documents_path.rglob("*.pdf"))
        else:
            markdown_files = list(documents_path.glob("*.md"))
            pdf_files = list(documents_path.glob("*.pdf"))

        all_files = markdown_files + pdf_files
        logger.info(f"Found {len(markdown_files)} markdown files and {len(pdf_files)} PDF files")

        indexed_count = 0
        markdown_count = 0
        pdf_count = 0

        for file_path in all_files:
            try:
                is_pdf = file_path.suffix.lower() == ".pdf"

                if is_pdf:
                    # Extract text from PDF
                    text, is_scanned = extract_text_from_pdf(file_path)
                    if not text.strip():
                        logger.warning(f"Failed to extract text from PDF {file_path.name}")
                        continue

                    # Extract metadata from PDF
                    pdf_metadata = extract_metadata_from_pdf(file_path)
                    metadata_filename = pdf_metadata.get("filename", file_path.name)
                    metadata_title = pdf_metadata.get("title", file_path.stem)
                    metadata_authors = pdf_metadata.get("authors", [])
                    metadata_year = pdf_metadata.get("year")

                    # Chunk PDF document
                    chunks = chunker.chunk_pdf_document(
                        text, document_id=metadata_filename
                    )

                    pdf_count += 1
                else:
                    # Extract metadata from Markdown
                    metadata = extract_metadata_from_file(file_path)
                    if not metadata:
                        logger.warning(f"Failed to extract metadata from {file_path.name}")
                        continue

                    # Read markdown document
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()

                    # Chunk markdown document
                    chunks = chunker.chunk_document(
                        text, document_id=metadata.filename, keep_abstract=True
                    )

                    metadata_filename = metadata.filename
                    metadata_title = metadata.title
                    metadata_authors = metadata.authors
                    metadata_year = metadata.year

                    markdown_count += 1

                # Index chunks
                for chunk in chunks:
                    # Clean metadata (ChromaDB doesn't accept None)
                    clean_metadata = {
                        "filename": metadata_filename or "unknown",
                        "title": metadata_title or "unknown",
                        "authors": ", ".join(metadata_authors) if metadata_authors else "unknown",
                        "year": str(metadata_year) if metadata_year else "unknown",
                        "section": chunk.section or "unknown",
                        "file_type": "pdf" if is_pdf else "markdown",
                    }

                    # Add tags and instruments only for markdown
                    if not is_pdf and 'metadata' in locals():
                        clean_metadata["tags"] = ", ".join(metadata.tags) if metadata.tags else "unknown"
                        clean_metadata["instruments"] = ", ".join(metadata.instruments) if metadata.instruments else "unknown"

                    search_engine.index_document(
                        doc_id=chunk.chunk_id,
                        text=chunk.text,
                        metadata=clean_metadata,
                    )

                indexed_count += 1
                file_type = "PDF" if is_pdf else "Markdown"
                logger.info(f"Indexed {file_type} {metadata_filename} ({len(chunks)} chunks)")

            except Exception as e:
                logger.error(f"Error indexing {file_path.name}: {e}")
                continue

        return {
            "status": "success",
            "indexed_files": indexed_count,
            "markdown_files": markdown_count,
            "pdf_files": pdf_count,
            "total_files": len(all_files),
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
    text_filter: Optional[str] = None,
) -> Dict:
    """
    Perform hybrid semantic + keyword search across research papers.

    Args:
        query: Search query (e.g., "glacier albedo feedback mechanisms")
        top_k: Number of results to return (default: 10)
        alpha: Balance between semantic (1.0) and keyword (0.0) search
               (default: 0.5 for balanced search)
        text_filter: Optional full text filter (JSON string)
                    Examples:
                    - Regex: '{"$regex": "MODIS.*MOD10A1"}'
                    - Contains: '{"$contains": "albedo"}'
                    - AND: '{"$and": [{"$contains": "Alaska"}, {"$contains": "glacier"}]}'
                    - OR: '{"$or": [{"$contains": "MODIS"}, {"$contains": "Sentinel"}]}'

    Returns:
        List of matching papers with relevance scores and metadata
    """
    if not search_engine:
        return {"error": "Search engine not initialized"}

    try:
        logger.info(f"Search: '{query}' (top_k={top_k}, alpha={alpha})")

        # Parse text_filter JSON if provided
        where_document = None
        if text_filter:
            try:
                where_document = json.loads(text_filter)
                logger.info(f"Text filter applied: {where_document}")
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid text_filter JSON: {e}")
                return {"error": f"Invalid text_filter JSON: {str(e)}"}

        # Use reranking search for improved precision (~35% better)
        doc_ids, scores, metadatas, documents = search_engine.search_with_reranking(
            query=query, top_k=top_k, alpha=alpha, where_document=where_document
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


def _truncate_text_for_prompt(text: str, max_chars: int) -> str:
    """Limit a passage to roughly `max_chars` while avoiding a hard cut in the middle of a word."""
    text = text.strip()
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return f"{truncated}..."


@mcp.tool()
def generate_rag_answer(
    question: str,
    top_k: int = 3,
    alpha: float = 0.7,
    text_filter: Optional[str] = None,
    metadata_filter: Optional[str] = None,
    context_limit_chars: int = 4000,
) -> Dict:
    """
    Build a RAG prompt using the reranked passages as the only context.

    This helper returns a cited context and a ready-to-send prompt that Claude (or any other
    generator) can use so that the response is evidence-backed and free of hallucinations.
    """
    if not search_engine:
        return {"error": "Search engine not initialized"}

    query = question.strip()
    if not query:
        return {"error": "Question must not be empty"}

    where_document = None
    if text_filter:
        try:
            where_document = json.loads(text_filter)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid text_filter JSON: {e}")
            return {"error": f"Invalid text_filter JSON: {str(e)}"}

    where_filter = None
    if metadata_filter:
        try:
            where_filter = json.loads(metadata_filter)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid metadata_filter JSON: {e}")
            return {"error": f"Invalid metadata_filter JSON: {str(e)}"}

    effective_top_k = max(1, top_k)

    try:
        doc_ids, scores, metadatas, documents = search_engine.search_with_reranking(
            query=query,
            top_k=effective_top_k,
            alpha=alpha,
            where_document=where_document,
            where_filter=where_filter,
        )

        if not doc_ids:
            return {"question": query, "top_k": effective_top_k, "context_passages": [], "prompt": ""}

        per_passage_limit = (
            1500
            if context_limit_chars <= 0
            else max(256, min(1500, int(context_limit_chars / effective_top_k)))
        )

        context_entries = []
        used_chars = 0
        for idx, (doc_id, score, metadata, document) in enumerate(
            zip(doc_ids, scores, metadatas, documents), start=1
        ):
            passage = _truncate_text_for_prompt(document, per_passage_limit)
            if not passage:
                continue

            source_label = f"[{idx}]"
            authors = metadata.get("authors", "Unknown")
            year = metadata.get("year", "Unknown")
            title = metadata.get("title", "Untitled")
            section = metadata.get("section", "unknown")

            entry = {
                "index": idx,
                "doc_id": doc_id,
                "title": title,
                "authors": authors,
                "year": year,
                "section": section,
                "score": float(score),
                "text": passage,
                "source_label": source_label,
            }

            used_chars += len(passage)
            context_entries.append(entry)

            if context_limit_chars > 0 and used_chars >= context_limit_chars:
                break

        if not context_entries:
            return {
                "question": query,
                "top_k": 0,
                "context_passages": [],
                "prompt": "",
                "note": "Aucun passage utilisable après la limitation de contexte.",
            }

        context_sections = []
        for entry in context_entries:
            source_info = f"{entry['title']} — {entry['authors']} ({entry['year']})"
            context_sections.append(
                f"{entry['source_label']} {entry['text']}\n— {source_info} | section: {entry['section']} | score: {entry['score']:.3f}"
            )

        context_payload = "\n\n".join(context_sections)
        prompt = (
            "Tu vas répondre uniquement à partir des passages numérotés ci-dessous et en citant "
            "chaque affirmation ([1], [2], ...).\n\n"
            f"Contexte :\n{context_payload}\n\n"
            f"Question : {query}\n\n"
            "Réponds en citant les sources correspondantes et ne réinvente rien."
        )

        return {
            "question": query,
            "top_k": len(context_entries),
            "alpha": alpha,
            "context_characters": used_chars,
            "context_passages": context_entries,
            "prompt": prompt,
            "note": "Utilise la valeur `prompt` avec ton modèle de génération et cite les passages en respectant les labels.",
        }

    except Exception as e:
        logger.error(f"RAG generation error: {e}")
        return {"error": str(e)}


@mcp.tool()
def search_fulltext(
    pattern: str,
    pattern_type: str = "contains",
    combine_with: Optional[List[str]] = None,
    combine_mode: str = "and",
    top_k: int = 10,
) -> Dict:
    """
    Full text search with simplified syntax for exact matching, regex, and boolean logic.

    Args:
        pattern: Text pattern to search for
        pattern_type: Type of search - 'contains', 'regex', or 'exact'
        combine_with: Additional patterns for AND/OR combinations (optional)
        combine_mode: 'and' or 'or' for combining multiple patterns
        top_k: Number of results to return (default: 10)

    Returns:
        List of documents matching the full text search criteria

    Examples:
        - search_fulltext("MODIS", "contains")
        - search_fulltext("MODIS.*MOD10A1", "regex")
        - search_fulltext("albedo", "contains", ["Alaska", "glacier"], "and")
        - search_fulltext("MODIS", "contains", ["Sentinel-2", "GCOM-C"], "or")
    """
    if not search_engine:
        return {"error": "Search engine not initialized"}

    try:
        logger.info(f"Full text search: pattern='{pattern}', type={pattern_type}, combine={combine_with}")

        # Build where_document filter
        if pattern_type == "contains":
            where_doc = {"$contains": pattern}
        elif pattern_type == "regex":
            where_doc = {"$regex": pattern}
        elif pattern_type == "exact":
            # Exact match using regex with anchors and escaped pattern
            where_doc = {"$regex": f"^{re.escape(pattern)}$"}
        else:
            return {"error": f"Invalid pattern_type: {pattern_type}. Use 'contains', 'regex', or 'exact'"}

        # Combine with additional patterns if provided
        if combine_with:
            conditions = [where_doc]
            for extra_pattern in combine_with:
                conditions.append({"$contains": extra_pattern})

            operator = "$and" if combine_mode == "and" else "$or"
            where_doc = {operator: conditions}

        # Execute search with low alpha to favor keyword matching
        doc_ids, scores, metadatas, documents = search_engine.search(
            query=pattern,  # Use pattern as semantic fallback query
            top_k=top_k,
            alpha=0.2,  # Favor keyword matching over semantic
            where_document=where_doc
        )

        # Format results
        results = []
        for doc_id, score, metadata, document in zip(doc_ids, scores, metadatas, documents):
            # Highlight matching text (first 300 chars)
            excerpt = document[:300] + ("..." if len(document) > 300 else "")

            results.append({
                "doc_id": doc_id,
                "match_score": float(score),
                "title": metadata.get("title", "Unknown"),
                "authors": metadata.get("authors", "Unknown"),
                "year": metadata.get("year"),
                "section": metadata.get("section", "unknown"),
                "excerpt": excerpt,
                "tags": metadata.get("tags", "").split(", ") if metadata.get("tags") else [],
                "instruments": metadata.get("instruments", "").split(", ") if metadata.get("instruments") else [],
            })

        return {
            "pattern": pattern,
            "pattern_type": pattern_type,
            "combine_with": combine_with,
            "combine_mode": combine_mode,
            "num_results": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Full text search error: {e}")
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
        # Ensure BM25 index is loaded
        search_engine._ensure_bm25_loaded()

        # Get all documents and filter by year
        all_ids = search_engine.doc_ids
        results = []

        for doc_id in all_ids:
            metadata = search_engine.get_metadata(doc_id)
            doc_year_str = metadata.get("year")

            # Convert string to int (handle "unknown" and None)
            try:
                doc_year = int(doc_year_str) if doc_year_str and doc_year_str != "unknown" else None
            except (ValueError, TypeError):
                doc_year = None

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
        # Ensure BM25 index is loaded
        search_engine._ensure_bm25_loaded()

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
def index_documents(directory_path: str, recursive: bool = False) -> Dict:
    """
    Index all markdown and PDF documents from a directory.

    Args:
        directory_path: Path to directory containing documents (markdown and/or PDFs)
        recursive: If True, search in subdirectories recursively (useful for Zotero storage)

    Returns:
        Status of indexing operation
    """
    if not search_engine or not chunker:
        return {"error": "Search engine not initialized"}

    try:
        return index_all_documents(directory_path, recursive=recursive)
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
        # Ensure BM25 index is loaded
        search_engine._ensure_bm25_loaded()

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
