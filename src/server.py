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
from .indexing.chroma_client import initialize_chroma
from .indexing.hybrid_search import HybridSearchEngine
from .indexing.zotero_indexer import ZoteroLibraryIndexer
from .indexing.indexing_state import IndexingStateManager
from .indexing.deduplicator import DocumentDeduplicator

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
    - generate_rag_answer: Generate evidence-backed answers with citations
    - search_fulltext: Full-text search with regex and boolean logic
    - search_by_author: Find papers by author name
    - search_by_year: Find papers by publication year
    - get_metadata: Get detailed metadata for a paper
    - list_papers: List all indexed papers
    - index_zotero_library: Index Zotero library with incremental updates (68 chunks/doc)
    - get_collection_stats: Get indexing statistics
    """,
)

# Global search engine instance
search_engine: Optional[HybridSearchEngine] = None


def initialize_server() -> None:
    """Initialize the MCP server and search engine"""
    global search_engine

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

        logger.info("Server initialized successfully")

        # Auto-index disabled by default - use index_zotero_library() tool for manual/incremental indexing
        # Set auto_index_on_start=True in config to enable automatic indexing on startup

    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise


# Old index_all_documents function removed - replaced by ZoteroLibraryIndexer
# for better chunking (68 chunks/doc vs old system) and incremental updates


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool()
def search(
    query: str,
    top_k: int = 10,
    alpha: float = 0.5,
    text_filter: Optional[str] = None,
    source: str = "default",
    use_rrf: bool = True,
    rrf_k_parameter: int = 60,
    rrf_dense_weight: float = 0.7,
    rrf_sparse_weight: float = 0.3) -> Dict:
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
        source: Search source - 'local' (fast), 'cloud' (remote), or 'default' (use config)

        use_rrf: Use Reciprocal Rank Fusion (True) or alpha weighting (False)
        rrf_k_parameter: RRF smoothing parameter k (higher = more uniform ranking)
        rrf_dense_weight: Weight for dense semantic results in RRF (0.7 = 70%)
        rrf_sparse_weight: Weight for sparse keyword results in RRF (0.3 = 30%)
    Returns:
        List of matching papers with relevance scores and metadata
    """
    if not search_engine:
        return {"error": "Search engine not initialized"}

    try:
        logger.info(f"Search: '{query}' (top_k={top_k}, alpha={alpha}, source={source})")

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
            query=query, top_k=top_k, alpha=alpha, where_document=where_document, source=source, use_rrf=use_rrf,
            rrf_k_parameter=rrf_k_parameter,
            rrf_dense_weight=rrf_dense_weight,
            rrf_sparse_weight=rrf_sparse_weight)

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
    source: str = "default",
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
            source=source,
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
    source: str = "default",
) -> Dict:
    """
    Full text search with simplified syntax for exact matching, regex, and boolean logic.

    Args:
        pattern: Text pattern to search for
        pattern_type: Type of search - 'contains', 'regex', or 'exact'
        combine_with: Additional patterns for AND/OR combinations (optional)
        combine_mode: 'and' or 'or' for combining multiple patterns
        top_k: Number of results to return (default: 10)
        source: Search source - 'local' (fast), 'cloud' (remote), or 'default' (use config)

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
        logger.info(f"Full text search: pattern='{pattern}', type={pattern_type}, combine={combine_with}, source={source}")

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
            where_document=where_doc,
            source=source
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
def search_by_author(author_name: str, limit: int = 50, offset: int = 0) -> Dict:
    """
    Find all papers by a specific author (paginated).

    Args:
        author_name: Name of the author to search for (case-insensitive partial match)
        limit: Maximum number of papers to return (default: 50, max: 200)
        offset: Number of papers to skip (default: 0)

    Returns:
        Paginated list of papers by that author

    Examples:
        - search_by_author("Wang") → First 50 papers with "Wang" in author list
        - search_by_author("Smith", limit=100, offset=50) → Papers 51-150 by Smith
        - search_by_author("Unknown") → Papers with missing author metadata
    """
    if not search_engine:
        return {"error": "Search engine not initialized"}

    try:
        # Convert string parameters to int (MCP passes strings)
        limit = int(limit)
        offset = int(offset)

        # Clamp limit to reasonable bounds
        limit = min(max(1, limit), 200)
        offset = max(0, offset)

        logger.info(f"Search by author: {author_name} (limit={limit}, offset={offset})")

        # Get all documents from the collection to search through metadata
        # We use collection.get() directly since ChromaDB's metadata filtering
        # doesn't support substring matching on authors field
        all_data = search_engine.collection.get(
            limit=100000,  # Get all documents
            include=["metadatas", "documents"]
        )

        # Track metadata quality statistics
        total_documents = len(all_data["ids"])
        documents_with_authors = 0
        documents_with_unknown = 0
        documents_with_empty = 0

        # Filter by author name in metadata (case-insensitive partial match)
        matching_papers = []
        for doc_id, metadata, document in zip(all_data["ids"], all_data["metadatas"], all_data["documents"]):
            authors_str = metadata.get("authors", "") if metadata else ""

            # Track metadata quality
            if authors_str and authors_str != "['Unknown']":
                documents_with_authors += 1
            elif authors_str == "['Unknown']":
                documents_with_unknown += 1
            else:
                documents_with_empty += 1

            # Check for match (including "Unknown" placeholder)
            if author_name.lower() in authors_str.lower():
                # Extract unique paper identifier (before chunk suffix)
                paper_id = doc_id.rsplit("_chunk_", 1)[0]

                # Check if we've already added this paper
                if not any(p["paper_id"] == paper_id for p in matching_papers):
                    matching_papers.append({
                        "paper_id": paper_id,
                        "title": metadata.get("title", "Unknown") if metadata else "Unknown",
                        "authors": authors_str,
                        "year": metadata.get("year") if metadata else None,
                        "doc_id": doc_id,  # First chunk ID for reference
                    })

        # Sort by year (descending) and title
        # Convert year to int (stored as string in metadata)
        matching_papers.sort(
            key=lambda p: (-(int(p["year"] or 0)), p["title"].lower())
        )

        # Apply pagination
        total_papers = len(matching_papers)
        paginated_papers = matching_papers[offset:offset + limit]

        # Log metadata quality statistics
        logger.info(
            f"Author search metadata quality: "
            f"{documents_with_authors}/{total_documents} have authors, "
            f"{documents_with_unknown} have placeholder, "
            f"{documents_with_empty} are empty"
        )

        return {
            "author": author_name,
            "total_results": total_papers,
            "limit": limit,
            "offset": offset,
            "returned": len(paginated_papers),
            "has_more": (offset + limit) < total_papers,
            "papers": paginated_papers,
            "metadata_quality": {
                "total_documents": total_documents,
                "with_authors": documents_with_authors,
                "with_unknown_placeholder": documents_with_unknown,
                "completely_empty": documents_with_empty,
                "quality_percentage": round((documents_with_authors / total_documents * 100), 1) if total_documents > 0 else 0
            }
        }

    except Exception as e:
        logger.error(f"Author search error: {e}")
        return {"error": str(e)}


@mcp.tool()
def search_by_year(year: int, start_year: Optional[int] = None, limit: int = 50, offset: int = 0) -> Dict:
    """
    Find papers by publication year or year range (paginated).

    Args:
        year: Specific year or end year for range
        start_year: Optional start year for range search
        limit: Maximum number of papers to return (default: 50, max: 200)
        offset: Number of papers to skip (default: 0)

    Returns:
        Paginated list of papers from specified year(s)

    Examples:
        - search_by_year(2023) → First 50 papers from 2023
        - search_by_year(2025, start_year=2020) → Papers from 2020-2025
        - search_by_year(2023, limit=100, offset=50) → Papers 51-150 from 2023
    """
    if not search_engine:
        return {"error": "Search engine not initialized"}

    try:
        # Ensure BM25 index is loaded
        search_engine._ensure_bm25_loaded()

        # Clamp limit to reasonable bounds
        limit = min(max(1, limit), 200)
        offset = max(0, offset)

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

        # Apply pagination
        total_results = len(results)
        paginated_results = results[offset:offset + limit]

        return {
            "year_range": f"{start_year}-{year}" if start_year else str(year),
            "total_results": total_results,
            "limit": limit,
            "offset": offset,
            "returned": len(paginated_results),
            "has_more": (offset + limit) < total_results,
            "results": paginated_results,
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
def list_papers(limit: int = 50, offset: int = 0) -> Dict:
    """
    List indexed papers with their metadata (paginated).

    Args:
        limit: Maximum number of papers to return (default: 50, max: 200)
        offset: Number of papers to skip (default: 0)

    Returns:
        Paginated list of papers in the collection

    Examples:
        - list_papers() → First 50 papers
        - list_papers(limit=100, offset=50) → Papers 51-150
    """
    if not search_engine:
        return {"error": "Search engine not initialized"}

    try:
        # Ensure BM25 index is loaded
        search_engine._ensure_bm25_loaded()

        # Clamp limit to reasonable bounds
        limit = min(max(1, limit), 200)
        offset = max(0, offset)

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

        # Apply pagination
        total_papers = len(papers)
        paginated_papers = papers[offset:offset + limit]

        return {
            "total_papers": total_papers,
            "limit": limit,
            "offset": offset,
            "returned": len(paginated_papers),
            "has_more": (offset + limit) < total_papers,
            "papers": paginated_papers,
        }

    except Exception as e:
        logger.error(f"List papers error: {e}")
        return {"error": str(e)}


@mcp.tool()
def index_zotero_library(force_rebuild: bool = False, limit: Optional[int] = None) -> Dict:
    """
    Index Zotero library with intelligent incremental updates and document chunking.

    Features:
    - Incremental indexing (only processes new/modified documents)
    - Full-text chunking (avg 68 chunks per document for complete coverage)
    - Deduplication (DOI and title-based)
    - Progress tracking with statistics

    Args:
        force_rebuild: Force reindex all documents (default: False for incremental)
        limit: Limit number of documents to process (for testing, default: None)

    Returns:
        Indexing statistics including documents processed, chunks created, duplicates removed

    Examples:
        - index_zotero_library() → Incremental indexing (only new/modified docs)
        - index_zotero_library(force_rebuild=True) → Full reindex
        - index_zotero_library(limit=10) → Test with 10 documents
    """
    if not search_engine:
        return {"error": "Search engine not initialized"}

    try:
        logger.info(f"Starting Zotero indexing (force_rebuild={force_rebuild}, limit={limit})")

        # Initialize indexer components
        state_manager = IndexingStateManager()
        deduplicator = DocumentDeduplicator()

        # Create indexer with search engine
        indexer = ZoteroLibraryIndexer(
            search_engine=search_engine,
            library_path=config.documents_path,
            state_manager=state_manager,
            deduplicator=deduplicator,
        )

        # Run indexing workflow
        stats = indexer.index_library(
            force_rebuild=force_rebuild,
            limit=limit,
            save_state=True,
        )

        return {
            "status": "success",
            "statistics": stats,
            "message": f"Indexed {stats.get('added', 0) + stats.get('updated', 0)} documents"
        }

    except Exception as e:
        logger.error(f"Zotero indexing error: {e}")
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
