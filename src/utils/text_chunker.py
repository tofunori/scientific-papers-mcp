"""Text chunking utilities for document processing

This module provides chunking capabilities for splitting scientific papers
into semantically meaningful chunks for embedding and retrieval.

Uses LangChain's RecursiveCharacterTextSplitter with token-based counting
for optimal performance with scientific papers.
"""

from typing import List, Tuple, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentChunker:
    """
    Recursive text chunker optimized for scientific papers

    Uses LangChain's battle-tested RecursiveCharacterTextSplitter with:
    - Token-based counting (accurate for embeddings)
    - Paragraph-aware splitting (preserves semantic boundaries)
    - Configurable overlap (prevents context loss at boundaries)

    Based on 2024-2025 RAG best practices research:
    - Recursive chunking recommended for coherent technical documents
    - 512 tokens optimal for scientific papers
    - 20% overlap standard for maintaining context
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize document chunker

        Args:
            chunk_size: Target chunk size in tokens (default: 512)
            chunk_overlap: Number of overlapping tokens (default: 100, ~20%)
            encoding_name: Tokenizer encoding (default: cl100k_base for GPT models)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name

        # Initialize LangChain splitter with token-based counting
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
            # Separators prioritized for scientific papers:
            # 1. Paragraphs (double newline) - main semantic boundaries
            # 2. Sentences (period + space) - secondary boundaries
            # 3. Words (space) - fallback
            # 4. Characters - last resort
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        logger.info(
            f"DocumentChunker initialized: "
            f"chunk_size={chunk_size} tokens, "
            f"overlap={chunk_overlap} tokens ({chunk_overlap/chunk_size*100:.0f}%), "
            f"encoding={encoding_name}"
        )

    def chunk_text(
        self,
        text: str,
        metadata: Dict = None
    ) -> List[Tuple[str, Dict]]:
        """
        Split text into chunks with metadata

        Args:
            text: Full text to chunk
            metadata: Base metadata to attach to all chunks

        Returns:
            List of (chunk_text, chunk_metadata) tuples

        Example:
            >>> chunker = DocumentChunker(chunk_size=512, chunk_overlap=100)
            >>> text = "Long scientific paper content..."
            >>> metadata = {"title": "Paper Title", "doi": "10.1234/xyz"}
            >>> chunks = chunker.chunk_text(text, metadata)
            >>> print(len(chunks))  # 15-20 chunks for typical paper
            >>> print(chunks[0][1]["chunk_index"])  # 0
            >>> print(chunks[0][1]["total_chunks"])  # 18
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to chunk_text")
            return []

        # Split text using LangChain
        try:
            chunks = self.splitter.split_text(text)
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            # Fallback: return entire text as single chunk
            chunks = [text]

        # Prepare chunk tuples with metadata
        result = []
        for i, chunk in enumerate(chunks):
            # Copy base metadata and add chunk-specific fields
            chunk_meta = (metadata or {}).copy()

            # Add chunk tracking fields
            chunk_meta.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_id": f"{chunk_meta.get('item_key', 'unknown')}_{i:03d}",
                "has_overlap": i > 0,  # All chunks except first have overlap
            })

            result.append((chunk, chunk_meta))

        logger.debug(
            f"Split text into {len(chunks)} chunks "
            f"(avg length: {sum(len(c) for c in chunks) // len(chunks)} chars)"
        )

        return result

    def get_chunk_count_estimate(self, text: str) -> int:
        """
        Estimate number of chunks without actually splitting

        Args:
            text: Text to estimate

        Returns:
            Estimated number of chunks
        """
        if not text:
            return 0

        # Rough estimation: text length / (chunk_size * chars_per_token)
        # Assuming ~4 characters per token on average
        chars_per_token = 4
        estimated_tokens = len(text) / chars_per_token

        # Account for overlap
        effective_chunk_size = self.chunk_size - self.chunk_overlap
        estimated_chunks = int(estimated_tokens / effective_chunk_size) + 1

        return max(1, estimated_chunks)
