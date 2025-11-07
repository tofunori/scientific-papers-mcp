"""Intelligent hierarchical chunking for scientific documents"""

from dataclasses import dataclass
from typing import List, Optional
import logging
import re

from langchain.text_splitter import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of a document with metadata"""

    text: str
    chunk_id: str  # paper_id-section-chunk_num
    section: str  # The section header
    subsection: Optional[str] = None
    document_id: str = None
    chunk_num: int = 0


class ScientificPaperChunker:
    """Intelligent chunking for scientific papers"""

    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 50):
        """
        Initialize chunker

        Args:
            max_chunk_size: Maximum chunk size in tokens (approximate)
            chunk_overlap: Overlap in tokens between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

        # Headers to split on (hierarchical)
        self.headers_to_split_on = [
            ("#", "title"),
            ("##", "section"),
            ("###", "subsection"),
        ]

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False,
        )

    def chunk_document(
        self, markdown_text: str, document_id: str, keep_abstract: bool = True
    ) -> List[DocumentChunk]:
        """
        Intelligently chunk a scientific paper

        Args:
            markdown_text: Full markdown text of the paper
            document_id: Unique identifier for the document
            keep_abstract: Keep abstract as its own chunk

        Returns:
            List of DocumentChunk objects
        """
        chunks = []

        try:
            # Step 1: Split by headers (respect document structure)
            header_chunks = self.header_splitter.split_text(markdown_text)

            chunk_num = 0

            for header_chunk in header_chunks:
                text = header_chunk.page_content
                metadata = header_chunk.metadata

                # Step 2: If chunk is too large, split further
                if self._count_tokens(text) > self.max_chunk_size:
                    # Split large chunks by paragraphs or sentences
                    sub_chunks = self._split_large_chunk(text)

                    for sub_text in sub_chunks:
                        chunk = DocumentChunk(
                            text=sub_text.strip(),
                            chunk_id=f"{document_id}-{chunk_num}",
                            section=metadata.get("section", "unknown"),
                            subsection=metadata.get("subsection"),
                            document_id=document_id,
                            chunk_num=chunk_num,
                        )
                        if chunk.text:  # Only add non-empty chunks
                            chunks.append(chunk)
                            chunk_num += 1
                else:
                    # Keep chunk as is
                    chunk = DocumentChunk(
                        text=text.strip(),
                        chunk_id=f"{document_id}-{chunk_num}",
                        section=metadata.get("section", "unknown"),
                        subsection=metadata.get("subsection"),
                        document_id=document_id,
                        chunk_num=chunk_num,
                    )
                    if chunk.text:
                        chunks.append(chunk)
                        chunk_num += 1

            logger.info(f"Chunked '{document_id}' into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error chunking document '{document_id}': {e}")
            # Fallback: return entire document as one chunk
            return [
                DocumentChunk(
                    text=markdown_text,
                    chunk_id=f"{document_id}-0",
                    section="full_document",
                    document_id=document_id,
                    chunk_num=0,
                )
            ]

    def chunk_pdf_document(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Chunk a PDF document (text without markdown structure)

        For PDFs, we rely on paragraph breaks and sentence splitting
        since PDFs don't have markdown headers.

        Args:
            text: Full text of the PDF document
            document_id: Unique identifier for the document

        Returns:
            List of DocumentChunk objects
        """
        chunks = []

        try:
            # Step 1: Split by paragraphs (double newlines)
            paragraphs = text.split("\n\n")

            chunk_num = 0
            current_section = "Introduction"  # Default section name

            for i, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                # Detect section headers (single lines followed by double newline)
                if len(paragraph.split("\n")) == 1 and len(paragraph) < 100:
                    # Might be a section header
                    if any(
                        keyword in paragraph.lower()
                        for keyword in [
                            "abstract",
                            "introduction",
                            "method",
                            "results",
                            "discussion",
                            "conclusion",
                            "references",
                        ]
                    ):
                        current_section = paragraph
                        continue

                # Step 2: If paragraph is too large, split further
                if self._count_tokens(paragraph) > self.max_chunk_size:
                    sub_chunks = self._split_large_chunk(paragraph)

                    for sub_text in sub_chunks:
                        chunk = DocumentChunk(
                            text=sub_text.strip(),
                            chunk_id=f"{document_id}-{chunk_num}",
                            section=current_section,
                            document_id=document_id,
                            chunk_num=chunk_num,
                        )
                        if chunk.text:
                            chunks.append(chunk)
                            chunk_num += 1
                else:
                    chunk = DocumentChunk(
                        text=paragraph,
                        chunk_id=f"{document_id}-{chunk_num}",
                        section=current_section,
                        document_id=document_id,
                        chunk_num=chunk_num,
                    )
                    if chunk.text:
                        chunks.append(chunk)
                        chunk_num += 1

            logger.info(f"Chunked PDF '{document_id}' into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error chunking PDF '{document_id}': {e}")
            # Fallback: return entire document as one chunk
            return [
                DocumentChunk(
                    text=text,
                    chunk_id=f"{document_id}-0",
                    section="full_document",
                    document_id=document_id,
                    chunk_num=0,
                )
            ]

    def _count_tokens(self, text: str) -> int:
        """
        Approximate token count (words ≈ 1.3 tokens)

        Args:
            text: Text to count

        Returns:
            Approximate token count
        """
        words = len(text.split())
        return int(words * 1.3)

    def _split_large_chunk(self, text: str) -> List[str]:
        """
        Split a large chunk by paragraphs or sentences

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # Split by double newline (paragraphs)
        paragraphs = text.split("\n\n")

        # Combine paragraphs until we reach max size
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if (
                self._count_tokens(current_chunk + "\n\n" + paragraph)
                <= self.max_chunk_size
            ):
                current_chunk += "\n\n" + paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        # If still too large, split by sentences
        final_chunks = []
        for chunk in chunks:
            if self._count_tokens(chunk) > self.max_chunk_size:
                sentences = self._split_by_sentences(chunk)
                final_chunks.extend(sentences)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Split text by sentences (fallback for very large paragraphs)

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting on period, question mark, exclamation
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Combine sentences to reach target chunk size
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if (
                self._count_tokens(current_chunk + " " + sentence)
                <= self.max_chunk_size
            ):
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
