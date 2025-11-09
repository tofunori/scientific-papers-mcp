"""Data models for Zotero documents"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import hashlib


@dataclass
class ZoteroDocument:
    """
    Rich metadata structure for Zotero documents
    Inspired by Zotero MCP implementation for optimal search and deduplication
    """

    # Identifiers
    item_key: str  # Unique identifier (hash of file path if not from Zotero API)
    file_path: Path
    filename: str

    # Core metadata
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None

    # Extended bibliographic data
    doi: Optional[str] = None
    citation_key: Optional[str] = None  # e.g., "Smith2023glacier"
    publication: Optional[str] = None  # Journal or conference name
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

    # Zotero-specific metadata
    tags: List[str] = field(default_factory=list)
    collections: List[str] = field(default_factory=list)

    # Temporal tracking (for incremental indexing)
    date_added: datetime = field(default_factory=datetime.now)
    date_modified: datetime = field(default_factory=datetime.now)

    # Content flags
    has_fulltext: bool = False
    fulltext_source: Optional[str] = None  # "pdf", "markdown", "ocr"
    file_type: str = "pdf"  # "pdf", "markdown", "txt"
    is_scanned: bool = False  # True if OCR was needed
    page_count: Optional[int] = None

    # Full-text content (optional, for indexing)
    full_text: Optional[str] = None

    # Images extracted from PDF (for multimodal embedding)
    images: List[dict] = field(default_factory=list)

    def __post_init__(self):
        """Generate item_key if not provided"""
        if not self.item_key:
            self.item_key = self.generate_item_key()

        # Generate citation key if not provided
        if not self.citation_key and self.authors and self.year:
            self.citation_key = self.generate_citation_key()

    def generate_item_key(self) -> str:
        """
        Generate a unique item key from file path
        Format: first 8 chars of SHA256 hash (like Zotero's 8-char keys)
        """
        path_str = str(self.file_path.resolve())
        hash_obj = hashlib.sha256(path_str.encode())
        return hash_obj.hexdigest()[:8].upper()

    def generate_citation_key(self) -> str:
        """
        Generate a citation key in format: FirstAuthorLastNameYYYY
        Example: Smith2023, JonesAndDavis2024
        """
        if not self.authors or not self.year:
            return self.item_key

        # Get first author's last name
        first_author = self.authors[0]
        # Simple parsing: take last part after comma or space
        if "," in first_author:
            last_name = first_author.split(",")[0].strip()
        else:
            parts = first_author.split()
            last_name = parts[-1] if parts else first_author

        # Remove non-alphanumeric characters
        last_name = "".join(c for c in last_name if c.isalnum())

        return f"{last_name}{self.year}"

    def to_metadata_dict(self) -> dict:
        """
        Convert to metadata dictionary for ChromaDB storage
        Only includes serializable fields
        """
        return {
            "item_key": self.item_key,
            "filename": self.filename,
            "file_type": self.file_type,
            "title": self.title or "Unknown",
            "authors": ", ".join(self.authors) if self.authors else "Unknown",
            "year": str(self.year) if self.year else "Unknown",
            "doi": self.doi or "",
            "citation_key": self.citation_key or "",
            "publication": self.publication or "",
            "abstract": self.abstract or "",
            "keywords": ", ".join(self.keywords) if self.keywords else "",
            "tags": ", ".join(self.tags) if self.tags else "",
            "collections": ", ".join(self.collections) if self.collections else "",
            "has_fulltext": str(self.has_fulltext),
            "fulltext_source": self.fulltext_source or "",
            "is_scanned": str(self.is_scanned),
            "page_count": str(self.page_count) if self.page_count else "",
            "date_added": self.date_added.isoformat(),
            "date_modified": self.date_modified.isoformat(),
        }

    def to_embedding_text(self, include_fulltext: bool = True) -> str:
        """
        Hierarchical text composition for optimal embeddings
        Inspired by Zotero MCP's field prioritization

        Priority order:
        1. Title (highest semantic weight)
        2. Authors (critical for citation queries)
        3. Abstract (core semantic content)
        4. Publication venue
        5. Keywords/Tags (user-defined semantic markers)
        6. Full-text (if available and requested)
        """
        components = []

        # Priority 1: Title
        if self.title:
            components.append(f"Title: {self.title}")

        # Priority 2: Authors
        if self.authors:
            authors_str = "; ".join(self.authors)
            components.append(f"Authors: {authors_str}")

        # Priority 3: Abstract
        if self.abstract:
            components.append(f"Abstract: {self.abstract}")

        # Priority 4: Publication venue
        if self.publication:
            year_str = f" ({self.year})" if self.year else ""
            components.append(f"Published in: {self.publication}{year_str}")

        # Priority 5: Keywords and Tags
        all_keywords = list(set(self.keywords + self.tags))
        if all_keywords:
            components.append(f"Keywords: {', '.join(all_keywords)}")

        # Priority 6: Full-text (if available)
        # With chunking enabled, we want the complete text for optimal chunk generation
        if include_fulltext and self.has_fulltext and self.full_text:
            components.append(f"Content: {self.full_text}")

        return "\n\n".join(components)

    @property
    def normalized_title(self) -> str:
        """
        Normalized title for deduplication
        Lowercase, no punctuation, no articles (a, an, the)
        """
        if not self.title:
            return ""

        # Lowercase
        title = self.title.lower()

        # Remove punctuation
        import string

        title = "".join(c if c not in string.punctuation else " " for c in title)

        # Remove articles and extra spaces
        words = title.split()
        words = [w for w in words if w not in {"a", "an", "the"}]

        return " ".join(words)

    def to_chunks(
        self,
        chunker,  # DocumentChunker instance (avoid circular import)
        include_fulltext: bool = True
    ) -> list:
        """
        Generate chunks from document with metadata

        This method splits the document into smaller chunks suitable for embedding,
        preserving semantic boundaries while maintaining context through overlap.

        Args:
            chunker: DocumentChunker instance from utils.text_chunker
            include_fulltext: Include full text in chunks (default: True)

        Returns:
            List of (chunk_text, chunk_metadata) tuples

        Example:
            >>> from utils.text_chunker import DocumentChunker
            >>> chunker = DocumentChunker(chunk_size=512, chunk_overlap=100)
            >>> chunks = doc.to_chunks(chunker, include_fulltext=True)
            >>> print(f"Generated {len(chunks)} chunks")
            >>> print(f"First chunk: {chunks[0][1]['chunk_id']}")
        """
        # Get hierarchical text using same priority order as before:
        # Title → Authors → Abstract → Publication → Keywords → Full-text
        full_text = self.to_embedding_text(include_fulltext=include_fulltext)

        # Base metadata for all chunks (same as before, but now chunked)
        base_metadata = self.to_metadata_dict()

        # Chunk the text using the provided chunker
        chunks = chunker.chunk_text(full_text, base_metadata)

        return chunks

    def __repr__(self) -> str:
        """String representation for debugging"""
        authors_str = f"{self.authors[0]} et al." if len(self.authors) > 1 else self.authors[0] if self.authors else "Unknown"
        return f"ZoteroDocument(key={self.item_key}, title='{self.title[:50]}...', authors={authors_str}, year={self.year})"
