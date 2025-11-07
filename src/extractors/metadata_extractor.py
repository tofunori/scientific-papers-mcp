"""Metadata extraction from markdown and PDF documents"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import logging

from .patterns import (
    compile_patterns,
    extract_year_from_filename,
    extract_tags_from_text,
    extract_instruments_from_text,
    extract_authors_list,
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata extracted from a scientific document"""

    filename: str
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    tags: List[str] = field(default_factory=list)  # BRDF, albedo, aerosols, glaciers
    instruments: List[str] = field(
        default_factory=list
    )  # MODIS, Sentinel, Landsat, etc.

    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            "filename": self.filename,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "journal": self.journal,
            "doi": self.doi,
            "abstract": self.abstract,
            "tags": self.tags,
            "instruments": self.instruments,
        }

    def __hash__(self):
        """Allow DocumentMetadata to be used in sets/dicts"""
        return hash(self.filename)


class MetadataExtractor:
    """Extract metadata from markdown and PDF scientific documents"""

    def __init__(self):
        """Initialize the extractor with compiled patterns"""
        self.patterns = compile_patterns()

    def extract(self, text: str, filepath: str, is_pdf: bool = False) -> DocumentMetadata:
        """
        Extract metadata from document text (markdown or PDF)

        Args:
            text: Full text of the document
            filepath: Path to the document file
            is_pdf: If True, treat as PDF text; if False, treat as markdown

        Returns:
            DocumentMetadata object with extracted information
        """
        filename = Path(filepath).name
        file_type = "PDF" if is_pdf else "Markdown"
        logger.info(f"Extracting metadata from {filename} ({file_type})")

        # Extract structured fields
        title = self._extract_title(text) or filename
        authors = self._extract_authors(text)
        year = self._extract_year(text, filepath)
        journal = self._extract_journal(text)
        doi = self._extract_doi(text)
        abstract = self._extract_abstract(text)

        # Extract thematic information (only for markdown)
        tags = extract_tags_from_text(text) if not is_pdf else []
        instruments = extract_instruments_from_text(text) if not is_pdf else []

        return DocumentMetadata(
            filename=filename,
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            doi=doi,
            abstract=abstract,
            tags=tags,
            instruments=instruments,
        )

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract title from document"""
        match = self.patterns["title"].search(text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_authors(self, text: str) -> List[str]:
        """Extract and parse authors from document"""
        match = self.patterns["authors"].search(text)
        if match:
            authors_string = match.group(1).strip()
            return extract_authors_list(authors_string)
        return []

    def _extract_year(self, text: str, filepath: str) -> Optional[int]:
        """Extract year from document, fallback to filename"""
        # Try pattern matching in text
        match = self.patterns["year"].search(text)
        if match:
            try:
                year = int(match.group(1))
                if 1950 <= year <= 2050:
                    return year
            except ValueError:
                pass

        # Fallback: try to extract from filename
        year = extract_year_from_filename(filepath)
        if year:
            return year

        return None

    def _extract_journal(self, text: str) -> Optional[str]:
        """Extract journal name from document"""
        match = self.patterns["journal"].search(text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI from document"""
        match = self.patterns["doi"].search(text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_abstract(self, text: str) -> Optional[str]:
        """Extract abstract from document"""
        match = self.patterns["abstract"].search(text)
        if match:
            abstract = match.group(1).strip()
            # Limit to first 500 characters
            return abstract[:500] if len(abstract) > 500 else abstract
        return None


def extract_metadata_from_file(filepath: Path) -> Optional[DocumentMetadata]:
    """
    Convenience function to extract metadata from a markdown file

    Args:
        filepath: Path to markdown file

    Returns:
        DocumentMetadata or None if file cannot be read
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        extractor = MetadataExtractor()
        return extractor.extract(text, str(filepath), is_pdf=False)
    except Exception as e:
        logger.error(f"Error extracting metadata from {filepath}: {e}")
        return None
