"""PDF extraction with OCR support for scientific documents"""

from pathlib import Path
from typing import Optional, Tuple
import logging
import re

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_pdf_scanned(doc: "fitz.Document") -> bool:
    """
    Detect if PDF is scanned (image-based) or text-based

    Args:
        doc: PyMuPDF document object

    Returns:
        True if PDF appears to be scanned, False if text-based
    """
    try:
        # Sample first 5 pages to check for text content
        pages_to_check = min(5, len(doc))

        for page_num in range(pages_to_check):
            page = doc[page_num]
            text = page.get_text()

            # If we find substantial text, it's not scanned
            if len(text.strip()) > 100:
                return False

        return True
    except Exception as e:
        logger.warning(f"Error detecting if PDF is scanned: {e}")
        return False


def extract_text_from_pdf(pdf_path: Path) -> Tuple[str, bool]:
    """
    Extract text from PDF with automatic OCR detection

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (extracted_text, is_ocr_needed)

    Raises:
        ImportError: If PyMuPDF not installed
        FileNotFoundError: If PDF file not found
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError(
            "PyMuPDF (pymupdf) is required for PDF extraction. "
            "Install with: pip install pymupdf"
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info(f"Extracting text from PDF: {pdf_path.name}")

    try:
        doc = fitz.open(pdf_path)

        # Check if PDF is scanned
        is_scanned = is_pdf_scanned(doc)

        if is_scanned and not TESSERACT_AVAILABLE:
            logger.warning(
                f"PDF {pdf_path.name} appears to be scanned, but Tesseract is not installed. "
                "Install with: pip install pytesseract\n"
                "Also install Tesseract engine: https://github.com/UB-Mannheim/tesseract/wiki"
            )

        all_text = []

        for page_num, page in enumerate(doc):
            try:
                # First try native text extraction
                text = page.get_text()

                # If page is mostly empty and PDF is scanned, try OCR
                if is_scanned and len(text.strip()) < 100:
                    if TESSERACT_AVAILABLE:
                        text = _extract_text_with_ocr(page)
                    else:
                        logger.warning(f"Page {page_num + 1} is mostly empty (likely image-based)")

                if text.strip():
                    all_text.append(text)
            except Exception as e:
                logger.error(f"Error extracting text from page {page_num + 1}: {e}")
                continue

        doc.close()

        result_text = "\n\n".join(all_text)
        logger.info(f"Successfully extracted {len(result_text)} characters from PDF")

        return result_text, is_scanned

    except Exception as e:
        logger.error(f"Error opening PDF {pdf_path}: {e}")
        raise


def _extract_text_with_ocr(page: "fitz.Page") -> str:
    """
    Extract text from PDF page using Tesseract OCR

    Args:
        page: PyMuPDF page object

    Returns:
        Extracted text string
    """
    if not TESSERACT_AVAILABLE:
        raise ImportError(
            "Tesseract is required for OCR. Install with: pip install pytesseract\n"
            "Also install Tesseract engine: https://github.com/UB-Mannheim/tesseract/wiki"
        )

    try:
        # Render page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Extract text using Tesseract
        text = pytesseract.image_to_string(img)

        return text
    except Exception as e:
        logger.error(f"Error during OCR: {e}")
        return ""


def extract_metadata_from_pdf(pdf_path: Path) -> dict:
    """
    Extract metadata from PDF

    Tries to extract from native PDF metadata first, then falls back to
    regex extraction from text content.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dictionary with metadata fields
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError(
            "PyMuPDF (pymupdf) is required for PDF extraction. "
            "Install with: pip install pymupdf"
        )

    pdf_path = Path(pdf_path)
    filename = pdf_path.name
    metadata = {
        "filename": filename,
        "title": None,
        "authors": [],
        "year": None,
        "journal": None,
        "doi": None,
        "abstract": None,
    }

    try:
        doc = fitz.open(pdf_path)
        pdf_metadata = doc.metadata

        # Extract from native PDF metadata
        if pdf_metadata:
            if pdf_metadata.get("title"):
                metadata["title"] = pdf_metadata["title"].strip()

            if pdf_metadata.get("author"):
                author_str = pdf_metadata["author"].strip()
                metadata["authors"] = _parse_authors(author_str)

            if pdf_metadata.get("subject"):
                metadata["subject"] = pdf_metadata["subject"].strip()

            if pdf_metadata.get("keywords"):
                metadata["keywords"] = pdf_metadata["keywords"].strip()

        # Extract year from PDF creation/modification date
        if pdf_metadata and pdf_metadata.get("modDate"):
            year = _extract_year_from_pdf_date(pdf_metadata["modDate"])
            if year:
                metadata["year"] = year

        # If native metadata incomplete, try text extraction
        if not metadata["title"] or not metadata["authors"]:
            text_content = ""
            for page in doc:
                text_content += page.get_text()
                if len(text_content) > 2000:  # Only need first ~2000 chars
                    break

            # Fallback: extract from text content
            if not metadata["title"]:
                title = _extract_title_from_text(text_content)
                if title:
                    metadata["title"] = title

            if not metadata["authors"]:
                authors = _extract_authors_from_text(text_content)
                if authors:
                    metadata["authors"] = authors

            # Extract other fields from text
            if not metadata["year"]:
                year = _extract_year_from_text(text_content)
                if year:
                    metadata["year"] = year

            if not metadata["journal"]:
                journal = _extract_journal_from_text(text_content)
                if journal:
                    metadata["journal"] = journal

            if not metadata["doi"]:
                doi = _extract_doi_from_text(text_content)
                if doi:
                    metadata["doi"] = doi

            if not metadata["abstract"]:
                abstract = _extract_abstract_from_text(text_content)
                if abstract:
                    metadata["abstract"] = abstract

        # Fallback to filename if still no title
        if not metadata["title"]:
            metadata["title"] = filename

        doc.close()

    except Exception as e:
        logger.error(f"Error extracting metadata from {filename}: {e}")
        metadata["title"] = filename

    return metadata


def _extract_year_from_pdf_date(date_str: str) -> Optional[int]:
    """Extract year from PDF date string (format: D:YYYYMMDDhhmmss...)"""
    try:
        # PDF dates are in format D:YYYYMMDDhhmmss
        if date_str.startswith("D:"):
            year_str = date_str[2:6]
            year = int(year_str)
            if 1950 <= year <= 2050:
                return year
    except (ValueError, IndexError):
        pass
    return None


def _parse_authors(author_str: str) -> list:
    """Parse author string into list of authors"""
    # Handle various formats: "Author1, Author2", "Author1 and Author2", etc.
    author_str = author_str.strip()

    # Split by common separators
    separators = [" and ", ", ", ";"]
    authors = []

    current = author_str
    for sep in separators:
        if sep in current:
            parts = current.split(sep)
            authors = [p.strip() for p in parts if p.strip()]
            break

    if not authors:
        authors = [author_str] if author_str else []

    return authors


def _extract_title_from_text(text: str) -> Optional[str]:
    """Extract title from document text"""
    # Look for first line that looks like a title (all caps or title case)
    lines = text.split("\n")

    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if not line:
            continue

        # Skip very short lines and common headers
        if len(line) < 10 or len(line) > 200:
            continue

        if line.isupper():
            return line

        # Title case with reasonable length
        if len(line) > 20 and sum(1 for c in line if c.isupper()) > 2:
            return line

    return None


def _extract_authors_from_text(text: str) -> list:
    """Extract authors from document text"""
    # Look for author patterns in first 1000 chars
    text = text[:1000]

    # Common patterns for author lines
    author_patterns = [
        r"Authors?:\s*(.+?)(?:\n|$)",
        r"By\s+(.+?)(?:\n|$)",
        r"^([A-Z][a-z]+ [A-Z][a-z]+)\s+and\s+([A-Z][a-z]+ [A-Z][a-z]+)",
    ]

    for pattern in author_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            if len(match.groups()) > 1:
                authors = [g.strip() for g in match.groups() if g]
            else:
                author_str = match.group(1)
                authors = _parse_authors(author_str)

            if authors:
                return authors

    return []


def _extract_year_from_text(text: str) -> Optional[int]:
    """Extract year from document text"""
    # Look for year patterns
    year_pattern = r"\b(19|20)\d{2}\b"

    matches = re.findall(year_pattern, text[:1000])

    for match_str in matches:
        full_year = "".join(match_str)
        try:
            year = int(full_year + match_str.split()[-1])  # Avoid issues with overlapping matches
            if 1950 <= year <= 2050:
                return year
        except (ValueError, IndexError):
            continue

    # Simpler approach: find first 4-digit number that looks like a year
    for word in text[:1000].split():
        if word.isdigit() and len(word) == 4:
            try:
                year = int(word)
                if 1950 <= year <= 2050:
                    return year
            except ValueError:
                pass

    return None


def _extract_journal_from_text(text: str) -> Optional[str]:
    """Extract journal name from document text"""
    journal_patterns = [
        r"Journal:\s*(.+?)(?:\n|,)",
        r"Published in\s+(.+?)(?:\n|,)",
        r"(?:^|\n)(.+?)\s+\(Journal\)",
    ]

    for pattern in journal_patterns:
        match = re.search(pattern, text[:1000], re.IGNORECASE | re.MULTILINE)
        if match:
            journal = match.group(1).strip()
            if 10 < len(journal) < 200:  # Reasonable length
                return journal

    return None


def _extract_doi_from_text(text: str) -> Optional[str]:
    """Extract DOI from document text"""
    # DOI pattern
    doi_pattern = r"(?:doi:|DOI:?\s*)(?:https?://doi\.org/)?([0-9.]+/\S+)"

    match = re.search(doi_pattern, text[:2000], re.IGNORECASE)
    if match:
        doi = match.group(1)
        # Clean up trailing punctuation
        doi = doi.rstrip(".,;)")
        return doi

    return None


def _extract_abstract_from_text(text: str) -> Optional[str]:
    """Extract abstract from document text"""
    abstract_patterns = [
        r"(?:^|\n)\s*Abstract\s*\n(.+?)(?:\n\s*(?:Introduction|Keywords|References|$))",
        r"(?:^|\n)ABSTRACT\s*\n(.+?)(?:\n\s*(?:INTRODUCTION|KEYWORDS|$))",
    ]

    for pattern in abstract_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            # Limit to first 500 characters
            if len(abstract) > 500:
                abstract = abstract[:500].rsplit(" ", 1)[0] + "..."
            return abstract

    return None
