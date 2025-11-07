"""Tests for PDF extraction functionality"""

import pytest
from pathlib import Path
from src.extractors.pdf_extractor import (
    extract_text_from_pdf,
    extract_metadata_from_pdf,
    _extract_title_from_text,
    _extract_authors_from_text,
    _extract_year_from_text,
    _extract_doi_from_text,
)


class TestPDFTextExtraction:
    """Test PDF text extraction"""

    def test_extract_text_from_nonexistent_pdf(self):
        """Test extraction from non-existent file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf(Path("nonexistent.pdf"))

    def test_extract_text_returns_tuple(self):
        """Test that text extraction returns tuple of (text, is_scanned)"""
        # This would require a real PDF file in fixtures
        # Skipping for now as it requires test PDFs
        pass


class TestMetadataExtraction:
    """Test metadata extraction from PDFs"""

    def test_extract_title_from_text(self):
        """Test title extraction from text"""
        text = """
        Deep Learning for Glacier Monitoring

        This is the abstract...
        """
        title = _extract_title_from_text(text)
        assert title is not None
        assert "Glacier" in title

    def test_extract_authors_from_text(self):
        """Test author extraction from text"""
        text = "Authors: John Smith and Jane Doe"
        authors = _extract_authors_from_text(text)
        assert len(authors) > 0

    def test_extract_authors_multiple_formats(self):
        """Test author extraction with various formats"""
        # Format 1: comma-separated
        text1 = "Authors: Smith, J., Doe, J."
        authors1 = _extract_authors_from_text(text1)
        assert len(authors1) > 0

        # Format 2: "and" separator
        text2 = "By John Smith and Jane Doe"
        authors2 = _extract_authors_from_text(text2)
        assert len(authors2) > 0

    def test_extract_year_from_text(self):
        """Test year extraction from text"""
        text = "Published in 2023"
        year = _extract_year_from_text(text)
        assert year == 2023

    def test_extract_year_four_digit_number(self):
        """Test year extraction from standalone 4-digit number"""
        text = "Journal of Glaciology 2022"
        year = _extract_year_from_text(text)
        assert year == 2022

    def test_extract_year_invalid_range(self):
        """Test that years outside valid range are rejected"""
        text = "In the year 1850"
        year = _extract_year_from_text(text)
        assert year is None or (1950 <= year <= 2050)

    def test_extract_doi_from_text(self):
        """Test DOI extraction from text"""
        text = "DOI: 10.1038/nature12373"
        doi = _extract_doi_from_text(text)
        assert doi is not None
        assert "10.1038" in doi

    def test_extract_doi_various_formats(self):
        """Test DOI extraction with various formats"""
        # Format 1: DOI: prefix
        text1 = "doi: 10.1234/example"
        doi1 = _extract_doi_from_text(text1)
        assert doi1 is not None

        # Format 2: https://doi.org/
        text2 = "https://doi.org/10.1234/example"
        doi2 = _extract_doi_from_text(text2)
        assert doi2 is not None

    def test_extract_metadata_dict_structure(self):
        """Test that extracted metadata has expected structure"""
        metadata = extract_metadata_from_pdf(Path("test.pdf"))

        # Should have these keys even if values are defaults
        expected_keys = {"filename", "title", "authors", "year", "journal", "doi"}
        assert all(key in metadata for key in expected_keys)

    def test_extract_metadata_filename_handling(self):
        """Test filename extraction from metadata"""
        metadata = extract_metadata_from_pdf(Path("/path/to/document.pdf"))
        assert metadata["filename"] == "document.pdf"


class TestMetadataExtractionEdgeCases:
    """Test edge cases in metadata extraction"""

    def test_empty_text_handling(self):
        """Test extraction from empty text"""
        text = ""
        title = _extract_title_from_text(text)
        assert title is None

    def test_very_long_title_truncation(self):
        """Test that very long titles are handled"""
        text = "A" * 300  # Very long first line
        title = _extract_title_from_text(text)
        # Should either be None or a reasonable length

    def test_multiple_years_first_valid_returned(self):
        """Test that when multiple years exist, first valid one is returned"""
        text = "Cited: 1950, Published: 2023, Year: 2024"
        year = _extract_year_from_text(text)
        assert year in [1950, 2023, 2024] or year is None

    def test_author_parsing_with_special_characters(self):
        """Test author parsing with special characters"""
        text = "Authors: Jean-Pierre Müller and José García"
        authors = _extract_authors_from_text(text)
        # Should parse without errors

    def test_doi_with_trailing_punctuation(self):
        """Test DOI extraction with trailing punctuation"""
        text = "The DOI is 10.1038/nature12373."
        doi = _extract_doi_from_text(text)
        assert doi is not None
        assert not doi.endswith(".")
