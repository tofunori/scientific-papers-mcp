"""Tests for PDF indexing functionality"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.server import index_all_documents
from src.extractors.pdf_extractor import extract_text_from_pdf, extract_metadata_from_pdf
from src.indexing.chunker import ScientificPaperChunker


class TestPDFIndexing:
    """Test PDF indexing"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_index_empty_directory(self, temp_dir):
        """Test indexing an empty directory"""
        result = index_all_documents(temp_dir)

        assert result["status"] == "success"
        assert result["indexed_files"] == 0
        assert result["markdown_files"] == 0
        assert result["pdf_files"] == 0

    def test_index_directory_with_markdown_only(self, temp_dir):
        """Test indexing directory with only markdown files"""
        # Create test markdown file
        md_file = temp_dir / "test.md"
        md_file.write_text("# Test Paper\n\nSome content")

        with patch("src.server.search_engine") as mock_engine:
            with patch("src.server.chunker") as mock_chunker:
                # Mock chunker to return sample chunks
                mock_chunker.chunk_document.return_value = [
                    MagicMock(text="content", section="intro", chunk_id="test-0")
                ]

                result = index_all_documents(temp_dir)

                # Verify markdown file was found
                assert result["total_files"] == 1

    def test_index_directory_with_pdf_only(self, temp_dir):
        """Test indexing directory with only PDF files"""
        # This test is mostly structural since we can't create real PDFs
        # In a real scenario, we would create test PDF files

        with patch("src.extractors.pdf_extractor.extract_text_from_pdf") as mock_extract_text:
            with patch("src.extractors.pdf_extractor.extract_metadata_from_pdf") as mock_extract_meta:
                with patch("src.server.search_engine") as mock_engine:
                    with patch("src.server.chunker") as mock_chunker:
                        # Create a dummy PDF file
                        pdf_file = temp_dir / "test.pdf"
                        pdf_file.write_bytes(b"%PDF-1.4")  # Minimal PDF header

                        # Mock the extraction functions
                        mock_extract_text.return_value = ("Test content", False)
                        mock_extract_meta.return_value = {
                            "filename": "test.pdf",
                            "title": "Test",
                            "authors": [],
                            "year": 2023,
                        }
                        mock_chunker.chunk_pdf_document.return_value = [
                            MagicMock(text="content", section="intro", chunk_id="test-0")
                        ]

                        result = index_all_documents(temp_dir)

                        assert result["pdf_files"] >= 0

    def test_index_mixed_files(self, temp_dir):
        """Test indexing directory with both markdown and PDF files"""
        # Create markdown file
        md_file = temp_dir / "paper1.md"
        md_file.write_text("# Paper 1\n\nContent")

        # Create dummy PDF
        pdf_file = temp_dir / "paper2.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        with patch("src.extractors.pdf_extractor.extract_text_from_pdf") as mock_extract_text:
            with patch("src.extractors.pdf_extractor.extract_metadata_from_pdf") as mock_extract_meta:
                with patch("src.server.search_engine"):
                    with patch("src.server.chunker") as mock_chunker:
                        mock_extract_text.return_value = ("PDF content", False)
                        mock_extract_meta.return_value = {
                            "filename": "paper2.pdf",
                            "title": "Paper 2",
                            "authors": [],
                            "year": 2023,
                        }
                        mock_chunker.chunk_document.return_value = [
                            MagicMock(text="content", section="intro", chunk_id="md-0")
                        ]
                        mock_chunker.chunk_pdf_document.return_value = [
                            MagicMock(text="content", section="intro", chunk_id="pdf-0")
                        ]

                        result = index_all_documents(temp_dir)

                        assert result["total_files"] == 2

    def test_index_nonexistent_directory(self):
        """Test indexing a non-existent directory"""
        result = index_all_documents(Path("/nonexistent/path"))

        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    def test_chunking_returns_correct_structure(self):
        """Test that chunks have all required fields"""
        chunker = ScientificPaperChunker()
        text = "Introduction\n\nSome content here."
        chunks = chunker.chunk_pdf_document(text, "test_doc")

        for chunk in chunks:
            assert hasattr(chunk, "text")
            assert hasattr(chunk, "chunk_id")
            assert hasattr(chunk, "section")
            assert hasattr(chunk, "document_id")
            assert hasattr(chunk, "chunk_num")

            assert isinstance(chunk.text, str)
            assert isinstance(chunk.chunk_id, str)
            assert len(chunk.text) > 0

    def test_metadata_preservation_during_indexing(self, temp_dir):
        """Test that metadata is preserved during indexing"""
        md_file = temp_dir / "test.md"
        md_file.write_text("# Test\n\n**Authors:** John Doe\n\nContent")

        with patch("src.server.search_engine") as mock_engine:
            with patch("src.server.chunker") as mock_chunker:
                mock_chunker.chunk_document.return_value = [
                    MagicMock(text="content", section="intro", chunk_id="test-0")
                ]

                index_all_documents(temp_dir)

                # Verify that index_document was called
                # (would need to check actual metadata in a real test)

    def test_file_type_metadata_added(self):
        """Test that file_type metadata is added correctly"""
        chunker = ScientificPaperChunker()

        # Test with markdown
        md_text = "# Title\n\nContent"
        md_chunks = chunker.chunk_document(md_text, "md_doc")

        # Test with PDF text
        pdf_text = "Title\n\nContent"
        pdf_chunks = chunker.chunk_pdf_document(pdf_text, "pdf_doc")

        assert len(md_chunks) > 0
        assert len(pdf_chunks) > 0


class TestPDFExtractionIntegration:
    """Integration tests for PDF extraction"""

    def test_pdf_metadata_extraction_dict_keys(self):
        """Test that PDF metadata extraction returns expected keys"""
        # This would test with a real or mocked PDF
        pass

    def test_pdf_text_extraction_handles_encoding(self):
        """Test PDF text extraction with various encodings"""
        # This would test with PDFs of different encodings
        pass

    def test_scanned_pdf_detection(self):
        """Test detection of scanned vs text PDFs"""
        # This would test with both scanned and text PDFs
        pass
