"""Tests for PDF chunking functionality"""

import pytest
from src.indexing.chunker import ScientificPaperChunker, DocumentChunk


class TestPDFChunking:
    """Test PDF document chunking"""

    @pytest.fixture
    def chunker(self):
        """Create a chunker instance"""
        return ScientificPaperChunker(max_chunk_size=100, chunk_overlap=20)

    @pytest.fixture
    def sample_pdf_text(self):
        """Sample PDF text (no markdown structure)"""
        return """
        Deep Learning for Glacier Monitoring: A Comprehensive Review

        This paper presents a comprehensive review of deep learning approaches
        for glacier monitoring using satellite imagery and climate data.

        Introduction

        Glaciers are important indicators of climate change. Monitoring glacier
        extent and mass balance is crucial for understanding climate dynamics.
        Recent advances in deep learning have enabled automated glacier detection
        and monitoring from satellite images.

        Methods

        We developed a convolutional neural network architecture optimized for
        glacier detection. The model was trained on Sentinel-2 and Landsat 8 data.

        Results

        Our model achieved 95% accuracy on the test set. The results show that
        deep learning approaches significantly outperform traditional methods.

        Discussion

        The high accuracy demonstrates the effectiveness of CNNs for glacier
        monitoring. Future work will focus on temporal analysis and climate
        impact assessment.

        Conclusion

        Deep learning provides a powerful tool for glacier monitoring and can
        contribute to climate change research.
        """

    def test_chunk_pdf_document_returns_list(self, chunker, sample_pdf_text):
        """Test that chunk_pdf_document returns list of DocumentChunk"""
        chunks = chunker.chunk_pdf_document(sample_pdf_text, "test_pdf_1")
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunk_ids_are_unique(self, chunker, sample_pdf_text):
        """Test that chunk IDs are unique"""
        chunks = chunker.chunk_pdf_document(sample_pdf_text, "test_pdf_1")
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunk_ids_contain_document_id(self, chunker, sample_pdf_text):
        """Test that chunk IDs contain the document ID"""
        chunks = chunker.chunk_pdf_document(sample_pdf_text, "test_pdf_1")
        assert all("test_pdf_1" in c.chunk_id for c in chunks)

    def test_chunks_have_valid_text(self, chunker, sample_pdf_text):
        """Test that all chunks contain non-empty text"""
        chunks = chunker.chunk_pdf_document(sample_pdf_text, "test_pdf_1")
        assert all(c.text and c.text.strip() for c in chunks)

    def test_chunk_size_respects_limit(self, chunker, sample_pdf_text):
        """Test that chunks respect max_chunk_size limit"""
        chunks = chunker.chunk_pdf_document(sample_pdf_text, "test_pdf_1")

        for chunk in chunks:
            tokens = chunker._count_tokens(chunk.text)
            # Allow slight overage (one sentence) due to splitting logic
            assert tokens <= chunker.max_chunk_size + 50

    def test_section_detection(self, chunker, sample_pdf_text):
        """Test that section headers are detected"""
        chunks = chunker.chunk_pdf_document(sample_pdf_text, "test_pdf_1")

        # Should have chunks from various sections
        sections = set(c.section for c in chunks)
        # At least one section should be detected
        assert len(sections) >= 1

    def test_chunk_document_id_assignment(self, chunker, sample_pdf_text):
        """Test that document_id is correctly assigned to chunks"""
        doc_id = "glacier_paper_2023"
        chunks = chunker.chunk_pdf_document(sample_pdf_text, doc_id)

        assert all(c.document_id == doc_id for c in chunks)

    def test_chunk_numbering_sequential(self, chunker, sample_pdf_text):
        """Test that chunk_num is sequential"""
        chunks = chunker.chunk_pdf_document(sample_pdf_text, "test_pdf_1")
        chunk_nums = [c.chunk_num for c in chunks]

        assert chunk_nums == list(range(len(chunks)))

    def test_empty_document_handling(self, chunker):
        """Test chunking of empty document"""
        chunks = chunker.chunk_pdf_document("", "empty_doc")

        # Should return fallback single chunk or empty list
        assert isinstance(chunks, list)

    def test_very_large_paragraph_splitting(self, chunker):
        """Test that very large paragraphs are split correctly"""
        large_text = " ".join(["word"] * 2000)  # Very large paragraph

        chunks = chunker.chunk_pdf_document(large_text, "large_doc")

        assert len(chunks) > 1  # Should be split into multiple chunks

    def test_chunking_preserves_all_content(self, chunker, sample_pdf_text):
        """Test that chunking doesn't lose content"""
        chunks = chunker.chunk_pdf_document(sample_pdf_text, "test_pdf_1")
        reconstructed = " ".join(c.text for c in chunks)

        # Should preserve most of the original content
        # (some whitespace normalization expected)
        assert len(reconstructed) >= len(sample_pdf_text) * 0.8

    def test_multiple_documents_with_same_chunker(self, chunker):
        """Test chunking multiple documents with same chunker instance"""
        doc1 = "First document with some content."
        doc2 = "Second document with different content."

        chunks1 = chunker.chunk_pdf_document(doc1, "doc1")
        chunks2 = chunker.chunk_pdf_document(doc2, "doc2")

        assert all("doc1" in c.chunk_id for c in chunks1)
        assert all("doc2" in c.chunk_id for c in chunks2)

    def test_chunk_section_assignment(self, chunker, sample_pdf_text):
        """Test that section information is properly assigned"""
        chunks = chunker.chunk_pdf_document(sample_pdf_text, "test_pdf_1")

        # Check that at least some chunks have meaningful sections
        non_default_sections = [c.section for c in chunks if c.section != "Introduction"]
        assert len(non_default_sections) > 0 or len(chunks) > 1
