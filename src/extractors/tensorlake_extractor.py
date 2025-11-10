"""TensorLake-based PDF extraction with structured data support

TensorLake is a Document Ingestion API with state-of-the-art performance:
- Best-in-class table recognition (TEDS score: 86.79)
- Superior layout detection (F1 score: 91.7)
- Structured extraction with Pydantic models
- Built-in chunking and reading order preservation

Performance (Nov 2024 benchmarks):
- TensorLake: TEDS 86.79, F1 91.7 ($10/1k pages)
- Azure: TEDS 78.14, F1 88.1 ($10/1k pages)
- AWS Textract: TEDS 80.75, F1 88.4 ($15/1k pages)

API Documentation: https://docs.tensorlake.ai/
Benchmarks: https://www.tensorlake.ai/blog/benchmarks
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import re

logger = logging.getLogger(__name__)

# Try to import TensorLake
try:
    from tensorlake.documentai import DocumentAI, ParseStatus
    TENSORLAKE_AVAILABLE = True
except ImportError:
    TENSORLAKE_AVAILABLE = False
    logger.warning("tensorlake not installed. Install with: pip install tensorlake")


class TensorLakeExtractor:
    """
    Extract text and metadata from PDFs using TensorLake API

    TensorLake provides best-in-class document parsing with:
    - Superior table recognition (86.79 TEDS score)
    - Advanced layout detection (91.7 F1 score)
    - Structured extraction support
    - Built-in chunking

    Pricing: $10 per 1,000 pages
    Quality: Best-in-class (beats Azure and AWS)

    Example:
        extractor = TensorLakeExtractor(api_key="your-key")
        markdown, metadata, images = extractor.extract_text_from_pdf(pdf_path)
    """

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        retry_delay: int = 2,
        timeout: int = 300,
    ):
        """
        Initialize TensorLake extractor

        Args:
            api_key: TensorLake API key (from cloud.tensorlake.ai)
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
            timeout: Maximum time to wait for parsing completion

        Raises:
            ImportError: If tensorlake package not installed
        """
        if not TENSORLAKE_AVAILABLE:
            raise ImportError(
                "tensorlake is not installed. Install with:\n"
                "pip install tensorlake"
            )

        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        # Initialize TensorLake client
        self.client = DocumentAI(api_key=api_key)

        logger.info("TensorLake extractor initialized")

    def extract_text_from_pdf(
        self,
        pdf_path: Path,
        extract_images: bool = True,
    ) -> Tuple[str, Dict[str, Any], List[Dict]]:
        """
        Extract text from PDF using TensorLake

        Args:
            pdf_path: Path to PDF file
            extract_images: Whether to extract images (not yet implemented)

        Returns:
            Tuple of (markdown_text, metadata, images_list)
            - markdown_text: Full text in markdown format
            - metadata: Dictionary with extraction metadata
            - images_list: List of image dictionaries (empty for now)

        Raises:
            FileNotFoundError: If PDF file not found
            RuntimeError: If extraction fails
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting PDF with TensorLake: {pdf_path.name}")

        try:
            # Step 1: Upload PDF
            logger.debug(f"Uploading {pdf_path.name} to TensorLake...")
            file_id = self.client.upload(str(pdf_path))
            logger.debug(f"Uploaded successfully, file_id: {file_id}")

            # Step 2: Parse document
            logger.debug(f"Parsing {pdf_path.name}...")
            parse_id = self.client.parse(file_id)
            logger.debug(f"Parse started, parse_id: {parse_id}")

            # Step 3: Wait for completion with timeout
            logger.debug(f"Waiting for parsing to complete (timeout: {self.timeout}s)...")
            start_time = time.time()
            result = None

            while time.time() - start_time < self.timeout:
                result = self.client.wait_for_completion(parse_id, timeout=10)

                if result.status == ParseStatus.SUCCESSFUL:
                    logger.debug(f"Parsing completed successfully")
                    break
                elif result.status == ParseStatus.FAILED:
                    raise RuntimeError(f"TensorLake parsing failed: {result.error}")

                # Still processing, wait a bit
                time.sleep(2)
            else:
                raise RuntimeError(f"Parsing timed out after {self.timeout}s")

            # Step 4: Extract markdown from chunks
            markdown_chunks = []
            for chunk in result.chunks:
                markdown_chunks.append(chunk.content)

            markdown_text = "\n\n".join(markdown_chunks)

            # Step 5: Build metadata
            metadata = {
                "extraction_method": "tensorlake",
                "file_id": file_id,
                "parse_id": parse_id,
                "num_chunks": len(result.chunks),
                "status": result.status.value,
            }

            # Add page count if available
            if hasattr(result, 'num_pages'):
                metadata["page_count"] = result.num_pages

            # TODO: Image extraction not yet implemented
            images = []

            logger.info(
                f"Successfully extracted {len(markdown_text)} chars "
                f"in {len(result.chunks)} chunks from {pdf_path.name}"
            )

            return markdown_text, metadata, images

        except Exception as e:
            logger.error(f"Error extracting PDF with TensorLake: {e}")
            raise RuntimeError(f"TensorLake extraction failed: {e}")

    def extract_metadata_from_markdown(
        self,
        markdown_text: str,
        tensorlake_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract enhanced metadata from markdown text

        Uses regex patterns to extract:
        - Title (from first # header or bold text)
        - Authors (from author patterns)
        - Year (from date patterns)
        - DOI (from DOI patterns)
        - Abstract (from abstract section)

        Args:
            markdown_text: Markdown text from TensorLake
            tensorlake_metadata: Original metadata from TensorLake

        Returns:
            Dictionary with enhanced metadata
        """
        metadata = tensorlake_metadata.copy()

        # Extract title (first # header or first line)
        title_match = re.search(r'^#\s+(.+)$', markdown_text, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        else:
            # Try first bold text
            bold_match = re.search(r'\*\*(.+?)\*\*', markdown_text)
            if bold_match:
                metadata["title"] = bold_match.group(1).strip()

        # Extract authors (common patterns)
        author_patterns = [
            r'Authors?:\s*(.+?)(?:\n|$)',
            r'By\s+(.+?)(?:\n|$)',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)',
        ]
        for pattern in author_patterns:
            match = re.search(pattern, markdown_text[:2000], re.MULTILINE)
            if match:
                authors_str = match.group(1)
                # Split by common separators
                authors = re.split(r'[,;]|\sand\s', authors_str)
                metadata["authors"] = [a.strip() for a in authors if a.strip()]
                break

        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', markdown_text[:2000])
        if year_match:
            metadata["year"] = int(year_match.group(0))

        # Extract DOI
        doi_match = re.search(
            r'doi:\s*([0-9]{2}\.[0-9]{4,}(?:\.[0-9]+)*/[^\s]+)',
            markdown_text,
            re.IGNORECASE
        )
        if doi_match:
            metadata["doi"] = doi_match.group(1)

        # Extract abstract
        abstract_match = re.search(
            r'(?:Abstract|ABSTRACT)[:\s]*(.+?)(?:\n\n|\n#)',
            markdown_text,
            re.IGNORECASE | re.DOTALL
        )
        if abstract_match:
            metadata["abstract"] = abstract_match.group(1).strip()

        return metadata

    @staticmethod
    def is_available() -> bool:
        """Check if TensorLake is available"""
        return TENSORLAKE_AVAILABLE
