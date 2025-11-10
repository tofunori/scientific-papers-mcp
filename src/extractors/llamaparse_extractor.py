"""PDF extraction using LlamaParse API

LlamaParse is LlamaIndex's GenAI-native document parsing solution:
- Superior table/chart/equation extraction
- Natural language parsing instructions
- Multi-format support (PDF, DOCX, PPTX, ePub)
- More affordable than alternatives ($0.003/page)
- 7,000 free pages per week

API Documentation: https://docs.cloud.llamaindex.ai/
"""

from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
import time
import base64

logger = logging.getLogger(__name__)

# Try to import llama-parse
try:
    from llama_parse import LlamaParse
    LLAMAPARSE_AVAILABLE = True
except ImportError:
    LLAMAPARSE_AVAILABLE = False
    logger.warning("llama-parse not installed. Install with: pip install llama-parse")


class LlamaParseExtractor:
    """
    Extract text and metadata from PDFs using LlamaParse API

    LlamaParse converts documents to markdown with:
    - GenAI-native parsing (better than traditional OCR)
    - Natural language instructions for custom parsing
    - Superior table, chart, and equation extraction
    - Multi-format support (PDF, DOCX, PPTX, ePub)

    Pricing:
    - Free: 7,000 pages/week
    - Paid: $0.003/page beyond free tier
    - Much cheaper than alternatives

    API: https://cloud.llamaindex.ai/
    """

    def __init__(
        self,
        api_key: str,
        result_type: str = "markdown",
        parsing_instruction: Optional[str] = None,
        use_vendor_multimodal: bool = False,
        invalidate_cache: bool = False,
        num_workers: int = 4,
        max_timeout: int = 2000,
    ):
        """
        Initialize LlamaParse extractor

        Args:
            api_key: LlamaCloud API key
            result_type: Output format ("markdown" or "text")
            parsing_instruction: Natural language instructions for parsing
            use_vendor_multimodal: Use vendor multimodal models (more expensive but better)
            invalidate_cache: Force re-parsing (ignore cache)
            num_workers: Number of parallel workers
            max_timeout: Maximum timeout in seconds

        Raises:
            ImportError: If llama-parse not installed
        """
        if not LLAMAPARSE_AVAILABLE:
            raise ImportError(
                "llama-parse is not installed. Install with:\n"
                "pip install llama-parse"
            )

        self.api_key = api_key

        # Default parsing instruction optimized for scientific papers
        if parsing_instruction is None:
            parsing_instruction = (
                "Extract all text preserving document structure. "
                "For tables: preserve formatting and alignment. "
                "For equations: extract as LaTeX when possible. "
                "For figures: include captions and descriptions. "
                "Maintain section headers and hierarchy."
            )

        # Initialize LlamaParse
        self.parser = LlamaParse(
            api_key=api_key,
            result_type=result_type,
            parsing_instruction=parsing_instruction,
            use_vendor_multimodal_model=use_vendor_multimodal,
            invalidate_cache=invalidate_cache,
            num_workers=num_workers,
            max_timeout=max_timeout,
        )

        logger.info("LlamaParse extractor initialized")
        logger.debug(f"Parsing instruction: {parsing_instruction[:100]}...")

    def extract_text_from_pdf(
        self,
        pdf_path: Path,
        extract_images: bool = True,
        custom_instruction: Optional[str] = None,
    ) -> Tuple[str, Dict, List[Dict]]:
        """
        Extract text, metadata, and images from PDF using LlamaParse

        Args:
            pdf_path: Path to PDF file
            extract_images: Extract images (currently returns empty list, images embedded in markdown)
            custom_instruction: Override default parsing instruction

        Returns:
            Tuple of (markdown_text, metadata_dict, images_list)

        Raises:
            FileNotFoundError: If PDF file not found
            ValueError: If parsing fails
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting PDF using LlamaParse: {pdf_path.name}")

        try:
            start_time = time.time()

            # Update parsing instruction if custom provided
            if custom_instruction:
                self.parser.parsing_instruction = custom_instruction
                logger.debug(f"Using custom instruction: {custom_instruction[:100]}...")

            # Parse PDF (returns list of Document objects)
            documents = self.parser.load_data(str(pdf_path))

            elapsed = time.time() - start_time

            # Combine all document pages
            markdown_text = "\n\n".join([doc.text for doc in documents])

            # Extract metadata from LlamaParse response
            metadata = {}
            if documents:
                # Get metadata from first document
                first_doc = documents[0]
                metadata = first_doc.metadata if hasattr(first_doc, 'metadata') else {}

                # Add page count
                metadata['page_count'] = len(documents)

                # Add file info
                metadata['file_name'] = pdf_path.name
                metadata['file_size'] = pdf_path.stat().st_size

            # LlamaParse embeds images in markdown, so images_list is currently empty
            # Future: could extract base64 images from markdown if needed
            images_list = []

            logger.info(
                f"Successfully extracted {len(markdown_text)} chars, "
                f"{len(documents)} pages from {pdf_path.name} in {elapsed:.2f}s"
            )

            return markdown_text, metadata, images_list

        except Exception as e:
            logger.error(f"Error extracting PDF with LlamaParse: {e}")
            raise ValueError(f"LlamaParse extraction failed: {e}")

    def extract_metadata_from_markdown(self, markdown: str, parse_metadata: Dict) -> Dict:
        """
        Extract enhanced metadata from LlamaParse's markdown output

        Args:
            markdown: Markdown text from LlamaParse
            parse_metadata: Metadata returned by LlamaParse

        Returns:
            Enhanced metadata dictionary
        """
        metadata = {
            "title": None,
            "authors": [],
            "year": None,
            "journal": None,
            "doi": None,
            "abstract": None,
            "keywords": [],
            "page_count": parse_metadata.get("page_count", 0),
            "file_name": parse_metadata.get("file_name", ""),
        }

        # Merge LlamaParse metadata
        if parse_metadata:
            metadata.update({k: v for k, v in parse_metadata.items() if v})

        # Parse markdown for additional metadata (similar to Marker)
        lines = markdown.split('\n')

        # Extract title (usually first # heading)
        for line in lines[:20]:
            line = line.strip()
            if line.startswith('# ') and not metadata['title']:
                metadata['title'] = line[2:].strip()
                break

        # Extract abstract (look for Abstract section)
        abstract_section = False
        abstract_lines = []
        for line in lines:
            if line.strip().lower() in ['## abstract', '# abstract', '**abstract**']:
                abstract_section = True
                continue
            if abstract_section:
                if line.strip().startswith('#') or line.strip().startswith('##'):
                    break
                if line.strip():
                    abstract_lines.append(line.strip())
                if len(' '.join(abstract_lines)) > 500:
                    break

        if abstract_lines:
            metadata['abstract'] = ' '.join(abstract_lines)

        return metadata

    @staticmethod
    def is_available() -> bool:
        """Check if LlamaParse is available"""
        return LLAMAPARSE_AVAILABLE

    def set_custom_instruction(self, instruction: str):
        """
        Update parsing instruction

        Args:
            instruction: Natural language parsing instruction

        Example:
            extractor.set_custom_instruction(
                "Focus on extracting methodology sections. "
                "Preserve all equations in LaTeX format. "
                "Extract all tables with full formatting."
            )
        """
        self.parser.parsing_instruction = instruction
        logger.info(f"Updated parsing instruction: {instruction[:100]}...")
