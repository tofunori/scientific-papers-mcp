"""PDF extraction using local Marker installation"""

from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
import json

logger = logging.getLogger(__name__)

# Try to import marker
try:
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False
    logger.warning("Marker not installed. Install with: pip install marker-pdf")


class MarkerLocalExtractor:
    """
    Extract text and metadata from PDFs using local Marker installation

    Marker converts PDFs to markdown with high accuracy, especially for:
    - Scientific papers with equations (LaTeX)
    - Complex tables (including multi-page tables)
    - Hierarchical document structure

    Installation: pip install marker-pdf
    Documentation: https://github.com/datalab-to/marker
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        batch_multiplier: int = 1,
    ):
        """
        Initialize Marker local extractor

        Args:
            use_llm: Enable LLM for better quality (requires API key)
            llm_provider: LLM provider (e.g., "openai", "anthropic", "google")
            llm_model: LLM model name (e.g., "gpt-4", "claude-3-sonnet")
            batch_multiplier: Increase for better GPU utilization (default: 1)

        Raises:
            ImportError: If marker-pdf not installed
        """
        if not MARKER_AVAILABLE:
            raise ImportError(
                "marker-pdf is not installed. Install with:\n"
                "pip install marker-pdf\n\n"
                "For GPU support (recommended), also install:\n"
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            )

        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.batch_multiplier = batch_multiplier

        # Load models once (expensive operation)
        logger.info("Loading Marker models (this may take a few minutes on first run)...")
        try:
            self.models = load_all_models()
            logger.info("Marker models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Marker models: {e}")
            raise

    def extract_text_from_pdf(
        self,
        pdf_path: Path,
        extract_images: bool = True,
        output_format: str = "markdown"
    ) -> Tuple[str, Dict, List[Dict]]:
        """
        Extract text, metadata, and images from PDF using local Marker

        Args:
            pdf_path: Path to PDF file
            extract_images: Extract images from PDF
            output_format: "markdown" or "json"

        Returns:
            Tuple of (markdown_text, metadata_dict, images_list)

        Raises:
            FileNotFoundError: If PDF file not found
            ImportError: If marker-pdf not installed
        """
        if not MARKER_AVAILABLE:
            raise ImportError("marker-pdf is not installed")

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting PDF using local Marker: {pdf_path.name}")
        logger.debug(f"Settings: use_llm={self.use_llm}, extract_images={extract_images}")

        try:
            # Convert PDF to markdown
            result = convert_single_pdf(
                str(pdf_path),
                self.models,
                batch_multiplier=self.batch_multiplier,
            )

            # Extract results
            markdown_text = result.get("markdown", "")
            metadata = result.get("metadata", {})
            images = result.get("images", {}) if extract_images else {}

            # Convert images dict to list format (compatible with API extractor)
            images_list = []
            if images:
                for page_num, page_images in images.items():
                    for img_data in page_images:
                        images_list.append({
                            "page_num": page_num,
                            "image_base64": img_data.get("image", ""),
                            "width": img_data.get("width", 0),
                            "height": img_data.get("height", 0),
                            "format": "png"  # Marker typically outputs PNG
                        })

            # Add page count to metadata
            if "page_count" not in metadata:
                # Count pages from markdown page separators if paginate was enabled
                page_count = markdown_text.count("---PAGE---") + 1 if "---PAGE---" in markdown_text else 0
                metadata["page_count"] = page_count

            logger.info(
                f"Successfully extracted {len(markdown_text)} chars, "
                f"{len(images_list)} images from {pdf_path.name}"
            )

            return markdown_text, metadata, images_list

        except Exception as e:
            logger.error(f"Error extracting PDF with Marker: {e}")
            raise

    def extract_metadata_from_markdown(self, markdown: str, api_metadata: Dict) -> Dict:
        """
        Extract enhanced metadata from Marker's markdown output

        Args:
            markdown: Markdown text from Marker
            api_metadata: Metadata returned by Marker

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
            "page_count": api_metadata.get("page_count", 0)
        }

        # Merge Marker metadata
        if api_metadata:
            metadata.update({k: v for k, v in api_metadata.items() if v})

        # Parse markdown for additional metadata
        lines = markdown.split('\n')

        # Try to extract title (usually first # heading)
        for line in lines[:20]:
            line = line.strip()
            if line.startswith('# ') and not metadata['title']:
                metadata['title'] = line[2:].strip()
                break

        # Try to extract abstract (look for Abstract section)
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
                if len(' '.join(abstract_lines)) > 500:  # Limit abstract length
                    break

        if abstract_lines:
            metadata['abstract'] = ' '.join(abstract_lines)

        return metadata

    @staticmethod
    def is_available() -> bool:
        """Check if Marker is available"""
        return MARKER_AVAILABLE
