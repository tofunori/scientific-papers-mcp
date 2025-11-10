"""PDF extraction using local Marker installation with LLM support"""

from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
import json
import os

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

    LLM Support:
    - Gemini (default): Requires GOOGLE_API_KEY environment variable
    - Vertex AI: Requires GCP project ID (more reliable than Gemini)
    - Claude: Requires Anthropic API key
    - OpenAI: Requires OpenAI API key
    - Ollama: Free local LLM (no API key needed)

    Installation: pip install marker-pdf
    Documentation: https://github.com/datalab-to/marker
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_service: str = "gemini",
        batch_multiplier: int = 1,
        # Gemini settings
        google_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.0-flash",
        # Vertex settings
        vertex_project_id: Optional[str] = None,
        # Claude settings
        claude_api_key: Optional[str] = None,
        claude_model: str = "claude-3-sonnet-20240229",
        # OpenAI settings
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4-turbo",
        # Ollama settings
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama2",
    ):
        """
        Initialize Marker local extractor

        Args:
            use_llm: Enable LLM for better quality (default: False)
            llm_service: LLM service to use: "gemini", "vertex", "claude", "openai", "ollama"
            batch_multiplier: Increase for better GPU utilization (default: 1)
            google_api_key: Gemini API key (or set GOOGLE_API_KEY env var)
            gemini_model: Gemini model name (default: gemini-2.0-flash)
            vertex_project_id: GCP project ID for Vertex AI
            claude_api_key: Anthropic Claude API key
            claude_model: Claude model name
            openai_api_key: OpenAI API key
            openai_model: OpenAI model name
            ollama_base_url: Ollama server URL
            ollama_model: Ollama model name

        Raises:
            ImportError: If marker-pdf not installed
            ValueError: If LLM enabled but credentials missing
        """
        if not MARKER_AVAILABLE:
            raise ImportError(
                "marker-pdf is not installed. Install with:\n"
                "pip install marker-pdf\n\n"
                "For GPU support (recommended), also install:\n"
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            )

        self.use_llm = use_llm
        self.llm_service = llm_service
        self.batch_multiplier = batch_multiplier

        # Store LLM configuration
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.gemini_model = gemini_model
        self.vertex_project_id = vertex_project_id
        self.claude_api_key = claude_api_key
        self.claude_model = claude_model
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model

        # Validate LLM configuration if enabled
        if use_llm:
            self._validate_llm_config()

        # Load models once (expensive operation)
        logger.info("Loading Marker models (this may take a few minutes on first run)...")
        try:
            self.models = load_all_models()
            logger.info("Marker models loaded successfully")
            if use_llm:
                logger.info(f"LLM enabled: {llm_service} ({self._get_llm_model_name()})")
        except Exception as e:
            logger.error(f"Failed to load Marker models: {e}")
            raise

    def _validate_llm_config(self):
        """Validate LLM configuration based on selected service"""
        if self.llm_service == "gemini":
            if not self.google_api_key:
                raise ValueError(
                    "Gemini API key required for LLM. Set GOOGLE_API_KEY environment variable "
                    "or pass google_api_key parameter."
                )
        elif self.llm_service == "vertex":
            if not self.vertex_project_id:
                raise ValueError(
                    "Vertex AI requires vertex_project_id. Set MARKER_VERTEX_PROJECT_ID "
                    "or pass vertex_project_id parameter."
                )
        elif self.llm_service == "claude":
            if not self.claude_api_key:
                raise ValueError(
                    "Claude API key required. Set CLAUDE_API_KEY or pass claude_api_key parameter."
                )
        elif self.llm_service == "openai":
            if not self.openai_api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY or pass openai_api_key parameter."
                )
        elif self.llm_service == "ollama":
            # Ollama doesn't require API key, just URL
            logger.info(f"Using Ollama at {self.ollama_base_url}")
        else:
            raise ValueError(
                f"Unknown LLM service: {self.llm_service}. "
                f"Valid options: gemini, vertex, claude, openai, ollama"
            )

    def _get_llm_model_name(self) -> str:
        """Get the current LLM model name"""
        if self.llm_service == "gemini":
            return self.gemini_model
        elif self.llm_service == "vertex":
            return "vertex-ai"
        elif self.llm_service == "claude":
            return self.claude_model
        elif self.llm_service == "openai":
            return self.openai_model
        elif self.llm_service == "ollama":
            return self.ollama_model
        return "unknown"

    def _prepare_llm_env(self):
        """Prepare environment variables for LLM"""
        # Set Gemini API key if using Gemini
        if self.llm_service == "gemini" and self.google_api_key:
            os.environ["GOOGLE_API_KEY"] = self.google_api_key

        # Set Claude API key if using Claude
        if self.llm_service == "claude" and self.claude_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.claude_api_key

        # Set OpenAI API key if using OpenAI
        if self.llm_service == "openai" and self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key

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
            # Prepare LLM environment if enabled
            if self.use_llm:
                self._prepare_llm_env()
                logger.debug(f"LLM environment prepared: {self.llm_service}")

            # Note: As of marker-pdf 1.x, LLM configuration is primarily done via CLI args
            # For programmatic use, we set environment variables and rely on marker's defaults
            # Full LLM control requires using marker CLI: marker_single --use_llm --gemini_api_key ...

            # Convert PDF to markdown
            result = convert_single_pdf(
                str(pdf_path),
                self.models,
                batch_multiplier=self.batch_multiplier,
                # LLM is configured via environment variables set in _prepare_llm_env()
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
