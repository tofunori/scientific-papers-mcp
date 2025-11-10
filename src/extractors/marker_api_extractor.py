"""PDF extraction using Datalab Marker API with advanced features"""

from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
import requests
import time

logger = logging.getLogger(__name__)


class MarkerAPIExtractor:
    """
    Extract text and metadata from PDFs using Datalab Marker API

    Marker converts PDFs to markdown with high accuracy, especially for:
    - Scientific papers with equations (LaTeX)
    - Complex tables (including multi-page tables)
    - Hierarchical document structure

    API Documentation: https://documentation.datalab.to/docs/welcome/api
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://www.datalab.to/api/v1",
        timeout: int = 180,
        max_retries: int = 3
    ):
        """
        Initialize Marker API extractor

        Args:
            api_key: Datalab API key
            base_url: API base URL
            timeout: Request timeout in seconds (default: 180s for large PDFs)
            max_retries: Number of retry attempts for failed requests
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def extract_text_from_pdf(
        self,
        pdf_path: Path,
        use_llm: bool = True,
        force_ocr: bool = False,
        extract_images: bool = True,
        paginate: bool = False,
        output_format: str = "json"
    ) -> Tuple[str, Dict, List[Dict]]:
        """
        Extract text, metadata, and images from PDF using Marker API

        Args:
            pdf_path: Path to PDF file
            use_llm: Enable LLM for better table merging, inline math, etc.
            force_ocr: Force OCR on all pages (even if text is embedded)
            extract_images: Extract images from PDF
            paginate: Add page separators in markdown output
            output_format: "json", "markdown", or "html"

        Returns:
            Tuple of (markdown_text, metadata_dict, images_list)

        Raises:
            FileNotFoundError: If PDF file not found
            ValueError: If API key invalid or request fails
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting PDF using Marker API: {pdf_path.name}")
        logger.debug(f"Settings: use_llm={use_llm}, force_ocr={force_ocr}, extract_images={extract_images}")

        # Prepare request
        headers = {"X-API-Key": self.api_key}

        # API parameters
        data = {
            "output_format": output_format,
            "use_llm": str(use_llm).lower(),
            "force_ocr": str(force_ocr).lower(),
            "paginate": str(paginate).lower(),
            "disable_image_extraction": str(not extract_images).lower(),
        }

        # Make request with retry logic
        for attempt in range(self.max_retries):
            try:
                with open(pdf_path, 'rb') as f:
                    files = {'file': (pdf_path.name, f, 'application/pdf')}

                    response = requests.post(
                        f"{self.base_url}/marker",
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=self.timeout
                    )

                # Check response status
                if response.status_code == 200:
                    result = response.json()

                    # Validate response
                    if not result.get('success', False):
                        error_msg = result.get('error', 'Unknown error')
                        raise ValueError(f"Marker API returned error: {error_msg}")

                    # Extract data
                    markdown_text = result.get('markdown', '')
                    metadata = result.get('metadata', {})
                    images = result.get('images', [])
                    page_count = result.get('page_count', 0)

                    # Add page count to metadata
                    metadata['page_count'] = page_count

                    logger.info(
                        f"Successfully extracted {len(markdown_text)} chars, "
                        f"{len(images)} images, {page_count} pages from {pdf_path.name}"
                    )

                    return markdown_text, metadata, images

                elif response.status_code == 401:
                    raise ValueError("Invalid API key. Check your MARKER_API_KEY in .env")

                elif response.status_code == 429:
                    # Rate limit - retry with exponential backoff
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    if attempt < self.max_retries - 1:
                        logger.warning(f"{error_msg}. Retrying...")
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        raise ValueError(error_msg)

            except requests.Timeout:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Request timeout. Retrying... (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise ValueError(f"Request timeout after {self.max_retries} attempts")

            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Error: {e}. Retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise

        raise ValueError(f"Failed to extract PDF after {self.max_retries} attempts")

    def extract_metadata_from_markdown(self, markdown: str, api_metadata: Dict) -> Dict:
        """
        Extract enhanced metadata from Marker's markdown output and API metadata

        Marker's markdown often includes structured headers that we can parse

        Args:
            markdown: Markdown text from Marker
            api_metadata: Metadata returned by Marker API

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

        # Merge API metadata
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
