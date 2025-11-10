"""Structured metadata extraction using LlamaExtract API

LlamaExtract is LlamaIndex's structured data extraction solution:
- Schema-driven extraction with confidence scores
- Extract complex nested structures
- Multi-format support (PDF, DOCX, images)
- Vision-based extraction for tables and charts
- Automatic citation and source tracking

Perfect for extracting structured metadata from scientific papers:
- Title, authors, abstract, keywords
- Methodology, findings, conclusions
- References, figures, tables
- Dataset information

API Documentation: https://docs.cloud.llamaindex.ai/llamaextract
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)

# Try to import llama-extract
try:
    from llama_extract import LlamaExtract
    LLAMAEXTRACT_AVAILABLE = True
except ImportError:
    LLAMAEXTRACT_AVAILABLE = False
    logger.warning("llama-extract not installed. Install with: pip install llama-extract")


# Scientific paper extraction schema
SCIENTIFIC_PAPER_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Full title of the scientific paper"
        },
        "authors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of all authors in order"
        },
        "year": {
            "type": "integer",
            "description": "Publication year"
        },
        "abstract": {
            "type": "string",
            "description": "Full abstract text"
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Keywords or key concepts from the paper"
        },
        "doi": {
            "type": "string",
            "description": "DOI identifier if available"
        },
        "journal": {
            "type": "string",
            "description": "Journal or conference name"
        },
        "methodology": {
            "type": "string",
            "description": "Summary of research methodology and approach"
        },
        "main_findings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key findings and results from the research"
        },
        "datasets_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Datasets mentioned or used in the research"
        },
        "conclusions": {
            "type": "string",
            "description": "Main conclusions from the paper"
        },
        "references_count": {
            "type": "integer",
            "description": "Number of references cited"
        },
    },
    "required": ["title", "authors"],
}


class LlamaExtractExtractor:
    """
    Extract structured metadata from PDFs using LlamaExtract API

    LlamaExtract uses vision models to extract structured data from documents:
    - Schema-driven extraction (define what you want to extract)
    - Confidence scores for each field
    - Vision-based (works on scanned PDFs and images)
    - Handles tables, figures, and complex layouts

    Perfect for scientific papers where structured metadata is crucial.

    Pricing:
    - Free tier available
    - Pay-per-use for larger volumes
    - Much cheaper than manual extraction

    API: https://cloud.llamaindex.ai/
    """

    def __init__(
        self,
        api_key: str,
        schema: Optional[Dict] = None,
        schema_name: str = "scientific_paper",
    ):
        """
        Initialize LlamaExtract extractor

        Args:
            api_key: LlamaCloud API key (same as LlamaParse)
            schema: Custom extraction schema (JSON Schema format)
            schema_name: Name for the schema

        Raises:
            ImportError: If llama-extract not installed
        """
        if not LLAMAEXTRACT_AVAILABLE:
            raise ImportError(
                "llama-extract is not installed. Install with:\\n"
                "pip install llama-extract"
            )

        self.api_key = api_key
        self.schema_name = schema_name

        # Use provided schema or default scientific paper schema
        self.schema = schema or SCIENTIFIC_PAPER_SCHEMA

        # Initialize LlamaExtract client
        self.client = LlamaExtract(api_key=api_key)

        # Create or update extraction schema
        try:
            self.client.create_schema(
                schema_name=schema_name,
                schema=self.schema,
            )
            logger.info(f"LlamaExtract schema '{schema_name}' created/updated")
        except Exception as e:
            # Schema might already exist
            logger.debug(f"Schema creation note: {e}")

        logger.info("LlamaExtract extractor initialized")

    def extract_metadata_from_pdf(
        self,
        pdf_path: Path,
        schema_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract structured metadata from PDF using LlamaExtract

        Args:
            pdf_path: Path to PDF file
            schema_name: Schema to use (default: self.schema_name)

        Returns:
            Dictionary with extracted metadata and confidence scores

        Raises:
            FileNotFoundError: If PDF file not found
            ValueError: If extraction fails
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        schema_name = schema_name or self.schema_name

        logger.info(f"Extracting structured metadata from: {pdf_path.name}")

        try:
            # Extract structured data using LlamaExtract
            result = self.client.extract(
                file_path=str(pdf_path),
                schema_name=schema_name,
            )

            # Parse extraction result
            extracted_data = {}
            confidence_scores = {}

            if hasattr(result, 'data'):
                extracted_data = result.data

            if hasattr(result, 'confidence'):
                confidence_scores = result.confidence

            # Build final metadata with confidence scores
            metadata = {
                "extraction_method": "llamaextract",
                "schema_name": schema_name,
                "extracted_data": extracted_data,
                "confidence_scores": confidence_scores,
                "overall_confidence": self._calculate_overall_confidence(confidence_scores),
            }

            # Flatten extracted data to top level for easier access
            if extracted_data:
                for key, value in extracted_data.items():
                    if key not in metadata:
                        metadata[key] = value

            logger.info(
                f"Successfully extracted metadata from {pdf_path.name} "
                f"(confidence: {metadata['overall_confidence']:.2%})"
            )

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata with LlamaExtract: {e}")
            raise ValueError(f"LlamaExtract extraction failed: {e}")

    def _calculate_overall_confidence(self, confidence_scores: Dict[str, float]) -> float:
        """
        Calculate overall confidence score from field-level scores

        Args:
            confidence_scores: Dictionary of field -> confidence score

        Returns:
            Average confidence score (0.0 to 1.0)
        """
        if not confidence_scores:
            return 0.0

        scores = [score for score in confidence_scores.values() if isinstance(score, (int, float))]
        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def extract_with_fallback(
        self,
        pdf_path: Path,
        fallback_metadata: Optional[Dict] = None,
        min_confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Extract metadata with fallback to provided metadata if confidence is low

        Args:
            pdf_path: Path to PDF file
            fallback_metadata: Fallback metadata from other extractors (PyMuPDF, Marker, etc.)
            min_confidence: Minimum confidence threshold (0.0 to 1.0)

        Returns:
            Merged metadata with best available data
        """
        try:
            # Try LlamaExtract extraction
            extracted = self.extract_metadata_from_pdf(pdf_path)

            # If no fallback metadata, return extracted data
            if not fallback_metadata:
                return extracted

            # Merge with fallback metadata based on confidence
            merged = fallback_metadata.copy()

            for key, value in extracted.get("extracted_data", {}).items():
                # Get confidence for this field
                field_confidence = extracted.get("confidence_scores", {}).get(key, 0.0)

                # Use extracted value if confidence is high enough
                # OR if fallback doesn't have this field
                if field_confidence >= min_confidence or key not in merged:
                    merged[key] = value
                    merged[f"{key}_confidence"] = field_confidence

            # Add extraction metadata
            merged["llamaextract_used"] = True
            merged["overall_confidence"] = extracted.get("overall_confidence", 0.0)

            logger.info(
                f"Merged LlamaExtract with fallback metadata "
                f"(confidence: {merged['overall_confidence']:.2%})"
            )

            return merged

        except Exception as e:
            logger.warning(f"LlamaExtract failed, using fallback metadata: {e}")
            if fallback_metadata:
                fallback_metadata["llamaextract_used"] = False
                return fallback_metadata
            else:
                # No fallback, re-raise exception
                raise

    def batch_extract(
        self,
        pdf_paths: List[Path],
        schema_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract metadata from multiple PDFs in batch

        Args:
            pdf_paths: List of PDF file paths
            schema_name: Schema to use (default: self.schema_name)

        Returns:
            List of extracted metadata dictionaries
        """
        results = []

        for pdf_path in pdf_paths:
            try:
                metadata = self.extract_metadata_from_pdf(pdf_path, schema_name)
                results.append(metadata)
            except Exception as e:
                logger.error(f"Failed to extract {pdf_path.name}: {e}")
                results.append({
                    "file_path": str(pdf_path),
                    "error": str(e),
                    "extraction_method": "llamaextract",
                })

        logger.info(f"Batch extraction completed: {len(results)} documents")
        return results

    @staticmethod
    def is_available() -> bool:
        """Check if LlamaExtract is available"""
        return LLAMAEXTRACT_AVAILABLE

    def update_schema(self, schema: Dict, schema_name: Optional[str] = None):
        """
        Update extraction schema

        Args:
            schema: New schema definition (JSON Schema format)
            schema_name: Schema name (default: self.schema_name)

        Example:
            # Add new field to schema
            new_schema = {**SCIENTIFIC_PAPER_SCHEMA}
            new_schema["properties"]["citation_count"] = {
                "type": "integer",
                "description": "Number of times this paper has been cited"
            }
            extractor.update_schema(new_schema)
        """
        schema_name = schema_name or self.schema_name

        self.schema = schema
        self.client.create_schema(
            schema_name=schema_name,
            schema=schema,
        )

        logger.info(f"Schema '{schema_name}' updated successfully")
