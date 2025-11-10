"""PDF extraction with Docling - IBM's document understanding library"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("Docling not installed. Install with: pip install docling")


def extract_with_docling(
    pdf_path: Path,
    extract_tables: bool = True,
    extract_images: bool = True
) -> Tuple[str, Dict, str]:
    """
    Extract document content using Docling

    Args:
        pdf_path: Path to PDF file
        extract_tables: Whether to extract tables (default: True)
        extract_images: Whether to extract images (default: True)

    Returns:
        Tuple of (markdown_text, metadata, json_structure)
        - markdown_text: Document content in Markdown format with preserved structure
        - metadata: Extracted metadata (title, authors, etc.)
        - json_structure: Full document structure in JSON format

    Raises:
        ImportError: If Docling not installed
        FileNotFoundError: If PDF file not found
    """
    if not DOCLING_AVAILABLE:
        raise ImportError(
            "Docling is required for this feature. "
            "Install with: pip install docling"
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info(f"Extracting PDF with Docling: {pdf_path.name}")

    try:
        # Configure PDF pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = extract_tables
        pipeline_options.do_ocr = True  # Enable OCR for scanned PDFs

        # Create converter with options
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        # Convert document
        result = converter.convert(pdf_path)

        # Extract markdown with structure
        markdown_content = result.document.export_to_markdown()

        # Extract metadata
        metadata = _extract_metadata_from_docling(result.document)

        # Export full document structure as JSON
        json_structure = result.document.export_to_dict()

        logger.info(f"Successfully extracted {len(markdown_content)} characters with Docling")

        return markdown_content, metadata, json_structure

    except Exception as e:
        logger.error(f"Error extracting PDF with Docling: {e}")
        raise


def _extract_metadata_from_docling(document) -> Dict:
    """
    Extract metadata from Docling document

    Args:
        document: Docling document object

    Returns:
        Dictionary with metadata fields
    """
    metadata = {
        "title": None,
        "authors": [],
        "year": None,
        "journal": None,
        "doi": None,
        "abstract": None,
        "keywords": [],
        "page_count": None,
        "sections": [],
        "tables": [],
        "figures": [],
    }

    try:
        # Extract basic metadata
        if hasattr(document, 'title') and document.title:
            metadata["title"] = str(document.title)

        if hasattr(document, 'authors') and document.authors:
            metadata["authors"] = [str(author) for author in document.authors]

        # Extract DOI from document metadata or text
        if hasattr(document, 'metadata'):
            doc_meta = document.metadata
            if hasattr(doc_meta, 'doi') and doc_meta.doi:
                metadata["doi"] = str(doc_meta.doi)
            if hasattr(doc_meta, 'year') and doc_meta.year:
                metadata["year"] = int(doc_meta.year)

        # Count pages
        if hasattr(document, 'pages'):
            metadata["page_count"] = len(document.pages)

        # Extract section headings (hierarchical structure)
        if hasattr(document, 'body'):
            sections = []
            for item in document.body:
                if hasattr(item, 'label') and 'section' in str(item.label).lower():
                    if hasattr(item, 'text'):
                        sections.append(str(item.text))
            metadata["sections"] = sections

        # Extract tables
        if hasattr(document, 'tables'):
            tables_info = []
            for table in document.tables:
                table_data = {
                    "caption": getattr(table, 'caption', None),
                    "num_rows": getattr(table, 'num_rows', None),
                    "num_cols": getattr(table, 'num_cols', None),
                }
                tables_info.append(table_data)
            metadata["tables"] = tables_info

        # Extract figures
        if hasattr(document, 'figures'):
            figures_info = []
            for figure in document.figures:
                figure_data = {
                    "caption": getattr(figure, 'caption', None),
                    "page": getattr(figure, 'page', None),
                }
                figures_info.append(figure_data)
            metadata["figures"] = figures_info

    except Exception as e:
        logger.warning(f"Error extracting metadata from Docling document: {e}")

    return metadata


def extract_tables_from_docling(pdf_path: Path) -> List[Dict]:
    """
    Extract structured tables from PDF using Docling

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of dictionaries, each containing table data:
        - caption: Table caption/title
        - data: 2D array of cell contents
        - page: Page number
        - markdown: Markdown representation of the table
    """
    if not DOCLING_AVAILABLE:
        raise ImportError("Docling is required. Install with: pip install docling")

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info(f"Extracting tables with Docling: {pdf_path.name}")

    try:
        # Configure for table extraction
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(pdf_path)

        tables = []
        if hasattr(result.document, 'tables'):
            for idx, table in enumerate(result.document.tables):
                table_dict = {
                    "index": idx,
                    "caption": getattr(table, 'caption', None),
                    "page": getattr(table, 'page', None),
                    "data": None,
                    "markdown": None,
                }

                # Try to get table data
                if hasattr(table, 'data'):
                    table_dict["data"] = table.data

                # Try to export to markdown
                if hasattr(table, 'export_to_markdown'):
                    table_dict["markdown"] = table.export_to_markdown()

                tables.append(table_dict)

        logger.info(f"Extracted {len(tables)} tables from PDF")
        return tables

    except Exception as e:
        logger.error(f"Error extracting tables with Docling: {e}")
        raise


def compare_extractions(pdf_path: Path) -> Dict:
    """
    Compare PyMuPDF and Docling extractions side-by-side

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dictionary with comparison data
    """
    from .pdf_extractor import extract_text_from_pdf, extract_metadata_from_pdf

    pdf_path = Path(pdf_path)

    comparison = {
        "file": pdf_path.name,
        "pymupdf": {},
        "docling": {},
        "differences": {}
    }

    # PyMuPDF extraction
    try:
        logger.info("Extracting with PyMuPDF...")
        pymupdf_text, is_scanned, images = extract_text_from_pdf(pdf_path, extract_images=False)
        pymupdf_metadata = extract_metadata_from_pdf(pdf_path)

        comparison["pymupdf"] = {
            "text_length": len(pymupdf_text),
            "word_count": len(pymupdf_text.split()),
            "is_scanned": is_scanned,
            "metadata": pymupdf_metadata,
            "text_preview": pymupdf_text[:500],
        }
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")
        comparison["pymupdf"]["error"] = str(e)

    # Docling extraction
    try:
        logger.info("Extracting with Docling...")
        docling_markdown, docling_metadata, json_structure = extract_with_docling(pdf_path)

        comparison["docling"] = {
            "text_length": len(docling_markdown),
            "word_count": len(docling_markdown.split()),
            "metadata": docling_metadata,
            "text_preview": docling_markdown[:500],
            "num_sections": len(docling_metadata.get("sections", [])),
            "num_tables": len(docling_metadata.get("tables", [])),
            "num_figures": len(docling_metadata.get("figures", [])),
        }
    except Exception as e:
        logger.error(f"Docling extraction failed: {e}")
        comparison["docling"]["error"] = str(e)

    # Calculate differences
    if "error" not in comparison["pymupdf"] and "error" not in comparison["docling"]:
        comparison["differences"] = {
            "text_length_diff": comparison["docling"]["text_length"] - comparison["pymupdf"]["text_length"],
            "word_count_diff": comparison["docling"]["word_count"] - comparison["pymupdf"]["word_count"],
            "metadata_completeness": {
                "pymupdf": sum(1 for v in comparison["pymupdf"]["metadata"].values() if v),
                "docling": sum(1 for v in comparison["docling"]["metadata"].values() if v),
            }
        }

    return comparison
