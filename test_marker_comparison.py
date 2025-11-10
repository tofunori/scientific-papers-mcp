"""
Test script to compare PyMuPDF vs Marker for PDF extraction
"""

import sys
from pathlib import Path
import time
import logging
from typing import List, Dict
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from extractors.pdf_extractor import extract_text_from_pdf, extract_metadata_from_pdf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_with_marker(pdf_path: Path) -> Dict:
    """
    Extract content from PDF using Marker

    Returns:
        Dict with keys: text, metadata, images, processing_time
    """
    try:
        import marker
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        start_time = time.time()

        # Create models (loaded once, cached)
        models = create_model_dict()

        # Create converter
        converter = PdfConverter(artifact_dict=models)

        # Convert PDF
        rendered = converter(str(pdf_path))

        processing_time = time.time() - start_time

        # Extract content
        markdown_text = rendered.markdown
        metadata = rendered.metadata.model_dump()
        images = []

        # Extract images if available
        if hasattr(rendered, 'images') and rendered.images:
            for img_name, img_data in rendered.images.items():
                images.append({
                    'name': img_name,
                    'data': img_data
                })

        return {
            'text': markdown_text,
            'metadata': metadata,
            'images': images,
            'processing_time': processing_time,
            'text_length': len(markdown_text),
            'success': True,
            'error': None
        }

    except Exception as e:
        logger.error(f"Marker extraction failed: {e}")
        return {
            'text': None,
            'metadata': None,
            'images': [],
            'processing_time': 0,
            'text_length': 0,
            'success': False,
            'error': str(e)
        }


def extract_with_pymupdf(pdf_path: Path) -> Dict:
    """
    Extract content from PDF using current PyMuPDF method

    Returns:
        Dict with keys: text, metadata, images, processing_time
    """
    try:
        start_time = time.time()

        # Extract text and images
        text, is_scanned, images = extract_text_from_pdf(pdf_path, extract_images=True)

        # Extract metadata
        metadata = extract_metadata_from_pdf(pdf_path)

        processing_time = time.time() - start_time

        return {
            'text': text,
            'metadata': metadata,
            'images': images,
            'is_scanned': is_scanned,
            'processing_time': processing_time,
            'text_length': len(text),
            'success': True,
            'error': None
        }

    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")
        return {
            'text': None,
            'metadata': None,
            'images': [],
            'processing_time': 0,
            'text_length': 0,
            'success': False,
            'error': str(e)
        }


def compare_extractions(pdf_path: Path, output_dir: Path = None) -> Dict:
    """
    Compare Marker vs PyMuPDF extraction on a single PDF

    Args:
        pdf_path: Path to PDF file
        output_dir: Optional directory to save extracted text samples

    Returns:
        Comparison results dictionary
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing PDF: {pdf_path.name}")
    logger.info(f"{'='*80}\n")

    # Extract with both methods
    logger.info("🔍 Extracting with PyMuPDF (current method)...")
    pymupdf_result = extract_with_pymupdf(pdf_path)

    logger.info("🔍 Extracting with Marker...")
    marker_result = extract_with_marker(pdf_path)

    # Prepare comparison
    comparison = {
        'pdf_name': pdf_path.name,
        'pdf_size_mb': pdf_path.stat().st_size / (1024 * 1024),
        'pymupdf': {
            'success': pymupdf_result['success'],
            'processing_time': pymupdf_result['processing_time'],
            'text_length': pymupdf_result['text_length'],
            'num_images': len(pymupdf_result.get('images', [])),
            'metadata': pymupdf_result.get('metadata', {}),
            'error': pymupdf_result.get('error')
        },
        'marker': {
            'success': marker_result['success'],
            'processing_time': marker_result['processing_time'],
            'text_length': marker_result['text_length'],
            'num_images': len(marker_result.get('images', [])),
            'metadata': marker_result.get('metadata', {}),
            'error': marker_result.get('error')
        }
    }

    # Print comparison results
    print("\n" + "="*80)
    print(f"📊 COMPARISON RESULTS: {pdf_path.name}")
    print("="*80)

    print(f"\n📄 PDF Info:")
    print(f"   Size: {comparison['pdf_size_mb']:.2f} MB")

    print(f"\n⏱️  Processing Time:")
    print(f"   PyMuPDF: {pymupdf_result['processing_time']:.2f}s")
    print(f"   Marker:  {marker_result['processing_time']:.2f}s")
    if pymupdf_result['success'] and marker_result['success']:
        speedup = marker_result['processing_time'] / pymupdf_result['processing_time']
        print(f"   Ratio:   {speedup:.2f}x ({'Marker slower' if speedup > 1 else 'Marker faster'})")

    print(f"\n📝 Extracted Text Length:")
    print(f"   PyMuPDF: {pymupdf_result['text_length']:,} chars")
    print(f"   Marker:  {marker_result['text_length']:,} chars")
    if pymupdf_result['text_length'] > 0 and marker_result['text_length'] > 0:
        ratio = marker_result['text_length'] / pymupdf_result['text_length']
        print(f"   Ratio:   {ratio:.2f}x")

    print(f"\n🖼️  Images Extracted:")
    print(f"   PyMuPDF: {len(pymupdf_result.get('images', []))}")
    print(f"   Marker:  {len(marker_result.get('images', []))}")

    print(f"\n✅ Success Status:")
    print(f"   PyMuPDF: {'✓' if pymupdf_result['success'] else '✗'}")
    print(f"   Marker:  {'✓' if marker_result['success'] else '✗'}")

    if pymupdf_result.get('error'):
        print(f"\n❌ PyMuPDF Error: {pymupdf_result['error']}")
    if marker_result.get('error'):
        print(f"\n❌ Marker Error: {marker_result['error']}")

    # Save text samples if requested
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save PyMuPDF text
        if pymupdf_result['success'] and pymupdf_result['text']:
            pymupdf_file = output_dir / f"{pdf_path.stem}_pymupdf.txt"
            pymupdf_file.write_text(pymupdf_result['text'][:5000])  # First 5000 chars
            print(f"\n💾 Saved PyMuPDF sample: {pymupdf_file}")

        # Save Marker text
        if marker_result['success'] and marker_result['text']:
            marker_file = output_dir / f"{pdf_path.stem}_marker.md"
            marker_file.write_text(marker_result['text'][:5000])  # First 5000 chars
            print(f"💾 Saved Marker sample: {marker_file}")

    print("\n" + "="*80 + "\n")

    return comparison


def find_sample_pdfs(documents_path: Path, limit: int = 5) -> List[Path]:
    """
    Find sample PDFs from Zotero library

    Args:
        documents_path: Path to Zotero storage
        limit: Maximum number of PDFs to test

    Returns:
        List of PDF paths
    """
    pdf_files = []

    # Look for PDFs in Zotero storage subdirectories
    for item_dir in documents_path.iterdir():
        if not item_dir.is_dir():
            continue

        # Find PDF in this directory
        for pdf_file in item_dir.glob("*.pdf"):
            pdf_files.append(pdf_file)
            if len(pdf_files) >= limit:
                break

        if len(pdf_files) >= limit:
            break

    return pdf_files


def main():
    """Main test function"""
    import os
    from dotenv import load_dotenv

    # Load environment
    load_dotenv()

    # Get Zotero path from environment
    documents_path = os.getenv("DOCUMENTS_PATH")
    if not documents_path:
        logger.error("DOCUMENTS_PATH not set in .env file")
        return

    documents_path = Path(documents_path)
    if not documents_path.exists():
        logger.error(f"Documents path not found: {documents_path}")
        return

    # Find sample PDFs
    logger.info(f"🔍 Searching for PDFs in {documents_path}")
    pdf_files = find_sample_pdfs(documents_path, limit=5)

    if not pdf_files:
        logger.error("No PDF files found in Zotero library")
        return

    logger.info(f"✓ Found {len(pdf_files)} PDFs to test\n")

    # Create output directory for text samples
    output_dir = Path(__file__).parent / "test_outputs" / "marker_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compare each PDF
    results = []
    for pdf_path in pdf_files:
        try:
            comparison = compare_extractions(pdf_path, output_dir=output_dir)
            results.append(comparison)
        except Exception as e:
            logger.error(f"Error comparing {pdf_path.name}: {e}")
            continue

    # Save summary
    summary_file = output_dir / "comparison_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n✅ Comparison complete! Summary saved to: {summary_file}")

    # Print overall statistics
    print("\n" + "="*80)
    print("📊 OVERALL STATISTICS")
    print("="*80)

    total_pymupdf_time = sum(r['pymupdf']['processing_time'] for r in results)
    total_marker_time = sum(r['marker']['processing_time'] for r in results)

    print(f"\nTotal processing time ({len(results)} PDFs):")
    print(f"   PyMuPDF: {total_pymupdf_time:.2f}s")
    print(f"   Marker:  {total_marker_time:.2f}s")

    pymupdf_successes = sum(1 for r in results if r['pymupdf']['success'])
    marker_successes = sum(1 for r in results if r['marker']['success'])

    print(f"\nSuccess rate:")
    print(f"   PyMuPDF: {pymupdf_successes}/{len(results)}")
    print(f"   Marker:  {marker_successes}/{len(results)}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
