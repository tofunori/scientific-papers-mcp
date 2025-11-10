"""
Simple test script for Marker PDF extraction
Can be used with any PDF file
"""

import sys
from pathlib import Path
import time
import argparse

def test_marker_basic(pdf_path: Path):
    """
    Simple test of Marker on a single PDF

    Args:
        pdf_path: Path to PDF file to test
    """
    print("\n" + "="*80)
    print(f"Testing Marker on: {pdf_path.name}")
    print("="*80 + "\n")

    if not pdf_path.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        return

    print(f"📄 PDF size: {pdf_path.stat().st_size / 1024:.1f} KB")

    try:
        # Import Marker
        print("\n🔄 Loading Marker...")
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        print("✓ Marker loaded successfully\n")

        # Create models
        print("🔄 Creating Marker models (this may take a moment on first run)...")
        start_models = time.time()
        models = create_model_dict()
        models_time = time.time() - start_models
        print(f"✓ Models loaded in {models_time:.2f}s\n")

        # Create converter
        converter = PdfConverter(artifact_dict=models)

        # Convert PDF
        print("🔄 Converting PDF with Marker...")
        start_convert = time.time()
        rendered = converter(str(pdf_path))
        convert_time = time.time() - start_convert

        # Extract results
        markdown_text = rendered.markdown
        metadata = rendered.metadata.model_dump()

        print(f"✓ Conversion complete in {convert_time:.2f}s\n")

        # Show results
        print("="*80)
        print("📊 EXTRACTION RESULTS")
        print("="*80)

        print(f"\n📝 Extracted text length: {len(markdown_text):,} characters")
        print(f"⏱️  Processing time: {convert_time:.2f}s")

        # Show metadata
        print(f"\n📋 Metadata:")
        for key, value in metadata.items():
            if value and key not in ['toc', 'pages']:  # Skip large nested structures
                print(f"   {key}: {value}")

        # Show first 1000 chars of markdown
        print(f"\n📄 First 1000 characters of extracted Markdown:")
        print("-" * 80)
        print(markdown_text[:1000])
        if len(markdown_text) > 1000:
            print("...")
        print("-" * 80)

        # Save output
        output_dir = Path(__file__).parent / "test_outputs" / "marker_simple"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{pdf_path.stem}_marker.md"
        output_file.write_text(markdown_text)
        print(f"\n💾 Full markdown saved to: {output_file}")

        # Check for special features
        print(f"\n🔍 Special features detected:")
        has_tables = "| " in markdown_text or "|---|" in markdown_text
        has_math = "$" in markdown_text or "$$" in markdown_text
        has_code = "```" in markdown_text

        print(f"   Tables: {'✓ Yes' if has_tables else '✗ No'}")
        print(f"   Math equations: {'✓ Yes' if has_math else '✗ No'}")
        print(f"   Code blocks: {'✓ Yes' if has_code else '✗ No'}")

        print("\n" + "="*80)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")

    except ImportError as e:
        print(f"\n❌ Error: Marker not installed properly")
        print(f"   Details: {e}")
        print(f"\n   Install with: pip install marker-pdf")
        return

    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return


def main():
    parser = argparse.ArgumentParser(description="Test Marker PDF extraction")
    parser.add_argument("pdf_path", type=str, help="Path to PDF file to test")

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    test_marker_basic(pdf_path)


if __name__ == "__main__":
    main()
