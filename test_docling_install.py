#!/usr/bin/env python3
"""Quick test to verify Docling installation"""

def test_docling_import():
    """Test if Docling can be imported"""
    print("Testing Docling installation...")

    try:
        import docling
        print(f"✓ Docling version: {docling.__version__ if hasattr(docling, '__version__') else 'unknown'}")

        from docling.document_converter import DocumentConverter
        print("✓ DocumentConverter imported successfully")

        from docling.datamodel.base_models import InputFormat
        print("✓ InputFormat imported successfully")

        print("\n✅ Docling is installed correctly!")
        return True

    except ImportError as e:
        print(f"\n❌ Docling import failed: {e}")
        return False


if __name__ == "__main__":
    success = test_docling_import()
    exit(0 if success else 1)
