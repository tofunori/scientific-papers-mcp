#!/usr/bin/env python3
"""
Compare PyMuPDF vs Docling extraction on a PDF file
"""

import sys
import json
from pathlib import Path
from src.extractors.docling_extractor import extract_with_docling, compare_extractions
from src.extractors.pdf_extractor import extract_text_from_pdf, extract_metadata_from_pdf


def format_metadata(metadata: dict) -> str:
    """Format metadata for display"""
    lines = []
    for key, value in metadata.items():
        if value:
            if isinstance(value, list):
                if value:
                    lines.append(f"  • {key}: {', '.join(str(v) for v in value[:3])}")
                    if len(value) > 3:
                        lines.append(f"    ... and {len(value) - 3} more")
            elif isinstance(value, dict):
                lines.append(f"  • {key}: {len(value)} items")
            else:
                lines.append(f"  • {key}: {value}")
    return '\n'.join(lines) if lines else "  (no metadata)"


def main():
    if len(sys.argv) < 2:
        print("\n🔬 Compare PyMuPDF vs Docling Extraction\n")
        print("Usage: python compare_extractors.py <pdf_file> [--save-outputs]")
        print("\nOptions:")
        print("  --save-outputs    Save full extraction outputs to files")
        print("\nExample:")
        print('  python compare_extractors.py "article.pdf"')
        print('  python compare_extractors.py "article.pdf" --save-outputs')

        # Suggest a file from index
        import json
        state_file = Path("data/indexing_state.json")
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                if state.get('indexed_files'):
                    first_pdf = list(state['indexed_files'].keys())[0]
                    print(f'\n  python compare_extractors.py "{first_pdf}"')
        sys.exit(1)

    pdf_path = sys.argv[1]
    save_outputs = "--save-outputs" in sys.argv

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    print("=" * 80)
    print("🔬 PDF EXTRACTION COMPARISON: PyMuPDF vs Docling")
    print("=" * 80)
    print(f"\nFile: {pdf_file.name}")
    print(f"Size: {pdf_file.stat().st_size / 1024:.1f} KB\n")

    # ==============================================
    # PYMUPDF EXTRACTION
    # ==============================================
    print("─" * 80)
    print("📄 EXTRACTION #1: PyMuPDF (Current Pipeline)")
    print("─" * 80)

    try:
        pymupdf_text, is_scanned, images = extract_text_from_pdf(pdf_file, extract_images=False)
        pymupdf_metadata = extract_metadata_from_pdf(pdf_file)

        print(f"\n✓ Extraction completed")
        print(f"  • Method:         Native text extraction + OCR fallback")
        print(f"  • Text length:    {len(pymupdf_text):,} characters")
        print(f"  • Word count:     {len(pymupdf_text.split()):,} words")
        print(f"  • PDF type:       {'Scanned (OCR used)' if is_scanned else 'Native text'}")
        print(f"  • Pages:          {pymupdf_metadata.get('page_count', 'N/A')}")

        print(f"\n📋 Metadata extracted:")
        print(format_metadata(pymupdf_metadata))

        print(f"\n📝 Text preview (300 chars):")
        print("─" * 80)
        preview = pymupdf_text[:300].replace('\n', ' ').strip()
        print(preview)
        if len(pymupdf_text) > 300:
            print("[...]")
        print("─" * 80)

        if save_outputs:
            output_file = pdf_file.stem + "_pymupdf.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(pymupdf_text)
            print(f"\n💾 Full output saved: {output_file}")

    except Exception as e:
        print(f"\n❌ PyMuPDF extraction failed: {e}")
        import traceback
        traceback.print_exc()

    # ==============================================
    # DOCLING EXTRACTION
    # ==============================================
    print("\n\n" + "─" * 80)
    print("🚀 EXTRACTION #2: Docling (IBM Document Understanding)")
    print("─" * 80)

    try:
        docling_markdown, docling_metadata, json_structure = extract_with_docling(pdf_file)

        print(f"\n✓ Extraction completed")
        print(f"  • Method:         Vision-Language Model + Structure Analysis")
        print(f"  • Text length:    {len(docling_markdown):,} characters")
        print(f"  • Word count:     {len(docling_markdown.split()):,} words")
        print(f"  • Format:         Structured Markdown")
        print(f"  • Pages:          {docling_metadata.get('page_count', 'N/A')}")
        print(f"  • Sections:       {len(docling_metadata.get('sections', []))} detected")
        print(f"  • Tables:         {len(docling_metadata.get('tables', []))} detected")
        print(f"  • Figures:        {len(docling_metadata.get('figures', []))} detected")

        print(f"\n📋 Metadata extracted:")
        print(format_metadata(docling_metadata))

        print(f"\n📝 Markdown preview (300 chars):")
        print("─" * 80)
        preview = docling_markdown[:300].strip()
        print(preview)
        if len(docling_markdown) > 300:
            print("[...]")
        print("─" * 80)

        if save_outputs:
            # Save markdown
            md_file = pdf_file.stem + "_docling.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(docling_markdown)
            print(f"\n💾 Markdown saved: {md_file}")

            # Save JSON structure
            json_file = pdf_file.stem + "_docling_structure.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_structure, f, indent=2, ensure_ascii=False)
            print(f"💾 Structure saved: {json_file}")

    except Exception as e:
        print(f"\n❌ Docling extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==============================================
    # COMPARISON SUMMARY
    # ==============================================
    print("\n\n" + "=" * 80)
    print("📊 COMPARISON SUMMARY")
    print("=" * 80)

    pymupdf_len = len(pymupdf_text)
    docling_len = len(docling_markdown)

    pymupdf_words = len(pymupdf_text.split())
    docling_words = len(docling_markdown.split())

    print(f"\n📏 Content Length:")
    print(f"  PyMuPDF:  {pymupdf_len:,} chars | {pymupdf_words:,} words")
    print(f"  Docling:  {docling_len:,} chars | {docling_words:,} words")
    print(f"  Diff:     {docling_len - pymupdf_len:+,} chars | {docling_words - pymupdf_words:+,} words")

    print(f"\n📋 Metadata Completeness:")
    pymupdf_meta_count = sum(1 for v in pymupdf_metadata.values() if v)
    docling_meta_count = sum(1 for v in docling_metadata.values() if v)
    print(f"  PyMuPDF:  {pymupdf_meta_count}/10 fields populated")
    print(f"  Docling:  {docling_meta_count}/12 fields populated")

    print(f"\n🎯 Docling Advantages:")
    advantages = []
    if docling_metadata.get('sections'):
        advantages.append(f"  ✓ {len(docling_metadata['sections'])} sections with hierarchy detected")
    if docling_metadata.get('tables'):
        advantages.append(f"  ✓ {len(docling_metadata['tables'])} tables with structure preserved")
    if docling_metadata.get('figures'):
        advantages.append(f"  ✓ {len(docling_metadata['figures'])} figures identified with captions")
    if "##" in docling_markdown or "###" in docling_markdown:
        advantages.append("  ✓ Markdown headings for better document structure")

    if advantages:
        print('\n'.join(advantages))
    else:
        print("  (None detected for this document)")

    print(f"\n⚡ PyMuPDF Advantages:")
    print(f"  ✓ Faster extraction (no ML model inference)")
    print(f"  ✓ Lighter dependencies (~50MB vs ~1GB)")
    print(f"  ✓ Lower memory usage")

    # ==============================================
    # RECOMMENDATIONS
    # ==============================================
    print("\n\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)

    print(f"\n🎯 Use PyMuPDF when:")
    print(f"  • You need fast, lightweight extraction")
    print(f"  • Documents are simple (text-only, no complex layouts)")
    print(f"  • You don't need structure preservation")
    print(f"  • Memory/disk space is limited")

    print(f"\n🚀 Use Docling when:")
    print(f"  • You need preserved document structure (sections, headings)")
    print(f"  • Documents contain complex tables")
    print(f"  • You want better metadata extraction")
    print(f"  • Document understanding is critical for RAG quality")
    print(f"  • You're processing scientific papers with figures/equations")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
