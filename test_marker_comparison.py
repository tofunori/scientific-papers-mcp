#!/usr/bin/env python3
"""
Test script to compare PDF extraction methods:
- PyMuPDF (fast, lightweight)
- Marker API (high quality, cloud)
- Marker Local (high quality, local)
- LlamaParse (GenAI-native, affordable)

Usage:
    python test_marker_comparison.py path/to/paper.pdf [--all]
    python test_marker_comparison.py --help
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Import extractors
from src.extractors.pdf_extractor import (
    extract_text_from_pdf,
    extract_metadata_from_pdf
)

console = Console()


def test_pymupdf(pdf_path: Path) -> Dict:
    """Test PyMuPDF extraction"""
    console.print("\n[bold blue]Testing PyMuPDF...[/bold blue]")

    start_time = time.time()

    try:
        # Extract metadata
        metadata = extract_metadata_from_pdf(pdf_path)

        # Extract text and images
        full_text, is_scanned, images = extract_text_from_pdf(pdf_path, extract_images=True)

        elapsed = time.time() - start_time

        return {
            "method": "PyMuPDF",
            "success": True,
            "elapsed_time": elapsed,
            "text_length": len(full_text),
            "num_images": len(images),
            "is_scanned": is_scanned,
            "metadata": metadata,
            "text_preview": full_text[:500],
            "error": None
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "method": "PyMuPDF",
            "success": False,
            "elapsed_time": elapsed,
            "error": str(e)
        }


def test_marker_api(pdf_path: Path, use_llm: bool = True) -> Dict:
    """Test Marker API extraction"""
    console.print("\n[bold green]Testing Marker API...[/bold green]")

    # Check if API key is configured
    from src.config import config

    if not config.marker_api_key:
        return {
            "method": "Marker API",
            "success": False,
            "error": "No API key configured. Set MARKER_API_KEY in .env"
        }

    start_time = time.time()

    try:
        from src.extractors.marker_api_extractor import MarkerAPIExtractor

        extractor = MarkerAPIExtractor(api_key=config.marker_api_key)

        markdown_text, metadata, images = extractor.extract_text_from_pdf(
            pdf_path,
            use_llm=use_llm,
            extract_images=True
        )

        elapsed = time.time() - start_time

        return {
            "method": "Marker API",
            "success": True,
            "elapsed_time": elapsed,
            "text_length": len(markdown_text),
            "num_images": len(images),
            "is_scanned": metadata.get("is_scanned", False),
            "metadata": metadata,
            "text_preview": markdown_text[:500],
            "is_markdown": True,
            "error": None
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "method": "Marker API",
            "success": False,
            "elapsed_time": elapsed,
            "error": str(e)
        }


def test_marker_local(pdf_path: Path) -> Dict:
    """Test Marker Local extraction"""
    console.print("\n[bold yellow]Testing Marker Local...[/bold yellow]")

    start_time = time.time()

    try:
        from src.extractors.marker_local_extractor import MarkerLocalExtractor

        if not MarkerLocalExtractor.is_available():
            return {
                "method": "Marker Local",
                "success": False,
                "error": "marker-pdf not installed. Install with: pip install marker-pdf"
            }

        extractor = MarkerLocalExtractor()

        markdown_text, metadata, images = extractor.extract_text_from_pdf(
            pdf_path,
            extract_images=True
        )

        elapsed = time.time() - start_time

        return {
            "method": "Marker Local",
            "success": True,
            "elapsed_time": elapsed,
            "text_length": len(markdown_text),
            "num_images": len(images),
            "is_scanned": metadata.get("is_scanned", False),
            "metadata": metadata,
            "text_preview": markdown_text[:500],
            "is_markdown": True,
            "error": None
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "method": "Marker Local",
            "success": False,
            "elapsed_time": elapsed,
            "error": str(e)
        }


def test_llamaparse(pdf_path: Path) -> Dict:
    """Test LlamaParse extraction"""
    console.print("\n[bold magenta]Testing LlamaParse...[/bold magenta]")

    # Check if API key is configured
    from src.config import config

    if not config.llamaparse_api_key:
        return {
            "method": "LlamaParse",
            "success": False,
            "error": "No API key configured. Set LLAMA_CLOUD_API_KEY in .env"
        }

    start_time = time.time()

    try:
        from src.extractors.llamaparse_extractor import LlamaParseExtractor

        if not LlamaParseExtractor.is_available():
            return {
                "method": "LlamaParse",
                "success": False,
                "error": "llama-parse not installed. Install with: pip install llama-parse"
            }

        extractor = LlamaParseExtractor(
            api_key=config.llamaparse_api_key,
            result_type=config.llamaparse_result_type,
        )

        markdown_text, metadata, images = extractor.extract_text_from_pdf(
            pdf_path,
            extract_images=True
        )

        elapsed = time.time() - start_time

        # Extract enhanced metadata
        enhanced_metadata = extractor.extract_metadata_from_markdown(markdown_text, metadata)

        return {
            "method": "LlamaParse",
            "success": True,
            "elapsed_time": elapsed,
            "text_length": len(markdown_text),
            "num_images": len(images),
            "is_scanned": metadata.get("is_scanned", False),
            "metadata": enhanced_metadata,
            "text_preview": markdown_text[:500],
            "is_markdown": True,
            "error": None
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "method": "LlamaParse",
            "success": False,
            "elapsed_time": elapsed,
            "error": str(e)
        }


def display_results(results: List[Dict]):
    """Display comparison results in a nice table"""

    # Create comparison table
    table = Table(title="PDF Extraction Comparison", show_header=True, header_style="bold magenta")

    table.add_column("Method", style="cyan", width=15)
    table.add_column("Status", width=10)
    table.add_column("Time (s)", justify="right", width=10)
    table.add_column("Text Chars", justify="right", width=12)
    table.add_column("Images", justify="right", width=8)
    table.add_column("Markdown", width=10)

    for result in results:
        if result["success"]:
            table.add_row(
                result["method"],
                "[green]✓ Success[/green]",
                f"{result['elapsed_time']:.2f}",
                f"{result['text_length']:,}",
                str(result.get("num_images", 0)),
                "✓" if result.get("is_markdown", False) else "✗"
            )
        else:
            table.add_row(
                result["method"],
                "[red]✗ Failed[/red]",
                f"{result.get('elapsed_time', 0):.2f}",
                "-",
                "-",
                "-"
            )

    console.print("\n")
    console.print(table)

    # Display errors if any
    for result in results:
        if not result["success"]:
            console.print(f"\n[red]Error in {result['method']}:[/red] {result['error']}")

    # Display text previews for successful extractions
    console.print("\n[bold]Text Previews:[/bold]")
    for result in results:
        if result["success"]:
            console.print(Panel(
                result["text_preview"],
                title=f"{result['method']} - First 500 chars",
                border_style="blue"
            ))

    # Display metadata comparison
    console.print("\n[bold]Metadata Comparison:[/bold]")
    metadata_table = Table(show_header=True, header_style="bold cyan")
    metadata_table.add_column("Field", style="yellow")

    for result in results:
        if result["success"]:
            metadata_table.add_column(result["method"], style="white")

    # Compare key metadata fields
    fields = ["title", "authors", "year", "doi", "page_count"]

    for field in fields:
        row = [field.title()]
        for result in results:
            if result["success"]:
                value = result.get("metadata", {}).get(field, "N/A")
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value[:2])  # Show first 2 items
                row.append(str(value)[:30])  # Truncate long values
        metadata_table.add_row(*row)

    console.print(metadata_table)

    # Speed comparison
    successful_results = [r for r in results if r["success"]]
    if len(successful_results) > 1:
        fastest = min(successful_results, key=lambda x: x["elapsed_time"])
        console.print(f"\n[bold green]Fastest:[/bold green] {fastest['method']} ({fastest['elapsed_time']:.2f}s)")

        # Calculate speedup factors
        console.print("\n[bold]Speedup Factors:[/bold]")
        for result in successful_results:
            if result != fastest:
                speedup = result['elapsed_time'] / fastest['elapsed_time']
                console.print(f"  {fastest['method']} is {speedup:.1f}x faster than {result['method']}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare PDF extraction methods: PyMuPDF vs Marker API vs Marker Local vs LlamaParse"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        nargs="?",
        help="Path to PDF file to test"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all methods (default: PyMuPDF + Marker API + LlamaParse)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM for Marker API (faster but lower quality)"
    )

    args = parser.parse_args()

    # If no PDF provided, look for one in Zotero library
    if not args.pdf_path:
        from src.config import config
        pdf_files = list(config.documents_path.rglob("*.pdf"))
        if not pdf_files:
            console.print("[red]Error:[/red] No PDF files found in Zotero library")
            console.print(f"Library path: {config.documents_path}")
            sys.exit(1)

        pdf_path = pdf_files[0]
        console.print(f"[yellow]No PDF specified, using:[/yellow] {pdf_path.name}")
    else:
        pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        console.print(f"[red]Error:[/red] PDF file not found: {pdf_path}")
        sys.exit(1)

    console.print(Panel(
        f"[bold]Testing PDF Extraction Methods[/bold]\n\n"
        f"File: {pdf_path.name}\n"
        f"Size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB",
        title="PDF Extraction Comparison",
        border_style="green"
    ))

    results = []

    # Test PyMuPDF (always)
    results.append(test_pymupdf(pdf_path))

    # Test Marker API (default, unless --all)
    if not args.all:
        results.append(test_marker_api(pdf_path, use_llm=not args.no_llm))

    # Test LlamaParse (default, unless --all)
    if not args.all:
        results.append(test_llamaparse(pdf_path))

    # Test all methods with --all flag
    if args.all:
        results.append(test_marker_api(pdf_path, use_llm=not args.no_llm))
        results.append(test_marker_local(pdf_path))
        results.append(test_llamaparse(pdf_path))

    # Display results
    display_results(results)

    console.print("\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
