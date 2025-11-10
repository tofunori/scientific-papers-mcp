#!/usr/bin/env python3
"""
Test script for TensorLake PDF extraction

TensorLake: Best-in-class document parsing
- TEDS score: 86.79 (best)
- F1 score: 91.7
- Price: $10/1k pages

Usage:
    python test_tensorlake.py path/to/paper.pdf
    python test_tensorlake.py --help
"""

import sys
import time
from pathlib import Path
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.extractors.tensorlake_extractor import TensorLakeExtractor

console = Console()


def test_tensorlake(pdf_path: Path, api_key: str) -> dict:
    """Test TensorLake extraction on a PDF"""

    console.print("\n[bold cyan]Testing TensorLake PDF Extraction[/bold cyan]")
    console.print(f"[yellow]File:[/yellow] {pdf_path.name}")
    console.print(f"[yellow]Size:[/yellow] {pdf_path.stat().st_size / 1024 / 1024:.2f} MB\n")

    if not api_key:
        console.print("[red]Error:[/red] No TensorLake API key provided")
        console.print("Usage: python test_tensorlake.py <pdf> --api-key YOUR_KEY")
        console.print("Get your API key from: https://cloud.tensorlake.ai/")
        return {"success": False, "error": "No API key"}

    try:
        # Initialize extractor
        extractor = TensorLakeExtractor(api_key=api_key)
        console.print("[green]✓[/green] TensorLake extractor initialized")

        # Extract PDF
        start_time = time.time()

        with console.status("[bold green]Extracting PDF with TensorLake..."):
            markdown_text, metadata, images = extractor.extract_text_from_pdf(
                pdf_path,
                extract_images=True,
            )

        elapsed = time.time() - start_time

        # Extract enhanced metadata
        enhanced_metadata = extractor.extract_metadata_from_markdown(markdown_text, metadata)

        # Display results
        console.print(f"\n[green]✓ Extraction successful![/green] ({elapsed:.2f}s)\n")

        # Stats table
        stats_table = Table(title="Extraction Statistics", show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric", style="cyan", width=25)
        stats_table.add_column("Value", style="white", width=30)

        stats_table.add_row("Extraction Time", f"{elapsed:.2f}s")
        stats_table.add_row("Text Length", f"{len(markdown_text):,} characters")
        stats_table.add_row("Page Count", str(metadata.get("page_count", "N/A")))
        stats_table.add_row("Number of Chunks", str(metadata.get("num_chunks", "N/A")))
        stats_table.add_row("Images Extracted", str(len(images)))

        console.print(stats_table)

        # Metadata table
        console.print("\n[bold]Extracted Metadata:[/bold]")
        meta_table = Table(show_header=True, header_style="bold cyan")
        meta_table.add_column("Field", style="yellow", width=20)
        meta_table.add_column("Value", style="white", width=60)

        fields = ["title", "authors", "year", "doi", "abstract"]
        for field in fields:
            value = enhanced_metadata.get(field, "N/A")
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value[:3])  # Show first 3 items
            value_str = str(value)[:80]  # Truncate long values
            meta_table.add_row(field.title(), value_str)

        console.print(meta_table)

        # Text preview
        console.print("\n[bold]Markdown Preview (first 1000 chars):[/bold]")
        console.print(Panel(
            markdown_text[:1000] + ("..." if len(markdown_text) > 1000 else ""),
            title="Extracted Markdown",
            border_style="blue"
        ))

        # Cost calculation
        num_pages = metadata.get("page_count", 0)
        if num_pages:
            cost = (num_pages / 1000) * 10  # $10 per 1k pages
            console.print(f"\n[yellow]💰 Estimated cost:[/yellow] ${cost:.4f} ({num_pages} pages)")

        return {
            "success": True,
            "elapsed_time": elapsed,
            "text_length": len(markdown_text),
            "page_count": metadata.get("page_count", 0),
            "num_chunks": metadata.get("num_chunks", 0),
            "num_images": len(images),
            "metadata": enhanced_metadata,
            "text_preview": markdown_text[:500],
        }

    except Exception as e:
        console.print(f"\n[red]✗ Extraction failed:[/red] {str(e)}")
        import traceback
        console.print(f"\n[dim]{traceback.format_exc()}[/dim]")
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Test TensorLake PDF extraction"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        nargs="?",
        help="Path to PDF file to test"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="TensorLake API key (or set TENSORLAKE_API_KEY env var)"
    )

    args = parser.parse_args()

    # Get API key
    import os
    api_key = args.api_key or os.getenv("TENSORLAKE_API_KEY")

    if not api_key:
        console.print("[red]Error:[/red] TensorLake API key required")
        console.print("Usage: python test_tensorlake.py <pdf> --api-key YOUR_KEY")
        console.print("Or set: export TENSORLAKE_API_KEY=your_key")
        console.print("Get your key from: https://cloud.tensorlake.ai/")
        sys.exit(1)

    # Get PDF path
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

    # Test extraction
    result = test_tensorlake(pdf_path, api_key)

    if result["success"]:
        console.print("\n[bold green]Test completed successfully! ✓[/bold green]")
    else:
        console.print("\n[bold red]Test failed ✗[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
