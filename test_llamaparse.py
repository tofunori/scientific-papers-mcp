#!/usr/bin/env python3
"""
Test script for LlamaParse PDF extraction

Usage:
    python test_llamaparse.py path/to/paper.pdf
    python test_llamaparse.py --help
"""

import sys
import time
from pathlib import Path
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from src.extractors.llamaparse_extractor import LlamaParseExtractor
from src.config import config

console = Console()


def test_llamaparse(pdf_path: Path, custom_instruction: str = None) -> dict:
    """Test LlamaParse extraction on a PDF"""

    console.print("\n[bold cyan]Testing LlamaParse PDF Extraction[/bold cyan]")
    console.print(f"[yellow]File:[/yellow] {pdf_path.name}")
    console.print(f"[yellow]Size:[/yellow] {pdf_path.stat().st_size / 1024 / 1024:.2f} MB\n")

    if not config.llamaparse_api_key:
        console.print("[red]Error:[/red] No LlamaParse API key configured")
        console.print("Set LLAMA_CLOUD_API_KEY in .env file")
        console.print("Get your free API key from: https://cloud.llamaindex.ai/")
        return {"success": False, "error": "No API key"}

    try:
        # Initialize extractor
        extractor = LlamaParseExtractor(
            api_key=config.llamaparse_api_key,
            result_type=config.llamaparse_result_type,
            parsing_instruction=custom_instruction,
        )

        console.print("[green]✓[/green] LlamaParse extractor initialized")

        # Extract PDF
        start_time = time.time()

        with console.status("[bold green]Extracting PDF with LlamaParse..."):
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
        stats_table.add_row("Images Extracted", str(len(images)))
        stats_table.add_row("Is Scanned", str(metadata.get("is_scanned", False)))

        console.print(stats_table)

        # Metadata table
        console.print("\n[bold]Extracted Metadata:[/bold]")
        meta_table = Table(show_header=True, header_style="bold cyan")
        meta_table.add_column("Field", style="yellow", width=20)
        meta_table.add_column("Value", style="white", width=60)

        fields = ["title", "authors", "year", "doi", "journal", "abstract"]
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

        return {
            "success": True,
            "elapsed_time": elapsed,
            "text_length": len(markdown_text),
            "page_count": metadata.get("page_count", 0),
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
        description="Test LlamaParse PDF extraction"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        nargs="?",
        help="Path to PDF file to test"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        help="Custom parsing instruction (natural language)"
    )

    args = parser.parse_args()

    # If no PDF provided, look for one in Zotero library
    if not args.pdf_path:
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
    result = test_llamaparse(pdf_path, custom_instruction=args.instruction)

    if result["success"]:
        console.print("\n[bold green]Test completed successfully! ✓[/bold green]")
    else:
        console.print("\n[bold red]Test failed ✗[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
