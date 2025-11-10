#!/usr/bin/env python3
"""
Compare LlamaParse vs TensorLake PDF extraction

Compares two best-in-class document parsing APIs:
- LlamaParse: $3/1k pages + 7k free/week
- TensorLake: $10/1k pages, highest accuracy (TEDS 86.79)

Usage:
    python test_comparison.py path/to/paper.pdf --llama-key KEY1 --tensor-key KEY2
    python test_comparison.py --help
"""

import sys
import time
from pathlib import Path
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from src.extractors.llamaparse_extractor import LlamaParseExtractor
from src.extractors.tensorlake_extractor import TensorLakeExtractor

console = Console()


def test_llamaparse(pdf_path: Path, api_key: str) -> dict:
    """Test LlamaParse extraction"""
    console.print("\n[bold cyan]Testing LlamaParse...[/bold cyan]")

    try:
        extractor = LlamaParseExtractor(api_key=api_key, result_type="markdown")

        start_time = time.time()
        markdown_text, metadata, images = extractor.extract_text_from_pdf(pdf_path, extract_images=True)
        elapsed = time.time() - start_time

        enhanced_metadata = extractor.extract_metadata_from_markdown(markdown_text, metadata)

        return {
            "name": "LlamaParse",
            "success": True,
            "elapsed_time": elapsed,
            "text_length": len(markdown_text),
            "page_count": metadata.get("page_count", 0),
            "num_images": len(images),
            "metadata": enhanced_metadata,
            "text_preview": markdown_text[:500],
            "error": None
        }

    except Exception as e:
        return {
            "name": "LlamaParse",
            "success": False,
            "error": str(e)
        }


def test_tensorlake(pdf_path: Path, api_key: str) -> dict:
    """Test TensorLake extraction"""
    console.print("\n[bold magenta]Testing TensorLake...[/bold magenta]")

    try:
        extractor = TensorLakeExtractor(api_key=api_key)

        start_time = time.time()
        markdown_text, metadata, images = extractor.extract_text_from_pdf(pdf_path, extract_images=True)
        elapsed = time.time() - start_time

        enhanced_metadata = extractor.extract_metadata_from_markdown(markdown_text, metadata)

        return {
            "name": "TensorLake",
            "success": True,
            "elapsed_time": elapsed,
            "text_length": len(markdown_text),
            "page_count": metadata.get("page_count", 0),
            "num_chunks": metadata.get("num_chunks", 0),
            "num_images": len(images),
            "metadata": enhanced_metadata,
            "text_preview": markdown_text[:500],
            "error": None
        }

    except Exception as e:
        return {
            "name": "TensorLake",
            "success": False,
            "error": str(e)
        }


def display_comparison(results: list):
    """Display detailed comparison results"""

    console.print("\n[bold green]═══════════════════════════════════════════════[/bold green]")
    console.print("[bold green]           EXTRACTION COMPARISON RESULTS        [/bold green]")
    console.print("[bold green]═══════════════════════════════════════════════[/bold green]\n")

    # Performance comparison table
    perf_table = Table(title="⚡ Performance Comparison", show_header=True, header_style="bold magenta")
    perf_table.add_column("Metric", style="cyan", width=20)
    perf_table.add_column("LlamaParse", style="blue", width=20)
    perf_table.add_column("TensorLake", style="magenta", width=20)
    perf_table.add_column("Winner", style="green", width=15)

    llama = next((r for r in results if r["name"] == "LlamaParse"), None)
    tensor = next((r for r in results if r["name"] == "TensorLake"), None)

    if llama and tensor and llama["success"] and tensor["success"]:
        # Extraction time
        winner_time = "LlamaParse" if llama["elapsed_time"] < tensor["elapsed_time"] else "TensorLake"
        perf_table.add_row(
            "Extraction Time",
            f"{llama['elapsed_time']:.2f}s",
            f"{tensor['elapsed_time']:.2f}s",
            f"✓ {winner_time}"
        )

        # Text length
        winner_length = "LlamaParse" if llama["text_length"] > tensor["text_length"] else "TensorLake"
        perf_table.add_row(
            "Text Length",
            f"{llama['text_length']:,} chars",
            f"{tensor['text_length']:,} chars",
            f"✓ {winner_length}"
        )

        # Page count
        perf_table.add_row(
            "Pages",
            str(llama["page_count"]),
            str(tensor["page_count"]),
            "Same" if llama["page_count"] == tensor["page_count"] else "Differs"
        )

        console.print(perf_table)

        # Cost comparison
        num_pages = llama["page_count"] or tensor["page_count"] or 10

        cost_table = Table(title="💰 Cost Comparison", show_header=True, header_style="bold yellow")
        cost_table.add_column("Service", style="cyan", width=20)
        cost_table.add_column("Price/1k pages", style="white", width=20)
        cost_table.add_column(f"Cost ({num_pages} pages)", style="white", width=20)
        cost_table.add_column("Free Tier", style="green", width=30)

        llama_cost = (num_pages / 1000) * 3
        tensor_cost = (num_pages / 1000) * 10

        cost_table.add_row(
            "LlamaParse",
            "$3.00",
            f"${llama_cost:.4f}",
            "7,000 pages/week FREE ✓"
        )
        cost_table.add_row(
            "TensorLake",
            "$10.00",
            f"${tensor_cost:.4f}",
            "No free tier"
        )

        console.print("\n")
        console.print(cost_table)

        # Quality comparison
        console.print("\n[bold]📊 Quality Benchmarks (from public data):[/bold]")
        quality_table = Table(show_header=True, header_style="bold cyan")
        quality_table.add_column("Service", style="cyan", width=20)
        quality_table.add_column("TEDS Score", style="white", width=15)
        quality_table.add_column("F1 Score", style="white", width=15)
        quality_table.add_column("Quality", style="white", width=30)

        quality_table.add_row("LlamaParse", "~80-85 (est.)", "~88-90 (est.)", "Excellent")
        quality_table.add_row("TensorLake", "86.79", "91.7", "Best-in-class ✓")

        console.print(quality_table)

        # Metadata comparison
        console.print("\n[bold]📝 Metadata Comparison:[/bold]")
        meta_table = Table(show_header=True, header_style="bold green")
        meta_table.add_column("Field", style="yellow", width=15)
        meta_table.add_column("LlamaParse", style="blue", width=35)
        meta_table.add_column("TensorLake", style="magenta", width=35)

        fields = ["title", "authors", "year", "doi"]
        for field in fields:
            llama_val = llama["metadata"].get(field, "N/A")
            tensor_val = tensor["metadata"].get(field, "N/A")

            if isinstance(llama_val, list):
                llama_val = ", ".join(str(v) for v in llama_val[:2])
            if isinstance(tensor_val, list):
                tensor_val = ", ".join(str(v) for v in tensor_val[:2])

            meta_table.add_row(
                field.title(),
                str(llama_val)[:30],
                str(tensor_val)[:30]
            )

        console.print(meta_table)

        # Text preview comparison
        console.print("\n[bold]📄 Text Preview Comparison:[/bold]")

        console.print(Panel(
            llama["text_preview"],
            title="LlamaParse - First 500 chars",
            border_style="blue"
        ))

        console.print(Panel(
            tensor["text_preview"],
            title="TensorLake - First 500 chars",
            border_style="magenta"
        ))

        # Recommendation
        console.print("\n[bold yellow]🎯 Recommendation:[/bold yellow]")

        if num_pages * 4 <= 7000:  # Monthly usage under free tier
            console.print(Panel(
                "[green]Use LlamaParse[/green]\n\n"
                f"• Your {num_pages * 4} pages/month are under 7k free tier\n"
                "• FREE for your use case\n"
                "• Excellent quality\n"
                "• Natural language instructions\n\n"
                "[yellow]Consider TensorLake if:[/yellow]\n"
                "• You need absolute best table accuracy (86.79 vs ~85)\n"
                "• Budget is not a concern\n"
                "• Processing >7k pages/week",
                title="💡 Best Choice",
                border_style="green"
            ))
        else:
            console.print(Panel(
                "[yellow]Both are viable options[/yellow]\n\n"
                "LlamaParse: Cheaper ($3/1k), good quality, 7k free/week\n"
                "TensorLake: Best quality (TEDS 86.79), more expensive ($10/1k)\n\n"
                "Choose based on: Budget vs. absolute best accuracy",
                title="💡 Decision",
                border_style="yellow"
            ))

    else:
        # Display errors
        console.print("[red]One or more extractions failed:[/red]\n")
        for result in results:
            if not result["success"]:
                console.print(f"[red]✗ {result['name']}:[/red] {result['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare LlamaParse vs TensorLake PDF extraction"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        nargs="?",
        help="Path to PDF file to test"
    )
    parser.add_argument(
        "--llama-key",
        type=str,
        help="LlamaParse API key (or set LLAMA_CLOUD_API_KEY env var)"
    )
    parser.add_argument(
        "--tensor-key",
        type=str,
        help="TensorLake API key (or set TENSORLAKE_API_KEY env var)"
    )

    args = parser.parse_args()

    # Get API keys
    import os
    llama_key = args.llama_key or os.getenv("LLAMA_CLOUD_API_KEY")
    tensor_key = args.tensor_key or os.getenv("TENSORLAKE_API_KEY")

    if not llama_key or not tensor_key:
        console.print("[red]Error:[/red] Both API keys required")
        console.print("\nUsage:")
        console.print("  python test_comparison.py <pdf> --llama-key KEY1 --tensor-key KEY2")
        console.print("\nOr set environment variables:")
        console.print("  export LLAMA_CLOUD_API_KEY=your_llama_key")
        console.print("  export TENSORLAKE_API_KEY=your_tensor_key")
        console.print("\nGet keys from:")
        console.print("  LlamaParse: https://cloud.llamaindex.ai/")
        console.print("  TensorLake: https://cloud.tensorlake.ai/")
        sys.exit(1)

    # Get PDF path
    if not args.pdf_path:
        from src.config import config
        pdf_files = list(config.documents_path.rglob("*.pdf"))
        if not pdf_files:
            console.print("[red]Error:[/red] No PDF files found")
            sys.exit(1)

        pdf_path = pdf_files[0]
        console.print(f"[yellow]No PDF specified, using:[/yellow] {pdf_path.name}")
    else:
        pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        console.print(f"[red]Error:[/red] PDF file not found: {pdf_path}")
        sys.exit(1)

    # Display test info
    console.print(Panel(
        f"[bold]Comparing PDF Extraction Methods[/bold]\n\n"
        f"File: {pdf_path.name}\n"
        f"Size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB\n\n"
        f"Testing: LlamaParse vs TensorLake",
        title="📊 Extraction Comparison",
        border_style="green"
    ))

    # Run tests
    results = []
    results.append(test_llamaparse(pdf_path, llama_key))
    results.append(test_tensorlake(pdf_path, tensor_key))

    # Display comparison
    display_comparison(results)

    # Check success
    all_success = all(r["success"] for r in results)
    if all_success:
        console.print("\n[bold green]✓ Comparison completed successfully![/bold green]")
    else:
        console.print("\n[bold red]✗ One or more tests failed[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
