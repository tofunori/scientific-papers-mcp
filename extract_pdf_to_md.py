#!/usr/bin/env python3
"""
Extraction simple d'un PDF vers Markdown avec PyMuPDF
"""

import sys
from pathlib import Path
from src.extractors.pdf_extractor import (
    extract_text_from_pdf,
    extract_metadata_from_pdf,
)


def pdf_to_markdown(pdf_path: str) -> str:
    """
    Convertit un PDF en Markdown avec PyMuPDF

    Args:
        pdf_path: Chemin vers le fichier PDF

    Returns:
        Contenu formaté en Markdown
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        return f"❌ Fichier non trouvé: {pdf_path}"

    # Extraire métadonnées
    metadata = extract_metadata_from_pdf(pdf_file)

    # Extraire texte
    text, is_scanned, images = extract_text_from_pdf(pdf_file, extract_images=False)

    # Construire le markdown
    md_content = []

    # En-tête avec métadonnées
    md_content.append("---")
    md_content.append(f"title: {metadata.get('title', 'Sans titre')}")
    if metadata.get('authors'):
        md_content.append(f"authors: {', '.join(metadata['authors'])}")
    if metadata.get('year'):
        md_content.append(f"year: {metadata['year']}")
    if metadata.get('doi'):
        md_content.append(f"doi: {metadata['doi']}")
    if metadata.get('journal'):
        md_content.append(f"journal: {metadata['journal']}")
    md_content.append(f"source: {pdf_file.name}")
    md_content.append(f"type: {'scanned_pdf' if is_scanned else 'text_pdf'}")
    md_content.append("---")
    md_content.append("")

    # Titre principal
    title = metadata.get('title', pdf_file.stem)
    md_content.append(f"# {title}")
    md_content.append("")

    # Auteurs et métadonnées
    if metadata.get('authors'):
        md_content.append(f"**Auteurs:** {', '.join(metadata['authors'])}")
    if metadata.get('year'):
        md_content.append(f"**Année:** {metadata['year']}")
    if metadata.get('journal'):
        md_content.append(f"**Journal:** {metadata['journal']}")
    if metadata.get('doi'):
        md_content.append(f"**DOI:** [{metadata['doi']}](https://doi.org/{metadata['doi']})")
    md_content.append("")

    # Abstract si disponible
    if metadata.get('abstract'):
        md_content.append("## Abstract")
        md_content.append("")
        md_content.append(metadata['abstract'])
        md_content.append("")

    # Keywords si disponibles
    if metadata.get('keywords'):
        md_content.append(f"**Keywords:** {', '.join(metadata['keywords'])}")
        md_content.append("")

    # Contenu principal
    md_content.append("## Contenu")
    md_content.append("")

    # Nettoyage basique du texte
    # Séparer en paragraphes
    lines = text.split('\n')
    current_paragraph = []

    for line in lines:
        line = line.strip()

        # Ligne vide = nouveau paragraphe
        if not line:
            if current_paragraph:
                md_content.append(' '.join(current_paragraph))
                md_content.append("")
                current_paragraph = []
        else:
            current_paragraph.append(line)

    # Dernier paragraphe
    if current_paragraph:
        md_content.append(' '.join(current_paragraph))

    # Métadonnées d'extraction
    md_content.append("")
    md_content.append("---")
    md_content.append("")
    md_content.append("## Informations d'extraction")
    md_content.append("")
    md_content.append(f"- **Méthode:** PyMuPDF (fitz)")
    md_content.append(f"- **Type de PDF:** {'Scanné (OCR)' if is_scanned else 'Texte natif'}")
    md_content.append(f"- **Caractères extraits:** {len(text):,}")
    md_content.append(f"- **Mots estimés:** {len(text.split()):,}")
    md_content.append(f"- **Pages:** {metadata.get('page_count', 'N/A')}")

    return '\n'.join(md_content)


def main():
    if len(sys.argv) < 2:
        print("\n📄 Extraction PDF → Markdown avec PyMuPDF\n")
        print("Usage: python extract_pdf_to_md.py <chemin_pdf> [output.md]")
        print("\nExemples:")
        print("  python extract_pdf_to_md.py article.pdf")
        print("  python extract_pdf_to_md.py article.pdf output.md")

        # Suggérer un fichier depuis l'index
        import json
        state_file = Path("data/indexing_state.json")
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                if state.get('indexed_files'):
                    first_pdf = list(state['indexed_files'].keys())[0]
                    print(f"\nExemple avec un fichier indexé:")
                    print(f'  python extract_pdf_to_md.py "{first_pdf}"')

        sys.exit(1)

    pdf_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\n📄 Extraction de: {Path(pdf_path).name}")
    print("=" * 80)

    # Extraire et convertir en markdown
    markdown_content = pdf_to_markdown(pdf_path)

    # Sauvegarder ou afficher
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"\n✓ Markdown sauvegardé dans: {output_file}")
        print(f"  Taille: {len(markdown_content):,} caractères")
    else:
        # Afficher à l'écran (limité)
        print("\n" + "─" * 80)
        print("MARKDOWN EXTRAIT (Preview - 2000 premiers caractères)")
        print("─" * 80)
        print(markdown_content[:2000])
        if len(markdown_content) > 2000:
            print("\n[... contenu tronqué ...]")
            print(f"\nTotal: {len(markdown_content):,} caractères")
            print("\n💡 Utilisez un fichier de sortie pour voir le contenu complet:")
            print(f'   python extract_pdf_to_md.py "{pdf_path}" output.md')
        print("─" * 80)


if __name__ == "__main__":
    main()
