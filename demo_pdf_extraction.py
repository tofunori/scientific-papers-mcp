#!/usr/bin/env python3
"""
Démonstration de l'extraction PDF avec PyMuPDF
"""

import sys
from pathlib import Path
from src.extractors.pdf_extractor import (
    extract_text_from_pdf,
    extract_metadata_from_pdf,
    extract_images_from_pdf,
)

def format_size(num_bytes):
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} GB"


def demo_extraction(pdf_path: str):
    """Démonstration complète de l'extraction PDF"""

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        print(f"❌ Fichier non trouvé: {pdf_path}")
        return

    print("=" * 80)
    print(f"📄 DÉMONSTRATION D'EXTRACTION PDF - PyMuPDF")
    print("=" * 80)
    print(f"\nFichier: {pdf_file.name}")
    print(f"Taille: {format_size(pdf_file.stat().st_size)}\n")

    # ========================================
    # 1. EXTRACTION DE MÉTADONNÉES
    # ========================================
    print("\n" + "─" * 80)
    print("📋 ÉTAPE 1: EXTRACTION DE MÉTADONNÉES")
    print("─" * 80)

    try:
        metadata = extract_metadata_from_pdf(pdf_file)

        print("\n📌 Métadonnées extraites:")
        print(f"  • Titre:        {metadata.get('title', 'N/A')}")
        print(f"  • Auteurs:      {', '.join(metadata.get('authors', [])) or 'N/A'}")
        print(f"  • Année:        {metadata.get('year', 'N/A')}")
        print(f"  • Journal:      {metadata.get('journal', 'N/A')}")
        print(f"  • DOI:          {metadata.get('doi', 'N/A')}")
        print(f"  • Pages:        {metadata.get('page_count', 'N/A')}")
        print(f"  • Keywords:     {', '.join(metadata.get('keywords', [])[:5]) or 'N/A'}")

        if metadata.get('abstract'):
            abstract_preview = metadata['abstract'][:200] + "..." if len(metadata['abstract']) > 200 else metadata['abstract']
            print(f"\n📝 Abstract (preview):")
            print(f"  {abstract_preview}")

    except Exception as e:
        print(f"❌ Erreur lors de l'extraction des métadonnées: {e}")

    # ========================================
    # 2. EXTRACTION DE TEXTE
    # ========================================
    print("\n\n" + "─" * 80)
    print("📝 ÉTAPE 2: EXTRACTION DE TEXTE")
    print("─" * 80)

    try:
        text, is_scanned, images = extract_text_from_pdf(pdf_file, extract_images=True)

        print(f"\n✓ Type de PDF:     {'🖼️  Scanné (OCR nécessaire)' if is_scanned else '📄 Texte natif'}")
        print(f"✓ Texte extrait:   {len(text)} caractères")
        print(f"✓ Mots estimés:    ~{len(text.split())} mots")
        print(f"✓ Lignes:          ~{len(text.splitlines())} lignes")

        # Afficher un extrait du texte
        print(f"\n📖 Extrait du texte (500 premiers caractères):")
        print("─" * 80)
        text_preview = text[:500].strip()
        print(text_preview)
        if len(text) > 500:
            print("\n[... texte tronqué ...]")
        print("─" * 80)

    except Exception as e:
        print(f"❌ Erreur lors de l'extraction du texte: {e}")
        images = []

    # ========================================
    # 3. EXTRACTION D'IMAGES
    # ========================================
    print("\n\n" + "─" * 80)
    print("🖼️  ÉTAPE 3: EXTRACTION D'IMAGES")
    print("─" * 80)

    try:
        if images:
            print(f"\n✓ Images extraites: {len(images)}")

            for i, img in enumerate(images[:3]):  # Montrer max 3 images
                print(f"\n  Image {i+1}:")
                print(f"    • Page:      {img['page_num'] + 1}")
                print(f"    • Dimensions: {img['width']}x{img['height']} px")
                print(f"    • Format:     {img['format'].upper()}")
                print(f"    • Base64:     {len(img['image_base64'])} caractères")
                # Montrer début du data URI
                data_preview = img['image_base64'][:80] + "..."
                print(f"    • Data URI:   {data_preview}")

            if len(images) > 3:
                print(f"\n  ... et {len(images) - 3} autres images")
        else:
            print("\nℹ️  Aucune image trouvée dans le PDF")

    except Exception as e:
        print(f"❌ Erreur lors de l'extraction des images: {e}")

    # ========================================
    # 4. RÉSUMÉ DE L'EXTRACTION
    # ========================================
    print("\n\n" + "=" * 80)
    print("📊 RÉSUMÉ DE L'EXTRACTION")
    print("=" * 80)

    print(f"""
Pipeline PyMuPDF actuel:

  ✓ Métadonnées:    {'✓' if metadata.get('title') else '✗'} Titre, {'✓' if metadata.get('doi') else '✗'} DOI, {'✓' if metadata.get('authors') else '✗'} Auteurs
  ✓ Texte:          {len(text)} caractères extraits
  ✓ OCR:            {'Tesseract' if is_scanned else 'Non nécessaire (texte natif)'}
  ✓ Images:         {len(images)} images extraites (pour embeddings multimodaux)

Méthode:
  • PyMuPDF (fitz) pour l'extraction native
  • Pytesseract pour OCR si document scanné
  • Regex pour extraction de métadonnées du texte
  • Extraction d'images en base64 pour Voyage/Jina multimodal
    """)

    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n📚 Utilisation: python demo_pdf_extraction.py <chemin_vers_pdf>")
        print("\nExemple:")
        print("  python demo_pdf_extraction.py /chemin/vers/article.pdf")
        print("\nOu indiquez le chemin de votre bibliothèque Zotero:")
        print("  python demo_pdf_extraction.py ~/Zotero/storage/ABC123XY/article.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    demo_extraction(pdf_path)
