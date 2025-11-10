#!/usr/bin/env python3
"""
Affiche un exemple d'extraction PyMuPDF depuis la base ChromaDB
"""

import chromadb
from pathlib import Path
import json

def show_extraction_example():
    """Affiche un exemple d'extraction d'un document indexé"""

    print("=" * 80)
    print("📚 EXEMPLE D'EXTRACTION PDF - Pipeline PyMuPDF Actuel")
    print("=" * 80)

    # Charger la collection ChromaDB
    chroma_path = Path("data/chroma")

    if not chroma_path.exists():
        print("\n❌ Aucune base ChromaDB trouvée.")
        print("   Exécutez d'abord: python index_zotero_library.py")
        return

    print(f"\n📂 Chargement de la collection: {chroma_path}")

    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(name="scientific_papers")

        # Obtenir le nombre total de documents
        count = collection.count()
        print(f"✓ Collection chargée: {count} documents indexés\n")

        # Prendre un exemple de document (premier)
        results = collection.get(
            limit=1,
            include=['metadatas', 'documents']
        )

        if not results['ids']:
            print("❌ Aucun document trouvé dans la collection")
            return

        # Extraire les données du premier document
        doc_id = results['ids'][0]
        metadata = results['metadatas'][0]
        document = results['documents'][0]

        # ========================================
        # AFFICHAGE DES MÉTADONNÉES EXTRAITES
        # ========================================
        print("─" * 80)
        print("📋 MÉTADONNÉES EXTRAITES PAR PyMuPDF")
        print("─" * 80)

        print(f"\n🆔 ID Document:     {doc_id}")
        print(f"📄 Titre:           {metadata.get('title', 'N/A')}")
        print(f"👥 Auteurs:         {metadata.get('authors', 'N/A')}")
        print(f"📅 Année:           {metadata.get('year', 'N/A')}")
        print(f"📰 Journal:         {metadata.get('journal', 'N/A')}")
        print(f"🔗 DOI:             {metadata.get('doi', 'N/A')}")
        print(f"📑 Type:            {metadata.get('type', 'N/A')}")
        print(f"📂 Source:          {metadata.get('source', 'N/A')}")

        if metadata.get('keywords'):
            keywords = metadata['keywords']
            if isinstance(keywords, str):
                keywords = keywords.split(',')[:5]
            print(f"🏷️  Keywords:        {', '.join(keywords)}")

        if metadata.get('collections'):
            print(f"📚 Collections:     {metadata['collections']}")

        # ========================================
        # CONTENU TEXTUEL EXTRAIT
        # ========================================
        print("\n\n" + "─" * 80)
        print("📝 CONTENU TEXTUEL EXTRAIT")
        print("─" * 80)

        if document:
            # Statistiques
            num_chars = len(document)
            num_words = len(document.split())
            num_lines = len(document.splitlines())

            print(f"\n✓ Caractères:       {num_chars:,}")
            print(f"✓ Mots estimés:     {num_words:,}")
            print(f"✓ Lignes:           {num_lines:,}")

            # Extrait du texte (premiers 800 caractères)
            print(f"\n📖 Extrait du document (800 premiers caractères):")
            print("─" * 80)

            # Nettoyage du texte pour l'affichage
            preview = document[:800].strip()

            # Ajouter des retours à la ligne pour meilleure lisibilité
            lines = preview.split('\n')
            for line in lines[:25]:  # Max 25 lignes
                if line.strip():
                    print(line[:80])  # Max 80 caractères par ligne

            if num_chars > 800:
                print("\n[... texte tronqué ...]")

            print("─" * 80)

        else:
            print("\n⚠️  Pas de contenu textuel trouvé")

        # ========================================
        # STRUCTURE DU CHUNK DANS CHROMADB
        # ========================================
        print("\n\n" + "─" * 80)
        print("🔍 STRUCTURE DU CHUNK DANS CHROMADB")
        print("─" * 80)

        print("\nCe document a été:")
        print("  1. ✓ Extrait avec PyMuPDF (fitz)")
        print("  2. ✓ Chunked avec LangChain RecursiveCharacterTextSplitter")
        print("  3. ✓ Embedded avec Voyage AI (voyage-context-3)")
        print("  4. ✓ Stocké dans ChromaDB avec métadonnées")

        print("\nLe chunk contient:")
        print(f"  • Text:       Le contenu du document")
        print(f"  • Embedding:  Vecteur 1024D (Voyage AI)")
        print(f"  • Metadata:   {len(metadata)} champs (titre, auteurs, DOI, etc.)")

        # ========================================
        # AFFICHER UN AUTRE EXEMPLE
        # ========================================
        print("\n\n" + "─" * 80)
        print("📚 AUTRE EXEMPLE DISPONIBLE")
        print("─" * 80)

        # Prendre un deuxième document avec DOI
        results2 = collection.get(
            limit=10,
            include=['metadatas']
        )

        # Trouver un document avec DOI
        doc_with_doi = None
        for i, meta in enumerate(results2['metadatas']):
            if meta.get('doi') and meta['doi'].startswith('10.'):
                doc_with_doi = meta
                doc_id_2 = results2['ids'][i]
                break

        if doc_with_doi:
            print(f"\nExemple avec DOI extrait:")
            print(f"  • Titre: {doc_with_doi.get('title', 'N/A')[:60]}...")
            print(f"  • DOI:   {doc_with_doi.get('doi', 'N/A')}")
            print(f"  • Année: {doc_with_doi.get('year', 'N/A')}")

        # ========================================
        # RÉSUMÉ
        # ========================================
        print("\n\n" + "=" * 80)
        print("📊 PIPELINE PyMuPDF - RÉSUMÉ")
        print("=" * 80)

        print(f"""
Ce que PyMuPDF extrait actuellement:

  ✓ Texte complet du PDF (natif ou OCR avec Tesseract)
  ✓ Métadonnées natives du PDF (titre, auteur, dates)
  ✓ Métadonnées par regex (DOI, abstract, keywords, journal)
  ✓ Images en base64 (pour embeddings multimodaux)

Limites identifiées:

  ⚠️  Pas de structure préservée (sections, hiérarchie)
  ⚠️  Extraction de tableaux basique (texte plat)
  ⚠️  Formules mathématiques non formatées
  ⚠️  Ordre de lecture parfois incorrect (colonnes)
  ⚠️  Métadonnées extraites par regex (moins fiable)

→ Docling pourrait améliorer: structure, tableaux, formules, et métadonnées
        """)

        print("=" * 80)

        # Suggestion
        print("\n💡 Pour voir l'extraction d'un PDF spécifique:")
        print("   python demo_pdf_extraction.py /chemin/vers/votre/fichier.pdf")

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    show_extraction_example()
