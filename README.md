# Scientific Papers MCP Server

Un serveur Model Context Protocol (MCP) pour la recherche intelligente dans une collection de documents scientifiques sur la glaciologie.

## 🎯 Fonctionnalités

- **Support Multi-format** : Markdown et PDF (texte et scannés avec OCR)
- **Recherche hybride** : Combinaison de recherche sémantique (embeddings) et par mots-clés (BM25)
- **Full Text Search** : Regex, wildcards, opérateurs booléens (AND/OR) pour recherches précises
- **Extraction automatique de métadonnées** : Année, auteurs, tags, instruments, datasets
- **Chunking intelligent** : Respect de la structure des sections markdown et paragraphes PDF
- **Windows natif** : Zéro dépendance externe, tout en Python
- **Auto-indexation** : Détection automatique de nouveaux fichiers

## 🔍 Full Text Search (Nouveau!)

Le serveur supporte maintenant la recherche par motifs textuels précis en plus de la recherche sémantique.

### Opérateurs disponibles

- **$contains** : Recherche de sous-chaînes
- **$regex** : Expressions régulières
- **$and** / **$or** : Combinaisons logiques
- **$not_contains** : Exclusion de termes

### Exemples d'utilisation

**Recherche simple (via Claude) :**
```
"Trouve les articles qui mentionnent exactement 'MODIS MOD10A1'"
"Cherche 'Alaska' ET 'wildfire aerosol' dans les documents"
"Articles avec pattern 'MOD[0-9]{2}A[0-9]'"
```

**Utilisation programmatique :**
```python
from src.indexing.hybrid_search import HybridSearchEngine

# Recherche avec contains
doc_ids, scores, _, _ = search_engine.search(
    query="glacier albedo",
    where_document={"$contains": "Alaska"}
)

# Regex pour acronymes
doc_ids, scores, _, _ = search_engine.search(
    query="satellite",
    where_document={"$regex": "MODIS.*MOD10A1"}
)

# Combinaison AND
doc_ids, scores, _, _ = search_engine.search(
    query="glacier",
    where_document={
        "$and": [
            {"$contains": "albedo"},
            {"$contains": "Alaska"}
        ]
    }
)

# Combinaison OR
doc_ids, scores, _, _ = search_engine.search(
    query="satellite",
    where_document={
        "$or": [
            {"$contains": "MODIS"},
            {"$contains": "Sentinel-2"}
        ]
    }
)
```

### Outil MCP : search_fulltext()

Nouveau: Syntaxe simplifiée pour recherches FTS via Claude Code.

```
Exemples dans Claude:
- "Utilise search_fulltext pour trouver 'MODIS'"
- "search_fulltext avec pattern 'wildfire.*aerosol' en regex"
- "Cherche 'glacier' ET 'albedo' ET 'Alaska' avec search_fulltext"
```

**Paramètres:**
- `pattern`: Motif à rechercher
- `pattern_type`: 'contains', 'regex', ou 'exact'
- `combine_with`: Liste de patterns additionnels
- `combine_mode`: 'and' ou 'or'

## 📄 Formats Supportés

### Markdown (.md)
- Structure hiérarchique avec headers (`#`, `##`, `###`)
- Extraction automatique de métadonnées (titre, auteurs, année, tags)
- Chunking respectant la structure documentaire

### PDF (.pdf)
- **PDFs textuels** : Extraction de texte natif
- **PDFs scannés** : OCR automatique avec Tesseract
- Extraction de métadonnées PDF natives (titre, auteur, sujet)
- Fallback regex si métadonnées manquantes
- Chunking par paragraphes et sections

### Installation de Tesseract (pour OCR)

Pour traiter les PDFs scannés, vous devez installer Tesseract OCR:

**Windows:**
1. Télécharger depuis: https://github.com/UB-Mannheim/tesseract/wiki
2. Exécuter le fichier d'installation (par défaut: `C:\Program Files\Tesseract-OCR`)
3. Vérifier que `tesseract.exe` est accessible

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

## 🚀 Installation

### Prérequis
- Python 3.10 ou supérieur
- pip ou uv

### Setup

1. Cloner le projet et naviguer au dossier:
```bash
cd scientific-papers-mcp
```

2. Installer les dépendances:
```bash
pip install -e .
```

Ou avec uv (plus rapide):
```bash
uv pip install -e .
```

3. Vérifier la configuration dans `.env`:
```bash
# Les chemins doivent pointer vers vos répertoires
DOCUMENTS_PATH=D:\Github\Revue-de-litterature---Maitrise\Articles
CHROMA_PATH=D:\Claude Code\scientific-papers-mcp\data\chroma
```

## 📚 Utilisation

### Avec Claude Code

1. Ajouter le serveur MCP à Claude Code:
```json
{
  "mcpServers": {
    "scientific-papers": {
      "command": "python",
      "args": ["-m", "src.server"]
    }
  }
}
```

2. Utiliser dans Claude Code:
```
Cherche les articles sur "variabilité de l'albédo"
```

### Avec des scripts Python

```python
from src.indexing.chroma_client import initialize_chroma
from src.indexing.hybrid_search import HybridSearchEngine

# Initialize
chroma_collection = initialize_chroma("path/to/chroma")
search_engine = HybridSearchEngine(chroma_collection)

# Search
results, scores = search_engine.hybrid_search(
    "glacier albedo feedback",
    top_k=5,
    alpha=0.5  # 50% semantic, 50% keyword
)
```

## 🏗️ Architecture

```
src/
├── server.py              # Point d'entrée FastMCP
├── config.py              # Configuration
│
├── extractors/
│   ├── metadata_extractor.py    # Extraction métadonnées (regex)
│   └── patterns.py              # Regex patterns
│
├── indexing/
│   ├── chroma_client.py         # Initialisation Chroma
│   ├── chunker.py               # Chunking hiérarchique
│   └── hybrid_search.py         # Recherche hybride dense+sparse
│
├── tools/
│   ├── search_tools.py          # MCP tools pour recherche
│   └── metadata_tools.py        # MCP tools pour métadonnées
│
└── utils/
    ├── logger.py                # Logging
    └── file_watcher.py          # Auto-indexation
```

## 📋 Phases d'implémentation

- [x] Phase 1: Structure & dépendances
- [ ] Phase 2: Extraction métadonnées
- [ ] Phase 3: Chroma DB & chunking
- [ ] Phase 4: Recherche hybride
- [ ] Phase 5: MCP tools
- [ ] Phase 6: Intégration Claude Code
- [ ] Phase 7: Optimisations

## 📊 Performance estimée

Pour 50-200 documents:
- Indexation : ~100 ms par document
- Recherche hybride : ~50 ms
- Avec reranking : ~250 ms
- Latence totale acceptable : <500ms

## 🔍 Exemples de requêtes

### Recherche sémantique
```
"Quels articles parlent de la variabilité de l'albédo?"
"Impact des feux de forêt sur les glaciers"
```

### Recherche de stats
```
"Troupe les valeurs d'albédo entre 0.7 et 0.9"
"Articles avec MODIS 2020-2023"
```

### Filtrage
```
"Articles de Ren et al."
"Études utilisant Sentinel-2"
```

## 📖 Exemples d'Utilisation avec PDFs

### Indexation de répertoire mixte
```bash
# Répertoire contenant Markdown et PDFs
papers/
├── paper1.md
├── paper2.pdf
└── research_2023.pdf

# Indexation automatique
python -c "from src.server import index_all_documents; index_all_documents('papers/')"
```

### Utilisation en Python
```python
from src.extractors.pdf_extractor import extract_text_from_pdf, extract_metadata_from_pdf
from pathlib import Path

# Extraire texte d'un PDF
text, is_scanned = extract_text_from_pdf(Path("paper.pdf"))

# Extraire métadonnées
metadata = extract_metadata_from_pdf(Path("paper.pdf"))
print(f"Titre: {metadata['title']}")
print(f"Auteurs: {metadata['authors']}")
print(f"Année: {metadata['year']}")
```

### Recherche sur PDFs et Markdown
```python
# Recherche hybride (retourne résultats de tous les formats)
results, scores = search_engine.hybrid_search(
    "glacier albedo",
    top_k=5,
    alpha=0.5
)

# Filtrer par type de document
pdf_results = [r for r in results if r.get('file_type') == 'pdf']
markdown_results = [r for r in results if r.get('file_type') == 'markdown']
```

## ⚙️ Configuration

Voir `.env` pour les paramètres :
- `DOCUMENTS_PATH` : Chemin vers vos documents (markdown et/ou PDF)
- `CHROMA_PATH` : Chemin pour la base de données vectorielle
- `DEFAULT_ALPHA` : Balance recherche sémantique (1.0) vs keyword (0.0)
- `MAX_CHUNK_SIZE` : Taille maximale des chunks en tokens
- `EMBEDDING_MODEL` : Modèle d'embeddings (défaut: intfloat/multilingual-e5-large)

### Notes sur les PDFs
- **Indexation mixte** : Placez Markdown et PDFs dans le même répertoire
- **Détection automatique** : Le serveur détecte automatiquement le format
- **Métadonnées** : Les PDFs extraient les métadonnées natives si disponibles
- **Marquage de type** : Chaque chunk a un champ `file_type` (pdf ou markdown) pour filtrer les résultats

## 🤝 Support

Pour les questions ou problèmes, consultez la documentation MCP builder.

## 📝 Licence

MIT
