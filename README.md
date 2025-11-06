# Scientific Papers MCP Server

Un serveur Model Context Protocol (MCP) pour la recherche intelligente dans une collection de documents scientifiques sur la glaciologie.

## 🎯 Fonctionnalités

- **Recherche hybride** : Combinaison de recherche sémantique (embeddings) et par mots-clés (BM25)
- **Extraction automatique de métadonnées** : Année, auteurs, tags, instruments, datasets
- **Chunking intelligent** : Respect de la structure des sections markdown
- **Windows natif** : Zéro dépendance externe, tout en Python
- **Auto-indexation** : Détection automatique de nouveaux fichiers

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

## ⚙️ Configuration

Voir `.env` pour les paramètres :
- `DOCUMENTS_PATH` : Chemin vers vos documents markdown
- `CHROMA_PATH` : Chemin pour la base de données vectorielle
- `DEFAULT_ALPHA` : Balance recherche sémantique (1.0) vs keyword (0.0)
- `MAX_CHUNK_SIZE` : Taille maximale des chunks en tokens

## 🤝 Support

Pour les questions ou problèmes, consultez la documentation MCP builder.

## 📝 Licence

MIT
