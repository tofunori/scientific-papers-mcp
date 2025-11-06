# Quick Start Guide

## Installation (déjà faite ✅)

Les dépendances sont déjà installées! Le serveur est prêt à être utilisé.

## Indexer vos documents

Avant la première utilisation, indexez vos documents markdown:

```bash
cd D:\Claude Code\scientific-papers-mcp

# Indexer tous les documents
python -c "
from src.server import index_all_documents, initialize_server
from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize before indexing
from src.indexing.chroma_client import initialize_chroma
from src.indexing.hybrid_search import HybridSearchEngine
from src.indexing.chunker import ScientificPaperChunker

collection = initialize_chroma(config.chroma_path, config.embedding_model)
search_engine = HybridSearchEngine(collection, config.embedding_model)
chunker = ScientificPaperChunker()

# Now index
result = index_all_documents(config.documents_path)
print(f'Indexed {result[\"indexed_files\"]} files')
"
```

## Configuration avec Claude Code

### Option 1: Configuration manuelle (Recommandée)

1. Ouvrez ou créez le fichier:
   ```
   C:\Users\[YourUsername]\AppData\Local\Claude Code\.claude\claude.json
   ```

2. Ajoutez cette configuration:
   ```json
   {
     "mcpServers": {
       "scientific-papers": {
         "command": "python",
         "args": ["-m", "src.server"],
         "cwd": "D:\\Claude Code\\scientific-papers-mcp"
       }
     }
   }
   ```

3. Redémarrez Claude Code

### Option 2: Vérifier la connexion

Dans Claude Code, tapez:
```
/mcp
```

Tu devrais voir:
```
scientific-papers: connected
```

## Utilisation

Une fois configuré, tu peux utiliser le MCP directement dans Claude Code!

### Exemples de requêtes

**Recherche générale (sémantique)**
```
Cherche les articles sur "variabilité de l'albédo"
```

**Recherche de stats (keyword)**
```
Trouve les valeurs d'albédo entre 0.6 et 0.8 avec MODIS
```

**Filtrer par auteur**
```
Montre-moi tous les articles de Ren et al
```

**Par année**
```
Quels articles ont été publiés en 2021?
```

**Lister les documents**
```
Liste tous les documents indexés
```

**Statistiques**
```
Donne-moi les statistiques sur ma collection
```

## Architecture

L'infrastructure est basée sur:

- **Chroma DB** : Base de données vectorielle locale
- **Sentence Transformers (multilingual-e5-large)** : Embeddings sémantiques
- **BM25** : Recherche par mots-clés
- **FastMCP** : Serveur Model Context Protocol
- **LangChain** : Chunking intelligent

## Configuration avancée

Tu peux ajuster les paramètres dans `.env`:

```env
# Balance recherche sémantique (1.0) vs keyword (0.0)
DEFAULT_ALPHA=0.5          # 0.5 = balanced

# Nombre de résultats
DEFAULT_TOP_K=10

# Taille des chunks
MAX_CHUNK_SIZE=1000        # tokens
CHUNK_OVERLAP=50
```

## Troubleshooting

### Erreur: "NoneType object"

Solution: Assurez-vous que le serveur a été initialisé avant l'indexation.

### Erreur: "Collection not found"

Solution: Vérifiez que `CHROMA_PATH` dans `.env` est correct et accessible.

### Pas de résultats

1. Vérifiez que les documents ont été indexés
2. Augmentez le nombre de résultats: `top_k=20`
3. Essayez une requête plus simple

## Documentation complète

- `README.md` - Documentation générale
- `SETUP_CLAUDE_CODE.md` - Configuration détaillée
- `pyproject.toml` - Dépendances et configuration

## Support

Vérifiez les logs:
```
logs/scientific-papers-mcp.log
```

Bonne recherche! 🚀
