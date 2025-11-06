# Integration avec Claude Code

## Configuration pour Claude Code

Pour utiliser le MCP server avec Claude Code, suivez ces étapes:

### 1. Configuration initiale

Le serveur est prêt à être utilisé! Avant de le configurer, assurez-vous d'indexer vos documents:

```bash
cd D:\Claude Code\scientific-papers-mcp

# Indexer tous les documents markdown
python -c "
from src.server import index_all_documents
from src.config import config

result = index_all_documents(config.documents_path)
print(result)
"
```

### 2. Ajouter le serveur MCP à Claude Code

#### Option A: Éditer manuellement ~/.claude.json (Windows)

Le fichier se trouve généralement à:
```
C:\Users\[YourUsername]\AppData\Local\Claude Code\.claude\claude.json
```

Ajoutez cette configuration dans la section `mcpServers`:

```json
{
  "mcpServers": {
    "scientific-papers": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "D:\\Claude Code\\scientific-papers-mcp",
      "env": {
        "PYTHONPATH": "D:\\Claude Code\\scientific-papers-mcp"
      }
    }
  }
}
```

#### Option B: Utiliser la ligne de commande Claude Code

```bash
claude mcp add scientific-papers -- python -m src.server --cwd D:\Claude Code\scientific-papers-mcp
```

### 3. Redémarrer Claude Code

Fermez et rouvrez Claude Code pour que le serveur soit détecté.

### 4. Vérifier la connexion

Dans Claude Code, tu peux vérifier si le serveur est connecté:

```
/mcp
```

Tu devrais voir:
```
- scientific-papers: connected
```

## Utilisation

Une fois intégré, tu peux utiliser les outils directement dans Claude Code:

### Recherche hybride
```
Cherche les articles sur "variabilité de l'albédo"
```

### Recherche par auteur
```
Trouve tous les articles de Ren et al.
```

### Recherche par année
```
Quels articles ont été publiés en 2020?
```

### Statistiques
```
Donne-moi les statistiques sur la collection de documents
```

### Lister les documents
```
Montre-moi tous les documents indexés
```

## Troubleshooting

### Le serveur ne démarre pas

Vérifiez:
1. Python 3.10+ est installé: `python --version`
2. Les dépendances sont installées: `pip install -e .`
3. Le chemin vers `documents_path` existe dans `.env`
4. Le chemin vers `chroma_path` est accessible en écriture

### Les documents ne sont pas trouvés

1. Vérifiez que `DOCUMENTS_PATH` dans `.env` pointe vers le bon dossier
2. Les fichiers doivent avoir l'extension `.md`
3. Indexez manuellement: `python -m src.server index_documents`

### Recherche ne retourne rien

1. Vérifiez que les documents ont été indexés
2. Essayez une requête plus générale
3. Vérifiez les logs: `logs/scientific-papers-mcp.log`

## Configuration avancée

Tu peux personnaliser le serveur en éditant `.env`:

```env
# Balance recherche sémantique (1.0) vs keyword (0.0)
DEFAULT_ALPHA=0.5

# Nombre de résultats par défaut
DEFAULT_TOP_K=10

# Taille max des chunks
MAX_CHUNK_SIZE=1000
```

## Prochaines étapes

- Indexer tous les documents: `index_documents`
- Essayer des requêtes de recherche variées
- Ajuster `DEFAULT_ALPHA` selon tes besoins de précision
