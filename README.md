# Scientific Papers MCP Server

A powerful Model Context Protocol (MCP) server for intelligent semantic and full-text search across a collection of scientific papers on glaciology, climate, and environmental research.

## 🎯 What This MCP Does

The **Scientific Papers MCP** enables Claude and other AI assistants to search through scientific research papers with both semantic understanding and precise keyword matching. It acts as a bridge between your AI and your document collection, handling:

- **Intelligent Document Indexing**: Automatically processes Markdown and PDF documents (including OCR for scanned papers)
- **Hybrid Search**: Combines semantic similarity (AI understands meaning) with keyword matching (precise text search)
- **Fast Vector Database**: Uses ChromaDB with vector embeddings for AI-powered search
- **Metadata Extraction**: Automatically extracts authors, publication year, datasets, instruments, and tags
- **Smart Chunking**: Breaks documents intelligently to preserve context

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid Search** | Combines semantic embeddings + BM25 keyword search for best results |
| **Multi-Format Support** | Handles Markdown, PDFs (text), and scanned PDFs (with OCR) |
| **Metadata Extraction** | Auto-detects year, authors, datasets, instruments via regex patterns |
| **Full-Text Search** | Supports regex, wildcards, AND/OR operators for precise queries |
| **Smart Chunking** | Respects document structure (sections, paragraphs) during indexing |
| **Fast Inference** | ~50-250ms search latency depending on method |
| **Multilingual Support** | Works with 100+ languages via multilingual-e5-large embeddings |

## 📚 How It Works: Technical Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Your Documents                            │
│              (Markdown, PDF, Scanned PDFs)                       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │    Document Processing       │
        │  ├─ PDF/Text Extraction      │
        │  ├─ OCR for Scanned PDFs     │
        │  └─ Metadata Extraction      │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │    Intelligent Chunking      │
        │  ├─ Respect Section Structure│
        │  ├─ Preserve Context         │
        │  └─ Optimize Token Count     │
        └──────────────┬───────────────┘
                       │
        ┌──────────────┴────────────────┐
        │                               │
        ▼                               ▼
    ┌─────────────────┐         ┌──────────────────┐
    │   Embeddings    │         │   BM25 Tokens    │
    │ (multilingual-  │         │  (Keyword Index) │
    │  e5-large)      │         │                  │
    └────────┬────────┘         └────────┬─────────┘
             │                           │
             ▼                           ▼
    ┌─────────────────┐         ┌──────────────────┐
    │    ChromaDB     │◄────────┤  Vector Database │
    │  Vector Store   │         │  + BM25 Index    │
    └────────┬────────┘         └──────────────────┘
             │
    ┌────────┴──────────────────┐
    │   Hybrid Search Engine    │
    │  ├─ Semantic Search       │
    │  ├─ Keyword Search        │
    │  └─ Result Fusion (Alpha) │
    └────────┬──────────────────┘
             │
             ▼
    ┌──────────────────────┐
    │  Ranked Results      │
    │  (Top-K matches)     │
    └──────────────────────┘
```

### The Hybrid Search Pipeline

**Why Hybrid?** One search method alone isn't enough:
- **Semantic search** understands meaning but can miss specific terms
- **Keyword search** finds exact terms but doesn't understand context

Our hybrid approach combines both:

```
Query: "glacier albedo feedback mechanisms"
       │
       ├─→ [SEMANTIC SEARCH]
       │   • Convert to embeddings (vector space)
       │   • Find semantically similar documents
       │   • Returns: {"doc1": 0.95, "doc2": 0.87, "doc3": 0.72}
       │
       └─→ [KEYWORD SEARCH (BM25)]
           • Search for exact terms
           • TF-IDF ranking
           • Returns: {"doc1": 0.89, "doc2": 0.65, "doc4": 0.58}

       ↓ [FUSION - Controlled by Alpha parameter]

       ├─ Alpha = 1.0  → 100% semantic, 0% keyword
       ├─ Alpha = 0.5  → 50% semantic, 50% keyword (RECOMMENDED)
       └─ Alpha = 0.0  → 0% semantic, 100% keyword

       ↓ [FINAL RANKING]

       Result: {"doc1": 0.92, "doc2": 0.76, "doc3": 0.65, "doc4": 0.29}
```

### The Embedding Model

The MCP uses **multilingual-e5-large** (560M parameters) for text embeddings:

- Converts text → 1024-dimensional vectors
- Trained on 1 billion text pairs with contrastive learning
- Supports 100+ languages
- Fast inference (~10ms per document)
- State-of-the-art on MTEB benchmark for multilingual retrieval

**Alternative models** (if you want to upgrade):
- `Qwen/Qwen3-Embedding-8B` - Newer (v2025), even better, needs 16GB+ RAM
- `allenai/specter2` - Specialized for scientific papers, English only

## 🔧 Installation & Setup

### Prerequisites

- Python 3.10+
- pip or uv package manager
- (Optional) Tesseract OCR for scanned PDFs

### Installation

1. **Clone and enter the project**
```bash
git clone https://github.com/tofunori/scientific-papers-mcp.git
cd scientific-papers-mcp
```

2. **Install dependencies**
```bash
pip install -e .
```

Or with uv (faster):
```bash
uv pip install -e .
```

3. **Configure paths** in `config.py` or `.env`
```python
DOCUMENTS_PATH = "path/to/your/papers"  # Markdown & PDFs
CHROMA_PATH = "path/to/chroma/db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
```

4. **(Optional) Install Tesseract OCR** for scanned PDFs

**Windows**: Download from https://github.com/UB-Mannheim/tesseract-ocr

**Linux (Debian/Ubuntu)**:
```bash
sudo apt-get install tesseract-ocr
```

**macOS**:
```bash
brew install tesseract
```

## 💡 Quick Start

### Method 1: Using Claude Code

Add to your Claude Code config:

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

Then in Claude:
```
Search for articles about glacier albedo feedback
Find papers mentioning MODIS and MOD10A1
```

### Method 2: Python Script

```python
from src.indexing.chroma_client import initialize_chroma
from src.indexing.hybrid_search import HybridSearchEngine

# Initialize once
chroma_collection = initialize_chroma("./data/chroma")
search_engine = HybridSearchEngine(chroma_collection)

# Perform hybrid search
doc_ids, scores, documents, metadata = search_engine.hybrid_search(
    query="glacier albedo feedback",
    top_k=5,
    alpha=0.5  # 50% semantic, 50% keyword
)

# Process results
for doc_id, score, text, meta in zip(doc_ids, scores, documents, metadata):
    print(f"Match: {score:.2%}")
    print(f"Title: {meta.get('title', 'Unknown')}")
    print(f"Authors: {meta.get('authors', 'Unknown')}")
    print(f"Text: {text[:200]}...\n")
```

## 🏗️ Architecture & Components

### Directory Structure

```
src/
├── server.py                    # MCP server entry point (FastMCP)
├── config.py                    # Configuration manager
│
├── extractors/
│   ├── pdf_extractor.py        # PDF text & metadata extraction
│   ├── metadata_extractor.py    # Regex-based metadata parsing
│   └── patterns.py              # Regex patterns for metadata
│
├── indexing/
│   ├── chroma_client.py        # Vector DB initialization & queries
│   ├── chunker.py              # Document chunking (respects structure)
│   └── hybrid_search.py        # Semantic + keyword search fusion
│
├── tools/
│   ├── search_tools.py         # MCP tools for searching
│   └── metadata_tools.py       # MCP tools for metadata queries
│
└── utils/
    ├── logger.py               # Structured logging
    └── file_watcher.py         # Auto-indexing on file changes
```

### Data Processing Pipeline

```
INDEXING (One-time, on startup)
├─ Scan documents folder (Markdown + PDFs)
├─ Extract text from each document
├─ Extract metadata (authors, year, etc.)
├─ Chunk respecting structure
├─ Generate embeddings (sentence-transformers)
├─ Tokenize for BM25
└─ Store in ChromaDB + BM25 index

SEARCHING (Per query, real-time)
├─ User sends query
├─ Generate query embeddings
├─ SEMANTIC: Find nearest vectors in ChromaDB
├─ KEYWORD: BM25 score matching
├─ Fusion: Combine scores using alpha
├─ Rank and return top-K results
└─ Return with metadata & relevance scores
```

## 🔍 Search Features

### 1. Hybrid Search (Recommended)

Balances semantic understanding with keyword precision:

```python
results, scores = search_engine.hybrid_search(
    query="glacier albedo feedback mechanisms",
    top_k=5,
    alpha=0.5  # Adjust 0.0-1.0
)
```

**When to adjust alpha:**
- `alpha=1.0`: Very abstract queries ("climate change impacts")
- `alpha=0.5`: Balanced queries (recommended default)
- `alpha=0.0`: Very specific/technical queries ("MODIS MOD10A1")

### 2. Full-Text Search (Precise)

For exact text matching with regex support:

```python
# Simple contains
results = search_engine.search(
    query="satellite",
    where_document={"$contains": "MODIS"}
)

# Regex pattern
results = search_engine.search(
    query="sensor",
    where_document={"$regex": "MOD[0-9]{2}A[0-9]"}
)

# Boolean logic
results = search_engine.search(
    query="glacier",
    where_document={
        "$and": [
            {"$contains": "albedo"},
            {"$contains": "Alaska"}
        ]
    }
)
```

**Available operators:**
| Operator | Use Case |
|----------|----------|
| `$contains` | Substring search |
| `$regex` | Regular expressions |
| `$and` | All conditions must match |
| `$or` | Any condition can match |
| `$not_contains` | Exclude results |

## 📄 Supported Document Formats

### Markdown (.md)

Best for:
- Structured notes
- Research summaries
- Already-formatted content

Features:
- Hierarchical structure respected (headers)
- Metadata in frontmatter
- Clean chunking by sections

Example:
```markdown
# Paper Title
**Authors:** Smith et al.
**Year:** 2023

## Introduction
...

## Methods
...
```

### PDF - Text-based

For standard PDFs with extractable text:
- Native text extraction (fast)
- Metadata from PDF properties
- Automatic chunking by paragraphs

### PDF - Scanned (OCR)

For scanned documents/images:
- Optical Character Recognition (Tesseract)
- Slower (~100-500ms per page)
- Fallback metadata extraction via regex

```python
from src.extractors.pdf_extractor import extract_text_from_pdf

text, is_scanned = extract_text_from_pdf("scanned_paper.pdf")
# Returns: (text, True) if OCR was used
```

## ⚙️ Configuration Reference

### Environment Variables (config.py)

```python
# Document paths
DOCUMENTS_PATH = "D:/path/to/papers"          # Where to find files
CHROMA_PATH = "D:/path/to/chroma/db"         # Vector DB location

# Search settings
DEFAULT_TOP_K = 10                            # Results per query
DEFAULT_ALPHA = 0.5                           # Semantic vs keyword

# Chunking (important for quality)
MAX_CHUNK_SIZE = 1000                         # Tokens per chunk
CHUNK_OVERLAP = 50                            # Token overlap

# Embedding model
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Indexing
AUTO_INDEX_ON_START = False                   # Reindex on startup?
WATCH_DIRECTORY = True                        # Auto-index new files?
```

### Tuning for Your Use Case

**Many short documents (papers, abstracts):**
```python
MAX_CHUNK_SIZE = 500
CHUNK_OVERLAP = 25
DEFAULT_ALPHA = 0.6  # Favor semantics
```

**Few long documents (theses, books):**
```python
MAX_CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100
DEFAULT_ALPHA = 0.4  # Favor keywords
```

**Highly technical content (lots of acronyms):**
```python
DEFAULT_ALPHA = 0.3  # More keyword-focused
```

## 📊 Performance Characteristics

### Indexing

| Metric | Time | Notes |
|--------|------|-------|
| Per markdown file | ~50-100ms | Depends on size |
| Per PDF (text) | ~100-200ms | Text extraction |
| Per PDF (scanned) | ~1-5s per page | OCR is slow |
| Embedding generation | ~10ms per document | Depends on chunk count |

### Searching

| Method | Latency | Memory |
|--------|---------|--------|
| Semantic search | 50-100ms | ~1-2GB |
| Keyword search | 10-50ms | ~100-500MB |
| Hybrid (both) | 100-150ms | ~1-2GB |
| With reranking | +100-150ms | Same |

**For 100 papers (~500 chunks):**
- Initial indexing: ~2-5 minutes
- Search latency: <200ms
- Memory usage: 3-5GB (Chroma + embeddings)

## 🧪 Usage Examples

### Example 1: Climate Data Search

```python
# Find papers about MODIS and albedo
results, scores = search_engine.hybrid_search(
    query="MODIS satellite albedo measurements",
    top_k=10,
    alpha=0.6
)

for doc, score in zip(results, scores):
    if score > 0.7:  # High confidence
        print(f"✓ {doc['title']} ({score:.1%})")
```

### Example 2: Precise Technical Search

```python
# Find specific sensor data
results, scores = search_engine.hybrid_search(
    query="MOD10A1",
    alpha=0.2  # Mostly keyword-based
)

# Further filter by year
from_2020 = [r for r in results if int(r.get('year', 0)) >= 2020]
```

### Example 3: Multi-criteria Query

```python
# Complex query with metadata filtering
results = search_engine.search(
    query="glacier dynamics",
    where_document={
        "$and": [
            {"$contains": "Alaska"},
            {"$regex": "Landsat|Sentinel"}
        ]
    }
)
```

## 🚦 Troubleshooting

### Issue: Low search quality

**Solution:** Adjust alpha parameter
```python
# Too many irrelevant results?
alpha=0.3  # More keyword focus

# Missing semantically related papers?
alpha=0.8  # More semantic focus
```

### Issue: OCR not working

**Cause:** Tesseract not installed

**Solution:**
```bash
# Windows: Download from GitHub
# Linux:
sudo apt-get install tesseract-ocr
# macOS:
brew install tesseract
```

### Issue: Slow search performance

**Cause 1:** Too many documents (~1000+)
- Consider splitting into smaller indexes

**Cause 2:** Large chunk sizes
- Reduce `MAX_CHUNK_SIZE` in config

**Cause 3:** Embedding model too large
- Use `intfloat/multilingual-e5-base` (smaller, slightly slower)

## 📈 Next Steps / Roadmap

- [ ] Support for vector reranking (cross-encoders)
- [ ] Citation graph analysis
- [ ] Document similarity clustering
- [ ] Query expansion with synonyms
- [ ] Performance optimizations (quantization)
- [ ] Support for spreadsheets and tables
- [ ] Web interface for searching

## 🔗 Related Resources

- **MCP Protocol**: https://modelcontextprotocol.io/
- **ChromaDB**: https://docs.trychroma.com/
- **Sentence Transformers**: https://sbert.net/
- **FastMCP**: https://github.com/jloops/fastmcp

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review example scripts in `examples/`

---

**Last Updated:** November 2025
**Current Version:** 0.1.0
**Dependencies Updated:** chromadb 1.3.4, sentence-transformers 5.1.2
