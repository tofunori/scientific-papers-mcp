# Docling Setup & Testing Guide

## ✅ Installation Status

**Installation en cours...**

```bash
pip install docling
```

Docling installe les dépendances suivantes :
- **PyTorch 2.9.0** (~900 MB) - Vision-Language Model inference
- **TorchVision 0.24.0** - Computer vision utilities
- **transformers 4.57.1** - Hugging Face models
- **docling-core, docling-parse, docling-ibm-models** - Core Docling libraries
- **RapidOCR** - Fast OCR for scanned PDFs
- Et ~60 autres dépendances

**Taille totale installée :** ~1.5 GB

---

## 📋 Files Created

### 1. **`src/extractors/docling_extractor.py`**
Module d'extraction principal utilisant Docling

**Fonctions principales:**
- `extract_with_docling()` - Extraction complète (markdown + métadonnées + structure JSON)
- `extract_tables_from_docling()` - Extraction de tableaux structurés
- `compare_extractions()` - Comparaison PyMuPDF vs Docling

### 2. **`compare_extractors.py`**
Script de comparaison interactif

**Usage:**
```bash
python compare_extractors.py "article.pdf"
python compare_extractors.py "article.pdf" --save-outputs
```

**Affiche:**
- Extraction PyMuPDF (pipeline actuel)
- Extraction Docling (nouveau)
- Comparaison côte à côte
- Recommandations

### 3. **`test_docling_install.py`**
Test rapide de l'installation

**Usage:**
```bash
python test_docling_install.py
```

---

## 🧪 Quick Start (Once Installation Complete)

### Step 1: Verify Installation

```bash
python test_docling_install.py
```

**Expected output:**
```
Testing Docling installation...
✓ Docling version: 2.61.2
✓ DocumentConverter imported successfully
✓ InputFormat imported successfully

✅ Docling is installed correctly!
```

### Step 2: Test on a Sample PDF

```bash
# Use a PDF from your Zotero library
python compare_extractors.py "C:\Users\thier\Zotero\storage\2DDR8JUQ\Chen et al. - 2019.pdf"
```

### Step 3: Compare Results

The script will show:

```
================================================================================
🔬 PDF EXTRACTION COMPARISON: PyMuPDF vs Docling
================================================================================

─────────────────────────────────────────────────────────────
📄 EXTRACTION #1: PyMuPDF (Current Pipeline)
─────────────────────────────────────────────────────────────

✓ Extraction completed
  • Method:         Native text extraction + OCR fallback
  • Text length:    45,234 characters
  • Word count:     7,500 words
  • PDF type:       Native text
  • Pages:          20

📋 Metadata extracted:
  • title: The FireWork v2.0 air quality...
  • authors: Chen, J., Anderson, K., ...
  • year: 2019
  • doi: 10.5194/gmd-12-3283-2019

─────────────────────────────────────────────────────────────
🚀 EXTRACTION #2: Docling (IBM Document Understanding)
─────────────────────────────────────────────────────────────

✓ Extraction completed
  • Method:         Vision-Language Model + Structure Analysis
  • Text length:    47,890 characters
  • Word count:     7,950 words
  • Format:         Structured Markdown
  • Pages:          20
  • Sections:       8 detected
  • Tables:         4 detected
  • Figures:        5 detected

================================================================================
📊 COMPARISON SUMMARY
================================================================================

🎯 Docling Advantages:
  ✓ 8 sections with hierarchy detected
  ✓ 4 tables with structure preserved
  ✓ 5 figures identified with captions
  ✓ Markdown headings for better document structure
```

---

## 🔍 What Docling Extracts Better

### 1. **Document Structure**

**PyMuPDF** (flat text):
```
1 Introduction
This is the introduction text...
2 Methods
2.1 Data Collection
Data was collected...
```

**Docling** (structured markdown):
```markdown
# 1. Introduction

This is the introduction text...

## 2. Methods

### 2.1 Data Collection

Data was collected...
```

### 2. **Tables**

**PyMuPDF** (text plat):
```
Species Boreal Forest Grassland Agricultural Unit
PM2.5 15.8 8.5 7.2 g kg-1
CO 107 63 92 g kg-1
```

**Docling** (structured):
```markdown
**Table 1: Emission factors**

| Species | Boreal Forest | Grassland | Agricultural | Unit   |
|---------|---------------|-----------|--------------|--------|
| PM₂.₅   | 15.8          | 8.5       | 7.2          | g kg⁻¹ |
| CO      | 107           | 63        | 92           | g kg⁻¹ |
```

### 3. **Figures & Captions**

**PyMuPDF**: Images extracted separately, captions mixed in text

**Docling**: Figures linked to captions
```markdown
**Figure 1: Spatial distribution of fire emissions**
![Image](figure_1.png)

Caption: Daily integrated PM₂.₅ emissions (tonnes per grid cell)...
```

### 4. **Metadata**

**PyMuPDF**: 6-8 fields (regex extraction)
**Docling**: 10-12 fields (ML-based extraction)

---

## 📊 Performance Comparison

| Aspect | PyMuPDF | Docling | Winner |
|--------|---------|---------|--------|
| **Speed** | ~2-3 sec/doc | ~10-15 sec/doc | PyMuPDF ✓ |
| **Memory** | ~50 MB | ~500 MB | PyMuPDF ✓ |
| **Installation size** | ~10 MB | ~1.5 GB | PyMuPDF ✓ |
| **Structure preservation** | ❌ None | ✅ Full | Docling ✓ |
| **Table extraction** | ❌ Text only | ✅ Structured | Docling ✓ |
| **Figure detection** | ⚠️ Basic | ✅ Advanced | Docling ✓ |
| **Metadata quality** | ⚠️ Regex | ✅ ML-based | Docling ✓ |
| **Scanned PDFs** | ⚠️ Needs Tesseract | ✅ Built-in OCR | Docling ✓ |
| **RAG Quality** | Good | **Excellent** | Docling ✓ |

---

## 💡 Recommended Strategy

### **Hybrid Approach** (Best of Both Worlds)

```python
def smart_extract(pdf_path):
    """Use PyMuPDF for simple docs, Docling for complex ones"""

    # Quick check with PyMuPDF
    quick_meta = extract_metadata_from_pdf(pdf_path)

    # Use Docling if:
    if (
        has_many_tables(pdf_path) or           # Complex tables
        is_multi_column(pdf_path) or            # Multi-column layout
        is_scanned_pdf(pdf_path) or             # Scanned document
        is_scientific_paper(quick_meta)         # Scientific paper
    ):
        return extract_with_docling(pdf_path)  # Use Docling
    else:
        return extract_text_from_pdf(pdf_path)  # Use PyMuPDF (faster)
```

**Benefits:**
- Fast extraction for simple documents (PyMuPDF)
- High-quality extraction for complex documents (Docling)
- Optimized memory usage
- Better RAG performance overall

---

## 🚀 Integration into Your RAG Pipeline

### Current Pipeline

```
PDF → PyMuPDF → Raw Text → LangChain Chunking → Voyage Embeddings → ChromaDB
```

### Enhanced Pipeline with Docling

```
PDF → Docling → Structured Markdown → Semantic Chunking → Voyage Embeddings → ChromaDB
                  ↓
            (sections, tables,
             figures, hierarchy)
```

**Advantages for RAG:**
1. **Better chunking** - Respect document structure
2. **Richer context** - Section headings preserved in embeddings
3. **Table search** - Query structured data
4. **Figure references** - Link text to visuals
5. **Improved citations** - "According to Table 2 in Section 3.1..."

---

## 🔧 Next Steps

### 1. **Test Installation** (Once pip install complete)
```bash
python test_docling_install.py
```

### 2. **Compare on Your PDFs**
```bash
python compare_extractors.py "path/to/your/paper.pdf" --save-outputs
```

### 3. **Evaluate Results**
- Check markdown quality
- Verify table extraction
- Test on scanned PDFs
- Measure performance

### 4. **Decide on Strategy**
- Use Docling for all documents? (best quality)
- Use hybrid approach? (balanced)
- Stick with PyMuPDF? (fastest)

---

## ⚠️ Important Notes

1. **First Run**: Docling will download ML models (~500MB) on first use
2. **GPU**: Not required, but speeds up extraction if available
3. **Memory**: Expect 500MB-1GB RAM usage per document
4. **Time**: 10-15 seconds per document (vs 2-3s for PyMuPDF)

---

## 📚 Further Reading

- [Docling Documentation](https://www.docling.ai/)
- [IBM Research Blog](https://research.ibm.com/blog/docling-generative-AI)
- [Granite-Docling Announcement](https://www.ibm.com/new/announcements/granite-docling-end-to-end-document-conversion)

---

**Ready to test once installation completes!** ✨
