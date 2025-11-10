# Marker PDF Extraction Setup Guide

This guide covers how to use Marker for high-quality PDF extraction in the Scientific Papers MCP server.

## Table of Contents

- [What is Marker?](#what-is-marker)
- [Why Use Marker?](#why-use-marker)
- [Setup Options](#setup-options)
  - [Option 1: Marker API (Recommended)](#option-1-marker-api-recommended)
  - [Option 2: Marker Local](#option-2-marker-local)
  - [Option 3: PyMuPDF (Default)](#option-3-pymupdf-default)
- [Configuration](#configuration)
- [Testing & Comparison](#testing--comparison)
- [Troubleshooting](#troubleshooting)
- [Cost Estimation](#cost-estimation)

---

## What is Marker?

[Marker](https://github.com/datalab-to/marker) by Datalab is a state-of-the-art PDF extraction tool that converts PDFs to **Markdown** with high accuracy.

**Key Features:**
- ✅ **Superior table extraction** (including multi-page tables)
- ✅ **LaTeX equation extraction** (preserves mathematical notation)
- ✅ **Document structure preservation** (headers, sections, hierarchy)
- ✅ **LLM enhancement** (optional, for even better quality)
- ✅ **OCR support** (for scanned documents)
- ✅ **10x faster than Nougat** on scientific papers

---

## Why Use Marker?

| Feature | PyMuPDF (Default) | Marker |
|---------|-------------------|--------|
| **Tables** | ⚠️ Often broken | ✅ Excellent |
| **Equations** | ❌ Plain text | ✅ LaTeX format |
| **Structure** | ❌ No hierarchy | ✅ Markdown headers |
| **Speed** | ✅ Very fast | ⚠️ Slower |
| **Cost** | ✅ Free | ⚠️ API has costs |
| **Quality** | ⚠️ Good | ✅ Excellent |

**Use Marker if:**
- Your papers have **complex tables** or **equations**
- You need **structured markdown** for better RAG
- Quality > speed for your use case

**Use PyMuPDF if:**
- You need **maximum speed**
- Your papers are mostly **plain text**
- You want **zero API costs**

---

## Setup Options

### Option 1: Marker API (Recommended)

**Best for:** Most users, especially those adding papers regularly

#### Step 1: Get API Key

1. Sign up at [datalab.to](https://www.datalab.to/)
2. Get **$5 free credits** (processes ~250-500 pages)
3. Copy your API key

#### Step 2: Configure

Edit your `.env` file:

```bash
# Enable Marker API
PDF_EXTRACTION_METHOD=marker_api

# Add your API key
MARKER_API_KEY=your_api_key_here

# Enable LLM for best quality (recommended)
# Uses Gemini 2.0 Flash server-side - no extra config needed!
MARKER_USE_LLM=true

# Optional: Force OCR for all pages
MARKER_FORCE_OCR=false

# Fallback to PyMuPDF if API fails
MARKER_FALLBACK_TO_PYMUPDF=true
```

#### Step 3: Test

```bash
python test_marker_comparison.py
```

**Advantages:**
- ✅ No heavy installation (~0 MB)
- ✅ No GPU needed
- ✅ Always up-to-date models
- ✅ Scalable (cloud processing)

**Disadvantages:**
- ❌ Requires internet connection
- ❌ Costs money after free credits
- ❌ Slower than local (network latency)

---

### Option 2: Marker Local

**Best for:** Heavy users, offline processing, or avoiding API costs

#### Step 1: Install Marker

```bash
# Basic installation (CPU)
pip install marker-pdf

# With GPU support (recommended for speed)
pip install marker-pdf
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Installation size:** ~3-5GB (includes PyTorch + ML models)

**First run:** Models will download automatically (~2-10 minutes)

#### Step 2: Configure

Edit your `.env` file:

```bash
# Enable Marker Local
PDF_EXTRACTION_METHOD=marker_local

# GPU batch multiplier (increase for better GPU usage)
MARKER_LOCAL_BATCH_MULTIPLIER=2

# Optional: Enable LLM for better quality
MARKER_LOCAL_USE_LLM=true

# Choose LLM service (see LLM Configuration section below)
MARKER_LOCAL_LLM_SERVICE=gemini  # Options: gemini, vertex, claude, openai, ollama

# Gemini API key (default LLM for Marker)
# Get from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_gemini_key_here
MARKER_GEMINI_MODEL=gemini-2.0-flash

# Fallback to PyMuPDF if Marker fails
MARKER_FALLBACK_TO_PYMUPDF=true
```

#### LLM Configuration for Marker Local

Marker Local supports multiple LLM providers. Here are your options:

**Option A: Gemini (Default, Recommended)** ⭐
```bash
MARKER_LOCAL_LLM_SERVICE=gemini
GOOGLE_API_KEY=your_gemini_key  # Get from https://makersuite.google.com/app/apikey
MARKER_GEMINI_MODEL=gemini-2.0-flash  # Or gemini-2.5-flash
```
- **Cost:** ~$0.10/million tokens (very cheap)
- **Speed:** Very fast
- **Quality:** Excellent for scientific papers

**Option B: Vertex AI (Production-grade)**
```bash
MARKER_LOCAL_LLM_SERVICE=vertex
MARKER_VERTEX_PROJECT_ID=your-gcp-project-id
```
- **Cost:** Similar to Gemini but with SLA
- **Reliability:** More reliable than Gemini API
- **Requires:** Google Cloud Project

**Option C: Claude (Alternative)**
```bash
MARKER_LOCAL_LLM_SERVICE=claude
CLAUDE_API_KEY=your_claude_key  # Get from https://console.anthropic.com/
MARKER_CLAUDE_MODEL=claude-3-sonnet-20240229
```
- **Cost:** ~$3-15/million tokens
- **Quality:** Excellent reasoning
- **Models:** claude-3-sonnet, claude-3-opus

**Option D: OpenAI (Alternative)**
```bash
MARKER_LOCAL_LLM_SERVICE=openai
OPENAI_API_KEY=your_openai_key  # Get from https://platform.openai.com/api-keys
MARKER_OPENAI_MODEL=gpt-4-turbo
```
- **Cost:** ~$10-30/million tokens
- **Quality:** Good all-around
- **Models:** gpt-4-turbo, gpt-4, gpt-3.5-turbo

**Option E: Ollama (100% Free, Local)** 💚
```bash
MARKER_LOCAL_LLM_SERVICE=ollama
MARKER_OLLAMA_BASE_URL=http://localhost:11434
MARKER_OLLAMA_MODEL=llama2  # Or mistral, mixtral, etc.
```
- **Cost:** FREE! Runs on your machine
- **Install:** https://ollama.ai/
- **Models:** llama2, mistral, mixtral, etc.
- **Requires:** ~4-8GB VRAM for good models

#### Step 3: Test

```bash
python test_marker_comparison.py --all
```

**Advantages:**
- ✅ 100% free (unlimited)
- ✅ No internet needed (after install)
- ✅ Faster on GPU
- ✅ Full control

**Disadvantages:**
- ❌ Large installation (~3-5GB)
- ❌ Slower on CPU
- ❌ Requires 8GB+ RAM
- ❌ Manual updates needed

---

### Option 3: PyMuPDF (Default)

**Best for:** Speed, simplicity, minimal resources

```bash
# Already configured by default
PDF_EXTRACTION_METHOD=pymupdf
```

**No additional setup needed!**

---

## Configuration

### Full Configuration Options

```bash
# ===================================================================
# PDF Extraction Method
# ===================================================================
# Options: "pymupdf", "marker_api", "marker_local"
PDF_EXTRACTION_METHOD=pymupdf

# ===================================================================
# Marker API Configuration
# ===================================================================
MARKER_API_KEY=                      # Get from datalab.to
MARKER_USE_LLM=true                  # Better tables/equations (recommended)
MARKER_FORCE_OCR=false               # Force OCR even if text embedded
MARKER_API_TIMEOUT=180               # Timeout in seconds (3 minutes)

# ===================================================================
# Marker Local Configuration
# ===================================================================
MARKER_LOCAL_BATCH_MULTIPLIER=1      # GPU batch size (1-4)
MARKER_LOCAL_USE_LLM=false           # Requires LLM API key
MARKER_LOCAL_LLM_PROVIDER=openai     # openai, anthropic, google
MARKER_LOCAL_LLM_MODEL=gpt-4         # Model name

# ===================================================================
# Fallback Settings
# ===================================================================
MARKER_FALLBACK_TO_PYMUPDF=true      # Fallback if Marker fails
```

---

## Testing & Comparison

### Compare All Methods

```bash
# Test PyMuPDF + Marker API
python test_marker_comparison.py path/to/paper.pdf

# Test all three methods
python test_marker_comparison.py path/to/paper.pdf --all

# Test without LLM (faster, lower quality)
python test_marker_comparison.py --no-llm

# Auto-select a paper from your library
python test_marker_comparison.py
```

### Sample Output

```
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Method        ┃ Status   ┃ Time (s) ┃ Text Chars ┃ Images ┃ Markdown ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ PyMuPDF       │ ✓ Success│     2.34 │     45,231 │     12 │ ✗        │
│ Marker API    │ ✓ Success│    15.67 │     52,108 │     12 │ ✓        │
│ Marker Local  │ ✓ Success│     8.92 │     51,943 │     12 │ ✓        │
└───────────────┴──────────┴──────────┴────────────┴────────┴──────────┘

Fastest: PyMuPDF (2.34s)
  PyMuPDF is 6.7x faster than Marker API
  PyMuPDF is 3.8x faster than Marker Local
```

---

## Troubleshooting

### Marker API Issues

**Problem:** `Invalid API key`
```bash
# Check your .env file
cat .env | grep MARKER_API_KEY

# Make sure key is set (no quotes)
MARKER_API_KEY=pa_abc123...
```

**Problem:** `Rate limited`
```bash
# Wait a few seconds, the extractor will retry automatically
# Or reduce concurrent requests
```

**Problem:** `Request timeout`
```bash
# Increase timeout in .env
MARKER_API_TIMEOUT=300  # 5 minutes for large PDFs
```

### Marker Local Issues

**Problem:** `marker-pdf not installed`
```bash
pip install marker-pdf
```

**Problem:** `Out of memory`
```bash
# Reduce batch multiplier
MARKER_LOCAL_BATCH_MULTIPLIER=1

# Or close other applications
# Marker needs ~4GB RAM minimum
```

**Problem:** `CUDA out of memory`
```bash
# Use CPU instead (slower but works)
# Or reduce batch multiplier
MARKER_LOCAL_BATCH_MULTIPLIER=1
```

**Problem:** `Models downloading slowly`
```bash
# First run downloads ~1-2GB of models
# Be patient, this only happens once
# Models are cached in ~/.cache/huggingface/
```

### PyMuPDF Fallback

If Marker fails, the system automatically falls back to PyMuPDF (if enabled):

```bash
# Check logs for fallback messages
WARN: Marker API failed for paper.pdf: Connection timeout
INFO: Falling back to PyMuPDF for paper.pdf
```

---

## Cost Estimation

### Marker API Pricing

**Free tier:** $5 credit (~250-500 pages)

**Estimated costs:**

| Library Size | Pages | One-time Cost | Monthly (new papers) |
|--------------|-------|---------------|----------------------|
| 50 papers    | 750   | $10-15        | $2-5                 |
| 160 papers   | 2,400 | $30-50        | $5-10                |
| 500 papers   | 7,500 | $100-150      | $10-20               |
| 1,000 papers | 15,000| $200-300      | $20-40               |

**Notes:**
- Prices are approximate (~$0.01-0.02/page)
- LLM mode costs slightly more but much better quality
- First $5 is free
- One-time cost for initial indexing
- Monthly cost assumes 10-20 new papers/month

### Marker Local Pricing

**Installation:** FREE (open source)

**Running costs:**
- Electricity for GPU: ~$0.10-0.50/hour
- No per-page costs
- Unlimited processing

**Break-even point:** ~500-1,000 pages (compared to API)

---

## Recommendations

### For 160 papers (~2,400 pages)

**Option A: Marker API** ⭐ RECOMMENDED
- Initial cost: ~$40 ($5 free + $35)
- Monthly: ~$5-10
- Best quality/effort ratio

**Option B: Hybrid Approach** 💡 SMART
```bash
# Use Marker API for complex papers with tables/equations
# Use PyMuPDF for simple text-heavy papers
# Manually toggle PDF_EXTRACTION_METHOD as needed
```

**Option C: Marker Local** 🔧 POWER USER
- Free but requires setup
- Good if you have GPU
- ~3-5GB disk space

**Option D: PyMuPDF Only** ⚡ FAST & FREE
- Already works
- Good enough for most papers
- Zero setup, zero cost

---

## Next Steps

1. **Choose your method** based on your needs and budget
2. **Configure `.env`** with your settings
3. **Run the test script** to compare quality
4. **Index your library** with your chosen method
5. **Monitor costs** (if using API)

Questions? Check the [Datalab documentation](https://documentation.datalab.to/) or open an issue.

---

## Additional Resources

- **Marker GitHub:** https://github.com/datalab-to/marker
- **Datalab API Docs:** https://documentation.datalab.to/docs/welcome/api
- **Pricing:** https://www.datalab.to/pricing
- **PyMuPDF Docs:** https://pymupdf.readthedocs.io/

---

**Happy extracting!** 🚀
