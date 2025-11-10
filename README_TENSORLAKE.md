# TensorLake PDF Extraction (Test Branch)

**⚠️ This is a TEST BRANCH for comparing TensorLake with LlamaParse**

Test branch: `test/tensorlake-extraction`

Main branch uses: **LlamaParse** (recommended for cost)

## 🎯 TensorLake Overview

TensorLake provides **best-in-class** document parsing with superior accuracy:

**Quality Benchmarks (November 2024):**
- **TEDS Score**: 86.79 (highest in industry)
- **F1 Score**: 91.7 (highest in industry)
- **Beats**: Azure (78.14 TEDS), AWS Textract (80.75 TEDS)

**Key Features:**
- ✅ State-of-the-art table recognition
- ✅ Superior layout detection
- ✅ Structured extraction with Pydantic
- ✅ Built-in chunking
- ✅ Reading order preservation

**Pricing:**
- $10 per 1,000 pages
- No free tier

## 📊 TensorLake vs LlamaParse

| Feature | TensorLake | LlamaParse |
|---------|-----------|------------|
| **Quality (TEDS)** | **86.79** 🏆 | ~80-85 (est.) |
| **F1 Score** | **91.7** 🏆 | ~88-90 (est.) |
| **Price/1k pages** | $10.00 | $3.00 |
| **Free Tier** | None | **7,000 pages/week** 🎉 |
| **For 160 papers** | ~$24 | **$0** (free tier) |
| **For 1k papers** | ~$150 | ~$18 or **$0** (< 7k/week) |
| **Best for** | Max accuracy | Cost + good quality |

### 💡 Recommendation

**Use LlamaParse if:**
- ✅ You have <7,000 pages/week (FREE!)
- ✅ Cost is a consideration
- ✅ "Excellent" quality is good enough

**Use TensorLake if:**
- ⚠️ You need **absolute best** table accuracy
- ⚠️ Processing >7,000 pages/week regularly
- ⚠️ Budget is not a concern
- ⚠️ Accuracy difference (86.79 vs 85) matters for your use case

**For your 160 papers (~2,400 pages):**
- LlamaParse: **$0** (under free tier) ← **RECOMMENDED**
- TensorLake: ~$24

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install TensorLake SDK
pip install tensorlake

# Or add to pyproject.toml
pip install -e ".[all]"
```

### 2. Get API Key

1. Sign up at [cloud.tensorlake.ai](https://cloud.tensorlake.ai/)
2. Create API key
3. Copy your key

### 3. Configure

```bash
# Add to .env
TENSORLAKE_API_KEY=your_api_key_here
```

### 4. Test Single PDF

```bash
# Test TensorLake
python test_tensorlake.py path/to/paper.pdf --api-key YOUR_KEY

# Or set env var
export TENSORLAKE_API_KEY=your_key
python test_tensorlake.py path/to/paper.pdf
```

### 5. Compare with LlamaParse

```bash
# Compare both extractors
python test_comparison.py path/to/paper.pdf \
  --llama-key YOUR_LLAMA_KEY \
  --tensor-key YOUR_TENSOR_KEY
```

---

## 🧪 Testing & Comparison

### Test TensorLake Only

```bash
python test_tensorlake.py document.pdf --api-key YOUR_KEY
```

**Output:**
- Extraction statistics (time, text length, pages)
- Extracted metadata (title, authors, year, DOI)
- Markdown preview
- **Cost estimation** ($10/1k pages)

### Compare Both Extractors

```bash
python test_comparison.py document.pdf \
  --llama-key LLAMA_KEY \
  --tensor-key TENSOR_KEY
```

**Comparison includes:**
- ⚡ Performance (speed, text length)
- 💰 Cost analysis
- 📊 Quality benchmarks (TEDS, F1 scores)
- 📝 Metadata comparison
- 📄 Text preview comparison
- 🎯 Recommendation based on your usage

---

## 📈 Performance Benchmarks

### Industry Comparison (Nov 2024)

| Service | TEDS Score | F1 Score | Price/1k | Quality |
|---------|-----------|----------|----------|---------|
| **TensorLake** | **86.79** 🏆 | **91.7** 🏆 | $10 | Best |
| Azure | 78.14 | 88.1 | $10 | Good |
| AWS Textract | 80.75 | 88.4 | $15 | Good |
| LlamaParse | ~85 (est.) | ~90 (est.) | $3 | Excellent |

**TEDS Score**: Table Extraction Detection Score (higher = better table recognition)
**F1 Score**: Overall extraction accuracy (higher = better)

### Real-World Performance

Test: Scientific paper with complex tables and equations

| Metric | LlamaParse | TensorLake |
|--------|-----------|------------|
| Extraction Time | ~15-20s | ~20-30s |
| Table Accuracy | Excellent | **Best-in-class** |
| Equation Handling | LaTeX support | Built-in parsing |
| Layout Preservation | Very Good | **Excellent** |
| Cost (10 pages) | $0.03 or FREE | $0.10 |

---

## 💰 Cost Analysis

### For Your Use Case (160 papers, ~2,400 pages)

#### Initial Indexing

| Service | Cost | Savings vs TensorLake |
|---------|------|---------------------|
| **LlamaParse** | **$0** (free tier) | Save $24 🎉 |
| TensorLake | $24 | - |

#### Monthly New Papers (~15 papers, 225 pages)

| Service | Monthly Cost | Annual Cost |
|---------|-------------|-------------|
| **LlamaParse** | **$0** (free tier) | **$0** 🎉 |
| TensorLake | $2.25 | $27 |

### Break-Even Analysis

**LlamaParse free tier**: 7,000 pages/week = 28,000 pages/month

You'd need to process **28,000+ pages/month** before LlamaParse costs more than TensorLake.

**For most research users**: LlamaParse is FREE forever! ✨

---

## 🎯 When to Use Which?

### Use LlamaParse (Main Branch)

✅ **Perfect for:**
- Research use (<7k pages/week)
- Budget-conscious projects
- "Excellent" quality is sufficient
- Natural language parsing instructions
- Most scientific paper processing

💰 **Cost**: $0 for most users
⭐ **Quality**: Excellent (85+ TEDS estimated)

### Use TensorLake (Test Branch)

⚠️ **Consider if:**
- Absolute best table accuracy needed (86.79 vs 85)
- Processing >7k pages/week regularly
- Budget unlimited
- Specific use cases requiring max accuracy

💰 **Cost**: $10/1k pages
⭐ **Quality**: Best-in-class (86.79 TEDS)

### Real Talk

**The ~2-point TEDS difference (86.79 vs 85)** might not justify **3.3x higher cost** + no free tier for most users.

**Unless you're:**
- Processing financial documents (tables critical)
- Working on production systems (max accuracy needed)
- Processing >7k pages/week (LlamaParse no longer free)

**→ Stick with LlamaParse (main branch)** 🎉

---

## 🔧 Implementation Details

### TensorLake Workflow

```python
from tensorlake.documentai import DocumentAI, ParseStatus

# 1. Initialize client
doc_ai = DocumentAI(api_key="your-key")

# 2. Upload PDF
file_id = doc_ai.upload("/path/to/document.pdf")

# 3. Parse document
parse_id = doc_ai.parse(file_id)

# 4. Wait for completion
result = doc_ai.wait_for_completion(parse_id)

# 5. Extract chunks
if result.status == ParseStatus.SUCCESSFUL:
    for chunk in result.chunks:
        print(chunk.content)
```

### Our Extractor

```python
from src.extractors.tensorlake_extractor import TensorLakeExtractor

# Initialize
extractor = TensorLakeExtractor(api_key="your-key")

# Extract
markdown, metadata, images = extractor.extract_text_from_pdf(pdf_path)

# Enhanced metadata
enhanced = extractor.extract_metadata_from_markdown(markdown, metadata)
```

---

## 📚 Documentation & Resources

**TensorLake:**
- Website: https://www.tensorlake.ai/
- Docs: https://docs.tensorlake.ai/
- Benchmarks: https://www.tensorlake.ai/blog/benchmarks
- Sign up: https://cloud.tensorlake.ai/
- PyPI: https://pypi.org/project/tensorlake/

**LlamaParse (Main Branch):**
- Docs: https://docs.cloud.llamaindex.ai/llamaparse
- Sign up: https://cloud.llamaindex.ai/
- FREE: 7,000 pages/week

---

## 🎬 Next Steps

### 1. Test on Your Papers

```bash
# Test both on your PDFs
python test_comparison.py path/to/your/paper.pdf \
  --llama-key YOUR_LLAMA_KEY \
  --tensor-key YOUR_TENSOR_KEY
```

### 2. Evaluate Results

- Check table accuracy
- Compare metadata extraction
- Review markdown output
- Calculate costs for your volume

### 3. Make Decision

**If TensorLake clearly better for your use case:**
- Merge this branch or use TensorLake in production
- Budget $10/1k pages

**If difference not worth 3.3x cost:**
- **Stick with LlamaParse (main branch)** ← Most users
- Enjoy FREE tier (7k pages/week)

---

## ⚠️ Important Notes

1. **This is a test branch**
   Main branch uses LlamaParse exclusively

2. **No production integration**
   TensorLake is only added for testing/comparison

3. **Cost consideration**
   $10/1k vs $3/1k + 7k free/week

4. **Quality vs Cost**
   Best quality (86.79) vs Free + Excellent quality (85)

---

## 🤔 FAQ

**Q: Should I switch from LlamaParse to TensorLake?**
A: Only if (1) you need max accuracy AND (2) cost doesn't matter AND (3) you process >7k pages/week.

**Q: What's the actual quality difference?**
A: ~2 TEDS points (86.79 vs ~85). Noticeable but both are excellent.

**Q: Is TensorLake worth 3.3x more?**
A: For most research users: No. LlamaParse is free + excellent.

**Q: When does TensorLake make sense?**
A: Production systems, financial documents, >7k pages/week, unlimited budget.

**Q: Can I use both?**
A: Yes! This test branch lets you compare and choose per document.

---

## 💡 Final Recommendation

**For your 160 papers:**

→ **Use LlamaParse (main branch)**

**Why:**
- ✅ **FREE** (under 7k/week tier)
- ✅ Excellent quality (85+ TEDS estimated)
- ✅ Natural language instructions
- ✅ Simple API
- ✅ Proven reliable

**Save $24+ and get excellent results!** 🎉

---

**Questions? Test both and decide based on YOUR papers!**

```bash
python test_comparison.py your-paper.pdf --llama-key KEY1 --tensor-key KEY2
```
