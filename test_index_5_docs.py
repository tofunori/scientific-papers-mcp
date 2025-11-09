#!/usr/bin/env python
"""Test script: Index 5 documents with Qwen3-4B to verify configuration"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.extractors.pdf_extractor import extract_text_from_pdf
from src.indexing.chroma_client import ChromaClientManager
from src.indexing.hybrid_search import HybridSearchEngine
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

print("\n" + "=" * 60)
print("TEST: Index 5 Documents with Qwen3-4B @ 2048 dims")
print("=" * 60)
print(f"ChromaDB path:  {config.chroma_path}")
print(f"Model:          {config.embedding_model}")
print(f"Dimensions:     {config.embedding_dimensions}")
print(f"Batch size:     {config.batch_size}")
print("=" * 60 + "\n")

# Initialize Chroma
logger.info("Initializing Chroma client...")
chroma_mgr = ChromaClientManager()
collection = chroma_mgr.get_or_create_collection("scientific_papers")
logger.info(f"Collection ready (count: {collection.count()})")

# Initialize hybrid search
logger.info("Initializing hybrid search engine...")
search_engine = HybridSearchEngine(collection, config.embedding_model)

# Get first 5 PDF files
doc_path = config.documents_path
pdf_files = sorted(list(doc_path.glob("**/*.pdf")))[:5]

logger.info(f"Found {len(pdf_files)} PDF files to test")

# Index 5 documents
doc_ids = []
texts = []
metadatas = []

for i, pdf_file in enumerate(pdf_files, 1):
    logger.info(f"\nProcessing {i}/5: {pdf_file.name[:60]}...")
    
    try:
        # Extract text
        text = extract_text_from_pdf(pdf_file)
        if not text or len(text.strip()) < 50:
            logger.warning(f"  Skipped (text too short)")
            continue
        
        doc_id = f"test_{i}_{pdf_file.stem[:30]}"
        doc_ids.append(doc_id)
        texts.append(text[:5000])  # First 5000 chars
        metadatas.append({
            "filename": pdf_file.name,
            "path": str(pdf_file),
            "test": "true"
        })
        
        logger.info(f"  Extracted {len(text)} chars")
    
    except Exception as e:
        logger.error(f"  Error: {e}")

# Batch index
if doc_ids:
    logger.info(f"\nBatch indexing {len(doc_ids)} documents...")
    search_engine.index_documents_batch(doc_ids, texts, metadatas)
    logger.info(f"Successfully indexed {len(doc_ids)} documents")

# Verify collection
collection_updated = chroma_mgr.get_or_create_collection("scientific_papers")
count = collection_updated.count()
logger.info(f"\nCollection now has {count} documents")

# Test search
logger.info("\nTesting search functionality...")
try:
    results = collection_updated.query(
        query_texts=["glacier albedo"],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    
    if results and results['documents'] and len(results['documents'][0]) > 0:
        logger.info(f"SEARCH SUCCESS: Found {len(results['documents'][0])} results")
        for i, doc in enumerate(results['documents'][0], 1):
            logger.info(f"  {i}. {doc[:80]}...")
    else:
        logger.warning("SEARCH RETURNED NO RESULTS")
        
except Exception as e:
    logger.error(f"SEARCH ERROR: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60 + "\n")
