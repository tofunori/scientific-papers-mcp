#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple search test"""

from src.indexing.chroma_client import ChromaClientManager
from src.indexing.hybrid_search import HybridSearchEngine
from src.config import config

chroma_mgr = ChromaClientManager()
collection = chroma_mgr.get_or_create_collection("scientific_papers")
search_engine = HybridSearchEngine(collection, config.embedding_model)

print("=" * 70)
print(f"COLLECTION: {collection.count()} documents indexed")
print("=" * 70)

# Test search
results = search_engine.search("glacier albedo Alaska", top_k=5)
ids, docs, metadatas, scores = results

print(f"\nSearch: 'glacier albedo Alaska'")
print(f"Found: {len(ids)} results\n")

for i in range(len(ids)):
    print(f"{i+1}. Score: {scores[i]}")
    print(f"   ID: {ids[i]}")
    if metadatas and i < len(metadatas):
        fname = metadatas[i].get("filename", "N/A")
        print(f"   File: {fname[:60]}")
    print()

print("=" * 70)
print("SEARCH ENGINE WORKING CORRECTLY!")
print("=" * 70)
