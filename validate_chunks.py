#!/usr/bin/env python3
"""Validation script to check chunk quality in ChromaDB"""

import chromadb
from pathlib import Path

# Connect to ChromaDB
chroma_path = Path("D:/Claude Code/scientific-papers-mcp/data/chroma")
client = chromadb.PersistentClient(path=str(chroma_path))
collection = client.get_collection("scientific_papers")

print("=" * 70)
print("ChromaDB Chunk Validation")
print("=" * 70)

# Get total count
total_count = collection.count()
print(f"Total chunks in collection: {total_count}")

# Get all chunks
results = collection.get(
    limit=total_count,
    include=['documents', 'metadatas']
)

# Analyze chunks
chunk_counts = {}
chunk_lengths = []

for doc, meta in zip(results['documents'], results['metadatas']):
    item_key = meta.get('item_key', 'unknown')
    chunk_index = meta.get('chunk_index', -1)
    total_chunks = meta.get('total_chunks', 1)

    if item_key not in chunk_counts:
        chunk_counts[item_key] = {
            'title': meta.get('title', 'Unknown'),
            'chunks': 0,
            'total_chunks_reported': total_chunks
        }

    chunk_counts[item_key]['chunks'] += 1
    chunk_lengths.append(len(doc))

print(f"\n{'-' * 70}")
print(f"Chunks per Document:")
print(f"{'-' * 70}")

for item_key, info in sorted(chunk_counts.items(), key=lambda x: x[1]['chunks'], reverse=True):
    # Handle Unicode characters in titles
    title = info['title'][:50].encode('ascii', errors='replace').decode('ascii')
    print(f"  {title:50} | {info['chunks']:2} chunks")

print(f"\n{'-' * 70}")
print(f"Chunk Length Statistics:")
print(f"{'-' * 70}")
print(f"  Average chunk length: {sum(chunk_lengths) // len(chunk_lengths)} chars")
print(f"  Min chunk length: {min(chunk_lengths)} chars")
print(f"  Max chunk length: {max(chunk_lengths)} chars")

print(f"\n{'-' * 70}")
print(f"Sample Chunks (first 3):")
print(f"{'-' * 70}")

for i in range(min(3, len(results['documents']))):
    doc = results['documents'][i]
    meta = results['metadatas'][i]

    print(f"\nChunk #{i+1}:")
    print(f"  Title: {meta.get('title', 'Unknown')[:60]}")
    print(f"  Chunk: {meta.get('chunk_index', -1)}/{meta.get('total_chunks', 1)}")
    print(f"  Chunk ID: {meta.get('chunk_id', 'unknown')}")
    print(f"  Length: {len(doc)} chars")
    print(f"  Has overlap: {meta.get('has_overlap', False)}")
    print(f"  Preview: {doc[:200]}...")

print(f"\n{'=' * 70}")
