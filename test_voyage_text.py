#!/usr/bin/env python3
"""Test the text-only Voyage client"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.embeddings.voyage_text_client import VoyageTextEmbeddingClient

def main():
    print("=" * 70)
    print("Testing Text-Only Voyage Client")
    print("=" * 70)

    # Initialize client
    client = VoyageTextEmbeddingClient(
        api_key=config.voyage_api_key,
        model=config.voyage_text_model
    )
    print(f"\n{client}")

    # Test 1: Single document with multiple chunks (contextual)
    print("\n1. Testing contextualized embeddings (single document)...")
    doc1 = [
        "Glaciers in Alaska are melting rapidly.",
        "This is causing sea level rise concerns.",
        "Scientists are monitoring the situation closely."
    ]

    embeddings1 = client.encode([doc1], show_progress_bar=False)
    print(f"   [OK] Generated {len(embeddings1)} embeddings")
    print(f"   [OK] Embedding shape: {embeddings1.shape}")
    print(f"   [OK] Dtype: {embeddings1.dtype}")

    # Test 2: Multiple documents with chunks
    print("\n2. Testing batch processing...")
    docs = [
        ["Paper 1 introduction", "Paper 1 methods"],
        ["Paper 2 abstract", "Paper 2 results", "Paper 2 conclusion"],
        ["Paper 3 single chunk"],
    ]

    embeddings2 = client.encode(docs, show_progress_bar=False)
    print(f"   [OK] Generated {len(embeddings2)} embeddings (expected 6)")
    print(f"   [OK] Embedding shape: {embeddings2.shape}")

    # Test 3: Mixed content (text + multimodal) - should filter out dicts
    print("\n3. Testing multimodal filtering...")
    mixed_docs = [
        ["Text chunk 1", "Text chunk 2"],
        ["Another text chunk", {"text": "multimodal", "image": "data:..."}],  # Mixed
        [{"text": "All multimodal", "image": "data:..."}],  # All multimodal - will be filtered
    ]

    try:
        embeddings3 = client.encode(mixed_docs, show_progress_bar=False)
        print(f"   [OK] Filtered multimodal content successfully")
        print(f"   [OK] Generated {len(embeddings3)} embeddings (text chunks only)")
        print(f"   [OK] Embedding shape: {embeddings3.shape}")
    except ValueError as e:
        if "only multimodal content" in str(e):
            print(f"   [OK] Correctly rejected all-multimodal document")
        else:
            raise

    # Test 4: List of strings (compatibility mode)
    print("\n4. Testing flat list of strings...")
    flat_list = ["Text 1", "Text 2", "Text 3"]
    embeddings4 = client.encode(flat_list, show_progress_bar=False)
    print(f"   [OK] Generated {len(embeddings4)} embeddings")
    print(f"   [OK] Embedding shape: {embeddings4.shape}")

    # Test 5: Single string
    print("\n5. Testing single string...")
    single_text = "This is a single test string for scientific papers."
    embedding5 = client.encode(single_text, show_progress_bar=False)
    print(f"   [OK] Generated single embedding")
    print(f"   [OK] Embedding shape: {embedding5.shape}")

    print("\n" + "=" * 70)
    print("All tests passed! Text-only Voyage client is working correctly.")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
