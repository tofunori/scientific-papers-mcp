"""Test script for Voyage AI hybrid embedding"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.embeddings.voyage_hybrid_client import VoyageHybridEmbeddingClient
from src.extractors.pdf_extractor import extract_text_from_pdf
from src.config import config


def test_voyage_hybrid_client():
    """Test Voyage AI hybrid client initialization"""
    print("=" * 80)
    print("Testing Voyage AI Hybrid Client Initialization")
    print("=" * 80)

    if not config.voyage_api_key:
        print("ERROR: Voyage API key not configured in .env")
        return None

    try:
        client = VoyageHybridEmbeddingClient(
            api_key=config.voyage_api_key,
            text_model=config.voyage_text_model,
            multimodal_model=config.voyage_multimodal_model,
        )
        print(f"SUCCESS: Client initialized:\n{client}")
        return client

    except Exception as e:
        print(f"ERROR: Failed to initialize client: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_text_embedding(client):
    """Test text-only embedding with voyage-context-3"""
    print("\n" + "=" * 80)
    print("Testing Text-Only Embedding (voyage-context-3)")
    print("=" * 80)

    if not client:
        print("ERROR: Client not available")
        return

    test_text = "Glacier albedo is affected by aerosols from forest fires"

    try:
        print(f"\n Encoding text: '{test_text}'")
        embedding = client.encode(test_text)

        print(f"\nSUCCESS: Text embedding generated:")
        print(f"  - Shape: {embedding.shape}")
        print(f"  - Dtype: {embedding.dtype}")
        print(f"  - First 5 values: {embedding[:5]}")
        print(f"  - Last 5 values: {embedding[-5:]}")
        print(f"  - Mean: {np.mean(embedding):.6f}")
        print(f"  - Std: {np.std(embedding):.6f}")

        return embedding

    except Exception as e:
        print(f"ERROR: Text embedding failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multimodal_embedding(client):
    """Test multimodal embedding with voyage-multimodal-3"""
    print("\n" + "=" * 80)
    print("Testing Multimodal Embedding (voyage-multimodal-3)")
    print("=" * 80)

    if not client:
        print("ERROR: Client not available")
        return

    # Extract text and images from a sample PDF
    zotero_path = Path(config.documents_path)
    pdf_files = list(zotero_path.rglob("*.pdf"))

    if not pdf_files:
        print("ERROR: No PDF files found in Zotero library")
        return

    sample_pdf = pdf_files[0]
    print(f"\n Using PDF: {sample_pdf.name}")

    try:
        text, is_scanned, images = extract_text_from_pdf(sample_pdf, extract_images=True)

        if not images:
            print("WARNING:  No images found in this PDF, trying another...")
            for pdf in pdf_files[1:6]:  # Try up to 5 PDFs
                text, is_scanned, images = extract_text_from_pdf(pdf, extract_images=True)
                if images:
                    sample_pdf = pdf
                    print(f"SUCCESS: Found images in: {sample_pdf.name}")
                    break

        if not images:
            print("ERROR: No PDFs with images found")
            return

        # Create multimodal input
        multimodal_input = {
            "text": text[:500],  # First 500 chars
            "image": images[0]["image_base64"]
        }

        print(f"\n  Encoding multimodal input:")
        print(f"  - Text length: {len(multimodal_input['text'])} chars")
        print(f"  - Image size: {images[0]['width']}x{images[0]['height']} pixels")
        print(f"  - Image page: {images[0]['page_num'] + 1}")

        embedding = client.encode([multimodal_input])

        print(f"\nSUCCESS: Multimodal embedding generated:")
        print(f"  - Shape: {embedding.shape}")
        print(f"  - Dtype: {embedding.dtype}")
        print(f"  - First 5 values: {embedding[0][:5]}")
        print(f"  - Last 5 values: {embedding[0][-5:]}")
        print(f"  - Mean: {np.mean(embedding[0]):.6f}")
        print(f"  - Std: {np.std(embedding[0]):.6f}")

        return embedding

    except Exception as e:
        print(f"ERROR: Multimodal embedding failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_mixed_batch_embedding(client):
    """Test mixed batch with both text and multimodal inputs"""
    print("\n" + "=" * 80)
    print("Testing Mixed Batch Embedding (Text + Multimodal)")
    print("=" * 80)

    if not client:
        print("ERROR: Client not available")
        return

    # Extract an image from a sample PDF
    zotero_path = Path(config.documents_path)
    pdf_files = list(zotero_path.rglob("*.pdf"))

    if not pdf_files:
        print("ERROR: No PDF files found")
        return

    # Find a PDF with images
    sample_image = None
    for pdf in pdf_files[:10]:
        _, _, images = extract_text_from_pdf(pdf, extract_images=True)
        if images:
            sample_image = images[0]
            break

    if not sample_image:
        print("WARNING:  No images found, skipping mixed batch test")
        return

    # Create mixed inputs
    mixed_inputs = [
        "Text chunk 1: Glacier albedo measurements",
        "Text chunk 2: Black carbon deposition on snow",
        {
            "text": "Multimodal chunk with image",
            "image": sample_image["image_base64"]
        },
        "Text chunk 3: Remote sensing of ice surfaces",
    ]

    try:
        print(f"\n Encoding mixed batch:")
        print(f"  - Input 1: Text-only")
        print(f"  - Input 2: Text-only")
        print(f"  - Input 3: Multimodal (text + image)")
        print(f"  - Input 4: Text-only")

        embeddings = client.encode(mixed_inputs)

        print(f"\nSUCCESS: Mixed batch embeddings generated:")
        print(f"  - Shape: {embeddings.shape}")
        print(f"  - Expected: (4, embedding_dim)")
        print(f"  - All embeddings have same dimension: {len(set(emb.shape[0] for emb in embeddings)) == 1}")

        for i, emb in enumerate(embeddings):
            input_type = "Multimodal" if i == 2 else "Text"
            print(f"\n  Embedding {i+1} ({input_type}):")
            print(f"    - Mean: {np.mean(emb):.6f}")
            print(f"    - Std: {np.std(emb):.6f}")

        return embeddings

    except Exception as e:
        print(f"ERROR: Mixed batch embedding failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_jina(client):
    """Compare Voyage embeddings with Jina embeddings"""
    print("\n" + "=" * 80)
    print("Comparing Voyage vs Jina Embeddings")
    print("=" * 80)

    if not client:
        print("ERROR: Client not available")
        return

    if not config.use_jina_api or not config.jina_api_key:
        print("WARNING:  Jina API not configured, skipping comparison")
        return

    try:
        from src.embeddings.jina_api_client import JinaV4EmbeddingClient

        jina_client = JinaV4EmbeddingClient(
            api_key=config.jina_api_key,
            model=config.jina_model
        )

        test_text = "Glacier albedo is affected by aerosols"

        # Get Voyage embedding
        voyage_emb = client.encode(test_text)

        # Get Jina embedding
        jina_emb = jina_client.encode(test_text)

        print(f"\nSUCCESS: Comparison results:")
        print(f"\n  Voyage (voyage-context-3):")
        print(f"    - Dimensions: {voyage_emb.shape[0]}")
        print(f"    - Mean: {np.mean(voyage_emb):.6f}")
        print(f"    - Std: {np.std(voyage_emb):.6f}")
        print(f"\n  Jina (jina-embeddings-v4):")
        print(f"    - Dimensions: {jina_emb.shape[0]}")
        print(f"    - Mean: {np.mean(jina_emb):.6f}")
        print(f"    - Std: {np.std(jina_emb):.6f}")

        print(f"\n  Dimension difference: {abs(voyage_emb.shape[0] - jina_emb.shape[0])}")
        print(f"  (Voyage: 512-2048, Jina: 1024-2048 via Matryoshka)")

    except Exception as e:
        print(f"ERROR: Comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nVOYAGE AI HYBRID EMBEDDING TEST")
    print("=" * 80)

    # Test 1: Initialize client
    client = test_voyage_hybrid_client()

    if client:
        # Test 2: Text-only embedding
        test_text_embedding(client)

        # Test 3: Multimodal embedding
        test_multimodal_embedding(client)

        # Test 4: Mixed batch
        test_mixed_batch_embedding(client)

        # Test 5: Compare with Jina
        compare_with_jina(client)

    print("\n" + "=" * 80)
    print("SUCCESS: All tests completed!")
    print("=" * 80)
