"""Test script for multimodal extraction and embedding"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.extractors.pdf_extractor import extract_text_from_pdf, extract_images_from_pdf
from src.embeddings.jina_api_client import JinaV4EmbeddingClient
from src.config import config

def test_image_extraction():
    """Test PDF image extraction"""
    print("=" * 80)
    print("Testing Image Extraction")
    print("=" * 80)

    # Find a sample PDF
    zotero_path = Path(config.documents_path)
    pdf_files = list(zotero_path.rglob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in Zotero library")
        return None

    # Take first PDF
    sample_pdf = pdf_files[0]
    print(f"\n📄 Testing with: {sample_pdf.name}")

    # Extract images
    try:
        images = extract_images_from_pdf(sample_pdf, max_images_per_page=2)
        print(f"✅ Extracted {len(images)} images")

        if images:
            for i, img in enumerate(images[:3]):  # Show first 3
                print(f"\n  Image {i+1}:")
                print(f"    - Page: {img['page_num'] + 1}")
                print(f"    - Size: {img['width']}x{img['height']} pixels")
                print(f"    - Format: {img['format']}")
                print(f"    - Base64 length: {len(img['image_base64'])} chars")
        else:
            print("  ⚠️  No images found in this PDF")

        return images[0] if images else None

    except Exception as e:
        print(f"❌ Error extracting images: {e}")
        return None


def test_text_and_images_extraction():
    """Test combined text + image extraction"""
    print("\n" + "=" * 80)
    print("Testing Combined Text + Image Extraction")
    print("=" * 80)

    zotero_path = Path(config.documents_path)
    pdf_files = list(zotero_path.rglob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found")
        return None, None

    sample_pdf = pdf_files[0]
    print(f"\n📄 Testing with: {sample_pdf.name}")

    try:
        text, is_scanned, images = extract_text_from_pdf(sample_pdf, extract_images=True)

        print(f"\n✅ Text extraction:")
        print(f"  - Characters: {len(text)}")
        print(f"  - Is scanned: {is_scanned}")
        print(f"  - First 200 chars: {text[:200]}...")

        print(f"\n✅ Image extraction:")
        print(f"  - Total images: {len(images)}")

        return text[:500], images[0] if images else None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None


def test_multimodal_embedding():
    """Test multimodal embedding with Jina API"""
    print("\n" + "=" * 80)
    print("Testing Multimodal Embedding API")
    print("=" * 80)

    if not config.use_jina_api:
        print("❌ Jina API is not enabled in config")
        return

    if not config.jina_api_key:
        print("❌ Jina API key not configured")
        return

    # Initialize client
    print(f"\n🔧 Initializing Jina API client...")
    try:
        client = JinaV4EmbeddingClient(
            api_key=config.jina_api_key,
            model=config.jina_model
        )
        print("✅ Client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return

    # Test 1: Text-only embedding
    print("\n📝 Test 1: Text-only embedding")
    try:
        text_embedding = client.encode("Glacier albedo is affected by aerosols")
        print(f"✅ Text embedding shape: {text_embedding.shape}")
        print(f"  - First 5 values: {text_embedding[:5]}")
    except Exception as e:
        print(f"❌ Text embedding failed: {e}")
        return

    # Test 2: Multimodal embedding (if we have an image)
    text_sample, image_sample = test_text_and_images_extraction()

    if text_sample and image_sample:
        print("\n🖼️  Test 2: Multimodal embedding (text + image)")
        try:
            multimodal_input = [
                {
                    "text": text_sample,
                    "image": image_sample["image_base64"]
                }
            ]

            mm_embedding = client.encode(multimodal_input)
            print(f"✅ Multimodal embedding shape: {mm_embedding.shape}")
            print(f"  - First 5 values: {mm_embedding[0][:5]}")

        except Exception as e:
            print(f"❌ Multimodal embedding failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  Skipping multimodal test (no image available)")


if __name__ == "__main__":
    print("\n🧪 MULTIMODAL FUNCTIONALITY TEST")
    print("=" * 80)

    # Test 1: Image extraction
    test_image_extraction()

    # Test 2: Multimodal embedding
    test_multimodal_embedding()

    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)
