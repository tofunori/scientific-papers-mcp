"""Voyage AI Hybrid Embedding Client

This module provides a hybrid client that automatically routes chunks to the appropriate
Voyage AI model:
- Text chunks (strings) → voyage-context-3 (contextualized text embeddings)
- Multimodal chunks (dicts) → voyage-multimodal-3 (text + image embeddings)

Compatible with SentenceTransformer interface for drop-in replacement.
"""

import logging
import time
import base64
import io
from typing import List, Union, Dict
import numpy as np
from tqdm import tqdm

try:
    import voyageai
except ImportError:
    raise ImportError(
        "voyageai package not installed. Install with: pip install voyageai"
    )

try:
    from PIL import Image
except ImportError:
    raise ImportError(
        "PIL package not installed. Install with: pip install Pillow"
    )

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class VoyageHybridEmbeddingClient:
    """
    Hybrid Voyage AI embedding client that automatically routes chunks to optimal models.

    Architecture:
    - Text-only chunks → voyage-context-3 (14.24% better than OpenAI v3-large)
    - Multimodal chunks → voyage-multimodal-3 (19.63% better than competitors)

    Features:
    - Automatic model selection based on input type
    - Batch processing with progress bars
    - Retry logic with exponential backoff
    - Compatible with SentenceTransformer interface
    """

    def __init__(
        self,
        api_key: str,
        text_model: str = "voyage-context-3",
        multimodal_model: str = "voyage-multimodal-3",
        max_retries: int = 3,
        timeout: int = 60,
    ):
        """
        Initialize Voyage AI hybrid embedding client.

        Args:
            api_key: Voyage AI API key
            text_model: Model for text-only chunks (default: voyage-context-3)
            multimodal_model: Model for multimodal chunks (default: voyage-multimodal-3)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.text_model = text_model
        self.multimodal_model = multimodal_model
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize Voyage AI client
        self.client = voyageai.Client(api_key=self.api_key)

        logger.info(
            f"Voyage AI hybrid client initialized:\n"
            f"  - Text model: {self.text_model}\n"
            f"  - Multimodal model: {self.multimodal_model}"
        )

    def encode(
        self,
        sentences: Union[str, List[str], List[dict], List[List[str]], List[List[dict]]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Encode texts and/or multimodal inputs using appropriate Voyage AI models.

        Compatible with SentenceTransformer.encode() interface.

        Supports contextualized embeddings: chunks from the same document (inner list)
        are encoded together, allowing the model to understand inter-chunk relationships.

        Args:
            sentences: Various input formats:
                - Single string: "text"
                - List of strings: ["text1", "text2"] (independent chunks)
                - List of lists: [["doc1_chunk1", "doc1_chunk2"], ["doc2_chunk1"]]
                  (chunks grouped by document for contextual embeddings)
                - Mixed multimodal: List of dicts or lists of dicts
            batch_size: Number of documents (not chunks) to encode per API call
            show_progress_bar: Whether to show progress bar
            convert_to_tensor: Ignored (kept for compatibility)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            numpy.ndarray: Array of embeddings with shape (n_chunks, embedding_dim)
        """
        # Handle single string input
        if isinstance(sentences, str):
            sentences = [[sentences]]  # Wrap as single-chunk document
            single_input = True
        # Handle list of strings (treat each as independent single-chunk document)
        elif isinstance(sentences, list) and len(sentences) > 0 and all(isinstance(s, str) for s in sentences):
            sentences = [[s] for s in sentences]  # Each string becomes a single-chunk doc
            single_input = False
        # Handle list of dicts (multimodal, each as independent single-chunk document)
        elif isinstance(sentences, list) and len(sentences) > 0 and all(isinstance(s, dict) for s in sentences):
            sentences = [[s] for s in sentences]
            single_input = False
        # Handle list of lists (documents with multiple chunks)
        elif isinstance(sentences, list) and len(sentences) > 0 and all(isinstance(s, list) for s in sentences):
            # Validate that inner lists are not empty and contain valid types
            valid = True
            for i, doc in enumerate(sentences):
                if not isinstance(doc, list):
                    valid = False
                    break
                if len(doc) == 0:
                    logger.warning(f"Empty document at index {i}, skipping")
                    continue
                # Check first element to determine type
                first_elem = doc[0]
                if not isinstance(first_elem, (str, dict)):
                    logger.error(f"Invalid inner list at index {i}: first element is {type(first_elem)}, expected str or dict")
                    valid = False
                    break

            if not valid:
                raise ValueError(
                    f"Invalid input format in nested list. Each inner list must contain strings or dicts. "
                    f"First invalid item at index {i}: {type(sentences[i])} with first element type {type(first_elem) if doc else 'empty'}"
                )

            # Filter out empty documents
            sentences = [doc for doc in sentences if len(doc) > 0]
            if len(sentences) == 0:
                raise ValueError("All documents are empty after filtering")

            single_input = False
        # Handle empty list
        elif isinstance(sentences, list) and len(sentences) == 0:
            raise ValueError("Cannot encode empty list")
        else:
            # Provide detailed error message
            if isinstance(sentences, list) and len(sentences) > 0:
                types_found = set(type(s).__name__ for s in sentences)
                logger.error(f"Mixed or invalid types in list: {types_found}")
                raise ValueError(
                    f"Invalid input format. List contains mixed or invalid types: {types_found}. "
                    f"Expected all strings, all dicts, or all lists."
                )
            else:
                raise ValueError(
                    f"Invalid input format. Expected str, List[str], List[dict], "
                    f"List[List[str]], or List[List[dict]], got {type(sentences)}"
                )

        # Now sentences is always List[List[...]]
        # Separate documents by type (text vs multimodal)
        text_documents = []
        multimodal_documents = []
        document_types = []  # Track original order and type

        total_chunks = 0
        for doc_idx, document in enumerate(sentences):
            # Check if this document contains any multimodal chunks
            has_multimodal = any(isinstance(chunk, dict) for chunk in document)

            if has_multimodal:
                multimodal_documents.append((doc_idx, document))
                document_types.append(("multimodal", len(multimodal_documents) - 1))
            else:
                text_documents.append((doc_idx, document))
                document_types.append(("text", len(text_documents) - 1))

            total_chunks += len(document)

        logger.info(
            f"Encoding {len(sentences)} documents ({total_chunks} chunks): "
            f"{len(text_documents)} text docs → {self.text_model}, "
            f"{len(multimodal_documents)} multimodal docs → {self.multimodal_model}"
        )

        # Encode text documents (with contextualized chunks)
        text_embeddings = {}
        if text_documents:
            text_docs = [doc for _, doc in text_documents]
            text_embeds = self._encode_text_documents(
                text_docs, batch_size, show_progress_bar
            )
            # Map back to original document indices
            for (doc_idx, _), doc_embeds in zip(text_documents, text_embeds):
                text_embeddings[doc_idx] = doc_embeds

        # Encode multimodal documents
        multimodal_embeddings = {}
        if multimodal_documents:
            mm_docs = [doc for _, doc in multimodal_documents]
            mm_embeds = self._encode_multimodal_documents(
                mm_docs, batch_size, show_progress_bar
            )
            # Map back to original document indices
            for (doc_idx, _), doc_embeds in zip(multimodal_documents, mm_embeds):
                multimodal_embeddings[doc_idx] = doc_embeds

        # Reconstruct in original order (flatten document embeddings into chunk embeddings)
        all_embeddings = []
        for doc_idx in range(len(sentences)):
            if doc_idx in text_embeddings:
                all_embeddings.extend(text_embeddings[doc_idx])
            elif doc_idx in multimodal_embeddings:
                all_embeddings.extend(multimodal_embeddings[doc_idx])
            else:
                raise ValueError(f"Missing embedding for document index {doc_idx}")

        # Convert to numpy array
        result = np.array(all_embeddings, dtype=np.float32)

        # Return single embedding if single input
        if single_input:
            return result[0]

        return result

    def _encode_text_documents(
        self,
        documents: List[List[str]],
        batch_size: int,
        show_progress_bar: bool
    ) -> List[List[np.ndarray]]:
        """
        Encode text-only documents using voyage-context-3 with contextualized embeddings.

        Each document is a list of chunks that are encoded together, allowing the model
        to understand inter-chunk relationships.

        Args:
            documents: List of documents, where each document is a list of text chunks
                      Example: [["doc1_chunk1", "doc1_chunk2"], ["doc2_chunk1"]]
            batch_size: Number of documents (not chunks) to encode per API call
            show_progress_bar: Show progress bar

        Returns:
            List of document embeddings, where each document has a list of chunk embeddings
            Example: [[emb1_1, emb1_2], [emb2_1]]
        """
        all_document_embeddings = []

        num_batches = (len(documents) + batch_size - 1) // batch_size
        iterator = range(0, len(documents), batch_size)

        if show_progress_bar and len(documents) > batch_size:
            iterator = tqdm(
                iterator,
                desc=f"Encoding text docs ({self.text_model})",
                total=num_batches,
                unit="batch"
            )

        for i in iterator:
            batch = documents[i:i + batch_size]
            # _encode_text_with_retry now expects List[List[str]] and returns List[List[np.ndarray]]
            document_embeddings = self._encode_text_with_retry(batch)
            all_document_embeddings.extend(document_embeddings)

        return all_document_embeddings

    def _encode_multimodal_documents(
        self,
        documents: List[List[Dict]],
        batch_size: int,
        show_progress_bar: bool
    ) -> List[List[np.ndarray]]:
        """
        Encode multimodal documents using voyage-multimodal-3 with contextualized embeddings.

        Args:
            documents: List of documents, where each document is a list of multimodal chunks
                      Each chunk is a dict with "text" and "image" keys
            batch_size: Number of documents to encode per API call
            show_progress_bar: Show progress bar

        Returns:
            List of document embeddings, where each document has a list of chunk embeddings
        """
        all_document_embeddings = []

        num_batches = (len(documents) + batch_size - 1) // batch_size
        iterator = range(0, len(documents), batch_size)

        if show_progress_bar and len(documents) > batch_size:
            iterator = tqdm(
                iterator,
                desc=f"Encoding multimodal docs ({self.multimodal_model})",
                total=num_batches,
                unit="batch"
            )

        for i in iterator:
            batch = documents[i:i + batch_size]
            document_embeddings = self._encode_multimodal_with_retry(batch)
            all_document_embeddings.extend(document_embeddings)

        return all_document_embeddings

    def _encode_text_with_retry(self, documents: List[List[str]]) -> List[List[np.ndarray]]:
        """
        Encode document batch with retry logic using contextualized embeddings.

        Uses Voyage AI's contextualized_embed() API which encodes each chunk in the context
        of other chunks from the same document.

        Args:
            documents: List of documents, where each document is a list of text chunks
                      Example: [["doc1_chunk1", "doc1_chunk2"], ["doc2_chunk1"]]

        Returns:
            List of document embeddings, where each document has a list of chunk embeddings
            Example: [[emb1_1, emb1_2], [emb2_1]]

        Raises:
            Exception: If all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # voyage-context-3 uses contextualized_embed() API
                # Pass documents as list of lists for contextual understanding
                result = self.client.contextualized_embed(
                    inputs=documents,  # Each inner list is a document with chunks
                    model=self.text_model,
                    input_type="document",  # For indexing documents
                )

                # Extract embeddings: result.results contains one ContextualizedEmbeddingsResult per document
                # Each result has .embeddings which is a list of embeddings (one per chunk)
                document_embeddings = []
                for res in result.results:
                    chunk_embeddings = [np.array(emb, dtype=np.float32) for emb in res.embeddings]
                    document_embeddings.append(chunk_embeddings)

                return document_embeddings

            except Exception as e:
                last_exception = e
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s

                logger.warning(
                    f"Text API request failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )

                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)

        # All retries failed
        error_msg = f"Failed to encode document batch after {self.max_retries} attempts: {last_exception}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def _encode_multimodal_with_retry(self, documents: List[List[Dict]]) -> List[List[np.ndarray]]:
        """
        Encode multimodal document batch with retry logic.

        Converts base64 images to PIL Images for Voyage AI API.

        Args:
            documents: List of documents, where each document is a list of multimodal chunks
                      Each chunk is a dict with "text" and "image" keys
                      image must be in data URI format: "data:image/jpeg;base64,..."

        Returns:
            List of document embeddings, where each document has a list of chunk embeddings

        Raises:
            Exception: If all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Flatten all chunks for API call (multimodal_embed doesn't support nested lists yet)
                # We'll track document boundaries to reconstruct later
                all_chunks = []
                chunk_counts = []

                for document in documents:
                    chunk_counts.append(len(document))
                    for chunk in document:
                        text = chunk.get("text", "")
                        image_data_uri = chunk.get("image", "")

                        # Extract base64 from data URI
                        if image_data_uri.startswith("data:image"):
                            base64_str = image_data_uri.split(",", 1)[1]
                        else:
                            base64_str = image_data_uri

                        # Decode and convert to PIL Image
                        image_bytes = base64.b64decode(base64_str)
                        pil_image = Image.open(io.BytesIO(image_bytes))

                        # Create Voyage API input: [text, PIL.Image]
                        all_chunks.append([text, pil_image])

                # Call Voyage multimodal_embed API
                result = self.client.multimodal_embed(
                    inputs=all_chunks,
                    model=self.multimodal_model,
                    input_type="document",
                )

                # Convert to numpy and reconstruct document structure
                flat_embeddings = [np.array(emb, dtype=np.float32) for emb in result.embeddings]

                # Split back into documents based on chunk_counts
                document_embeddings = []
                offset = 0
                for count in chunk_counts:
                    doc_embeds = flat_embeddings[offset:offset + count]
                    document_embeddings.append(doc_embeds)
                    offset += count

                return document_embeddings

            except Exception as e:
                last_exception = e
                wait_time = 2 ** attempt

                logger.warning(
                    f"Multimodal API request failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )

                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)

        # All retries failed
        error_msg = f"Failed to encode multimodal document batch after {self.max_retries} attempts: {last_exception}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def __repr__(self):
        return (
            f"VoyageHybridEmbeddingClient(\n"
            f"  text_model={self.text_model},\n"
            f"  multimodal_model={self.multimodal_model}\n"
            f")"
        )
