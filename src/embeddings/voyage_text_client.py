"""Voyage AI Text-Only Embedding Client with Contextualized Embeddings

This module provides a simplified client for Voyage AI that focuses on text-only
embeddings using the voyage-context-3 model.

Key Features:
- Text-only processing (filters out multimodal content)
- Contextualized embeddings via contextualized_embed() API
- Simpler architecture than hybrid client
- Compatible with SentenceTransformer interface
"""

import logging
import time
from typing import List, Union
import numpy as np
from tqdm import tqdm

try:
    import voyageai
except ImportError:
    raise ImportError(
        "voyageai package not installed. Install with: pip install voyageai"
    )

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class VoyageTextEmbeddingClient:
    """
    Text-only Voyage AI embedding client with contextualized embeddings.

    Uses voyage-context-3 model which provides:
    - 14.24% better performance than OpenAI text-embedding-3-large
    - Contextual understanding across document chunks
    - Optimized for document retrieval

    Features:
    - Filters out multimodal content automatically
    - Contextualized embeddings via contextualized_embed() API
    - Batch processing with progress bars
    - Retry logic with exponential backoff
    - Compatible with SentenceTransformer interface
    """

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-context-3",
        max_retries: int = 3,
        timeout: int = 60,
    ):
        """
        Initialize Voyage AI text embedding client.

        Args:
            api_key: Voyage AI API key
            model: Model name (default: voyage-context-3)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize Voyage AI client
        self.client = voyageai.Client(api_key=self.api_key)

        logger.info(
            f"Voyage AI text client initialized:\n"
            f"  - Model: {self.model}\n"
            f"  - Mode: Text-only (filters out multimodal content)"
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
        Encode texts using Voyage AI with contextualized embeddings.

        Compatible with SentenceTransformer.encode() interface.

        Automatically filters out multimodal content (dicts) and processes only
        text chunks (strings).

        Args:
            sentences: Various input formats:
                - Single string: "text"
                - List of strings: ["text1", "text2"]
                - List of lists: [["doc1_chunk1", "doc1_chunk2"], ["doc2_chunk1"]]
                - Mixed with dicts: Will filter out dicts, keep only strings
            batch_size: Number of documents to encode per API call
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
        # Handle list of dicts (multimodal) - filter to empty
        elif isinstance(sentences, list) and len(sentences) > 0 and all(isinstance(s, dict) for s in sentences):
            logger.warning(f"Received {len(sentences)} multimodal chunks - filtering out (text-only mode)")
            raise ValueError("Text-only client received only multimodal content. No text to embed.")
        # Handle MIXED flat list (strings + dicts) - filter out dicts
        elif isinstance(sentences, list) and len(sentences) > 0 and any(isinstance(s, str) for s in sentences) and any(isinstance(s, dict) for s in sentences):
            logger.info(f"Received mixed flat list with {sum(isinstance(s, str) for s in sentences)} text chunks and {sum(isinstance(s, dict) for s in sentences)} multimodal chunks")
            text_chunks = [s for s in sentences if isinstance(s, str)]
            if len(text_chunks) == 0:
                raise ValueError("All items in flat list are multimodal after filtering")
            logger.info(f"Filtered to {len(text_chunks)} text chunks (removed {len(sentences) - len(text_chunks)} multimodal chunks)")
            sentences = [[s] for s in text_chunks]  # Each text string becomes a single-chunk doc
            single_input = False
        # Handle list of lists (documents with multiple chunks)
        elif isinstance(sentences, list) and len(sentences) > 0 and all(isinstance(s, list) for s in sentences):
            # Filter out multimodal chunks and validate
            filtered_documents = []
            total_filtered = 0

            for doc_idx, document in enumerate(sentences):
                if not isinstance(document, list):
                    continue
                if len(document) == 0:
                    logger.warning(f"Empty document at index {doc_idx}, skipping")
                    continue

                # Filter out dicts (multimodal), keep only strings (text)
                text_chunks = [chunk for chunk in document if isinstance(chunk, str)]
                filtered_count = len(document) - len(text_chunks)

                if filtered_count > 0:
                    total_filtered += filtered_count
                    logger.debug(f"Filtered {filtered_count} multimodal chunks from document {doc_idx}")

                if len(text_chunks) > 0:
                    filtered_documents.append(text_chunks)

            if total_filtered > 0:
                logger.info(f"Filtered out {total_filtered} multimodal chunks (text-only mode)")

            if len(filtered_documents) == 0:
                raise ValueError("All documents are empty or contain only multimodal content after filtering")

            sentences = filtered_documents
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
                    f"Expected all strings or all lists."
                )
            else:
                raise ValueError(
                    f"Invalid input format. Expected str, List[str], or List[List[str]], "
                    f"got {type(sentences)}"
                )

        total_chunks = sum(len(doc) for doc in sentences)
        logger.info(
            f"Encoding {len(sentences)} documents with contextualized embeddings "
            f"({total_chunks} total text chunks)"
        )

        # Encode documents in batches
        all_document_embeddings = []

        num_batches = (len(sentences) + batch_size - 1) // batch_size
        iterator = range(0, len(sentences), batch_size)

        if show_progress_bar and len(sentences) > batch_size:
            iterator = tqdm(
                iterator,
                desc=f"Encoding with {self.model}",
                total=num_batches,
                unit="batch"
            )

        for i in iterator:
            batch = sentences[i:i + batch_size]
            document_embeddings = self._encode_documents_with_retry(batch)
            all_document_embeddings.extend(document_embeddings)

        # Flatten document embeddings into chunk embeddings
        all_embeddings = []
        for doc_embeds in all_document_embeddings:
            all_embeddings.extend(doc_embeds)

        # Convert to numpy array
        result = np.array(all_embeddings, dtype=np.float32)

        # Return single embedding if single input
        if single_input:
            return result[0]

        return result

    def _encode_documents_with_retry(self, documents: List[List[str]]) -> List[List[np.ndarray]]:
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
                    model=self.model,
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
                    f"Voyage API request failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )

                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)

        # All retries failed
        error_msg = f"Failed to encode document batch after {self.max_retries} attempts: {last_exception}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def __repr__(self):
        return (
            f"VoyageTextEmbeddingClient(\n"
            f"  model={self.model},\n"
            f"  mode=text-only\n"
            f")"
        )
