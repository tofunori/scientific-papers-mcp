"""Jina AI Embedding Client with Late Chunking Support

This module provides a client for Jina Embeddings v4 API with late chunking,
which preserves contextual information across document chunks.

Late Chunking: The API concatenates all chunks from a document, embeds the full
context, then returns individual chunk embeddings that understand the document context.
"""

import logging
import time
from typing import List, Union, Dict
import numpy as np
from tqdm import tqdm
import requests

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class JinaEmbeddingClient:
    """
    Jina AI embedding client with late chunking support.

    Late chunking enables contextual chunk embeddings by:
    1. Concatenating all chunks from a document
    2. Embedding the full concatenated text
    3. Internally chunking and returning embeddings that preserve context

    Features:
    - Late chunking for contextual embeddings
    - Task-specific adapters (retrieval.passage for indexing)
    - Batch processing with progress bars
    - Retry logic with exponential backoff
    - Compatible with SentenceTransformer interface
    """

    def __init__(
        self,
        api_key: str,
        model: str = "jina-embeddings-v4",
        max_retries: int = 3,
        timeout: int = 60,
    ):
        """
        Initialize Jina AI embedding client.

        Args:
            api_key: Jina AI API key
            model: Model name (default: jina-embeddings-v4)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.api_url = "https://api.jina.ai/v1/embeddings"

        logger.info(
            f"Jina AI client initialized with late chunking:\n"
            f"  - Model: {self.model}\n"
            f"  - Task: retrieval.passage (for document indexing)"
        )

    def encode(
        self,
        sentences: Union[str, List[str], List[List[str]]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Encode texts using Jina API with late chunking for contextual embeddings.

        Compatible with SentenceTransformer.encode() interface.

        Args:
            sentences: Input texts to encode:
                - Single string: "text"
                - List of strings: ["text1", "text2"] (treated as independent)
                - List of lists: [["doc1_chunk1", "doc1_chunk2"], ["doc2_chunk1"]]
                  (chunks from same document are encoded with shared context)
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
            # Filter out multimodal chunks (dicts) and validate
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

        logger.info(
            f"Encoding {len(sentences)} documents with late chunking "
            f"({sum(len(doc) for doc in sentences)} total chunks)"
        )

        # Encode documents in batches
        all_embeddings = []

        num_batches = (len(sentences) + batch_size - 1) // batch_size
        iterator = range(0, len(sentences), batch_size)

        if show_progress_bar and len(sentences) > batch_size:
            iterator = tqdm(
                iterator,
                desc=f"Encoding with late chunking ({self.model})",
                total=num_batches,
                unit="batch"
            )

        for i in iterator:
            batch = sentences[i:i + batch_size]
            embeddings = self._encode_batch_with_retry(batch)
            all_embeddings.extend(embeddings)

        # Convert to numpy array
        result = np.array(all_embeddings, dtype=np.float32)

        # Return single embedding if single input
        if single_input:
            return result[0]

        return result

    def _encode_batch_with_retry(self, documents: List[List[str]]) -> List[np.ndarray]:
        """
        Encode a batch of documents with retry logic using late chunking.

        IMPORTANT: Jina API requires FLAT lists, not nested lists.
        Late chunking is performed internally by Jina at the token level.

        Args:
            documents: List of documents, where each document is a list of chunks
                      Example: [["doc1_chunk1", "doc1_chunk2"], ["doc2_chunk1"]]
                      These will be flattened before sending to Jina API.

        Returns:
            List of embeddings (one per chunk, preserving order)

        Raises:
            Exception: If all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Flatten nested list structure for Jina API
                # Jina expects: ["text1", "text2", "text3"]
                # NOT: [["text1", "text2"], ["text3"]]
                flattened_chunks = []
                for document in documents:
                    flattened_chunks.extend(document)

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }

                data = {
                    "input": flattened_chunks,  # Flat list of strings
                    "model": self.model,
                    "task": "retrieval.passage",  # For document indexing
                    "late_chunking": True,  # Enable late chunking (internal to Jina)
                }

                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()

                result = response.json()

                # Extract embeddings from flat response
                # Result has one entry per chunk in the flattened list
                all_embeddings = []
                for chunk_result in result["data"]:
                    embedding = np.array(chunk_result["embedding"], dtype=np.float32)
                    all_embeddings.append(embedding)

                return all_embeddings

            except Exception as e:
                last_exception = e
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s

                # Try to get detailed error message from API response
                error_detail = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_body = e.response.json()
                        error_detail = f"{e}. API Response: {error_body}"
                    except:
                        try:
                            error_detail = f"{e}. API Response: {e.response.text}"
                        except:
                            pass

                logger.warning(
                    f"Jina API request failed (attempt {attempt + 1}/{self.max_retries}): {error_detail}. "
                    f"Retrying in {wait_time}s..."
                )

                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)

        # All retries failed
        error_msg = f"Failed to encode batch after {self.max_retries} attempts: {last_exception}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def __repr__(self):
        return (
            f"JinaEmbeddingClient(\n"
            f"  model={self.model},\n"
            f"  late_chunking=True,\n"
            f"  task=retrieval.passage\n"
            f")"
        )
