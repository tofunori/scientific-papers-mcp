"""Jina AI Embeddings v4 API Client

This module provides a wrapper around Jina AI's embeddings API that mimics
the SentenceTransformer interface for easy drop-in replacement.
"""

import logging
import time
from typing import List, Union
import numpy as np
import requests
from tqdm import tqdm

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class JinaV4EmbeddingClient:
    """
    API client for Jina Embeddings v4 that mimics SentenceTransformer interface.

    Features:
    - Batch processing with automatic splitting
    - Progress bars
    - Retry logic with exponential backoff
    - Compatible with existing SentenceTransformer-based code
    """

    def __init__(
        self,
        api_key: str,
        model: str = "jina-embeddings-v4",
        api_url: str = "https://api.jina.ai/v1/embeddings",
        max_retries: int = 3,
        timeout: int = 60,
    ):
        """
        Initialize Jina v4 embedding client.

        Args:
            api_key: Jina AI API key
            model: Model name (default: jina-embeddings-v4)
            api_url: API endpoint URL
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.max_retries = max_retries
        self.timeout = timeout

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

        logger.info(f"Jina v4 API client initialized (model: {self.model})")

    def encode(
        self,
        sentences: Union[str, List[str], List[dict]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Encode text and/or images using Jina v4 API (multimodal support).

        Compatible with SentenceTransformer.encode() interface.

        Args:
            sentences: Single sentence (str), list of sentences, or list of dicts for multimodal:
                - Text only: "text content" or ["text1", "text2"]
                - Multimodal: [{"text": "...", "image": "data:image/jpeg;base64,..."}]
                - Image only: [{"image": "data:image/jpeg;base64,..."}]
            batch_size: Number of inputs to encode per API call (max 1024)
            show_progress_bar: Whether to show progress bar
            convert_to_tensor: Ignored (kept for compatibility)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            numpy.ndarray: Array of embeddings with shape (n_inputs, embedding_dim)
        """
        # Handle single string input
        if isinstance(sentences, str):
            sentences = [sentences]
            single_input = True
        else:
            single_input = False

        # Jina API supports up to 1024 inputs per request
        batch_size = min(batch_size, 1024)

        all_embeddings = []

        # Process in batches
        num_batches = (len(sentences) + batch_size - 1) // batch_size

        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar and len(sentences) > batch_size:
            iterator = tqdm(
                iterator,
                desc="Encoding batches",
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

    def _encode_batch_with_retry(self, inputs: Union[List[str], List[dict]]) -> List[List[float]]:
        """
        Encode a batch of texts/images with retry logic.

        Args:
            inputs: List of texts (str) or multimodal inputs (dict)

        Returns:
            List of embeddings

        Raises:
            Exception: If all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return self._encode_batch(inputs)

            except requests.exceptions.RequestException as e:
                last_exception = e
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s

                logger.warning(
                    f"API request failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )

                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)

        # All retries failed
        error_msg = f"Failed to encode batch after {self.max_retries} attempts: {last_exception}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def _encode_batch(self, inputs: Union[List[str], List[dict]]) -> List[List[float]]:
        """
        Encode a single batch of texts/images via Jina API.

        Args:
            inputs: List of texts (str) or multimodal inputs (dict)
                Examples:
                - Text: ["text1", "text2"]
                - Multimodal: [{"text": "...", "image": "data:image/jpeg;base64,..."}]
                - Image only: [{"image": "data:image/jpeg;base64,..."}]

        Returns:
            List of embeddings

        Raises:
            requests.exceptions.RequestException: If API request fails
        """
        # The Jina API accepts both strings and dicts in the "input" field
        # Strings are automatically treated as text
        # Dicts must have "text" and/or "image" keys
        payload = {
            "model": self.model,
            "input": inputs,
            "encoding_type": "float",
            "task": "retrieval.passage",  # For indexing documents
        }

        response = self.session.post(
            self.api_url,
            json=payload,
            timeout=self.timeout
        )

        # Raise exception for HTTP errors
        response.raise_for_status()

        # Parse response
        data = response.json()

        if "data" not in data:
            raise Exception(f"Unexpected API response format: {data}")

        # Extract embeddings from response
        embeddings = [item["embedding"] for item in data["data"]]

        return embeddings

    def __del__(self):
        """Clean up session on deletion"""
        if hasattr(self, 'session'):
            self.session.close()
