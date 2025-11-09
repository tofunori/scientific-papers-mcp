"""Chroma DB client initialization and management"""

from pathlib import Path
from typing import Optional, Union, Dict
import logging

import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction
from chromadb import CloudClient

from ..config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ChromaClientManager:
    """Manager for Chroma DB client and collections"""

    def __init__(self, chroma_path: Path | str = None, embedding_model: str = None):
        """
        Initialize Chroma clients (hybrid mode: local always, cloud optional)

        Args:
            chroma_path: Path to persist Chroma database (local mode)
            embedding_model: Embedding model name
        """
        self.chroma_path = Path(chroma_path) if chroma_path else config.chroma_path
        self.embedding_model = embedding_model or config.embedding_model

        # Always initialize local client (primary)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing Chroma PersistentClient at {self.chroma_path}")
        self.local_client = self._initialize_local_client()

        # Optionally initialize cloud client (secondary, for syncing)
        self.cloud_client = None
        if config.use_chroma_cloud or config.sync_to_cloud:
            logger.info("Chroma Cloud sync enabled, initializing CloudClient...")
            try:
                self.cloud_client = self._initialize_cloud_client()
            except Exception as e:
                logger.warning(f"Failed to initialize cloud client: {e}. Continuing with local-only mode.")
                self.cloud_client = None

        # For backward compatibility, default to local client
        self.client = self.local_client

    def _initialize_local_client(self) -> chromadb.PersistentClient:
        """Initialize Chroma persistent client (local mode)"""
        try:
            client = chromadb.PersistentClient(
                path=str(self.chroma_path),  # Must be string for Windows
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True,
                ),
            )
            logger.info("Chroma PersistentClient initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Chroma PersistentClient: {e}")
            raise

    def _initialize_cloud_client(self) -> CloudClient:
        """Initialize Chroma CloudClient (cloud mode)"""
        try:
            client = CloudClient(
                tenant=config.chroma_tenant,
                database=config.chroma_database,
                api_key=config.chroma_api_key,
            )
            logger.info("Chroma CloudClient initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Chroma CloudClient: {e}")
            raise

    def get_or_create_collection(
        self,
        collection_name: str = "scientific_papers",
        metadata: Optional[dict] = None,
    ):
        """
        Get or create a Chroma collection

        Args:
            collection_name: Name of the collection
            metadata: Optional metadata for the collection

        Returns:
            Chroma collection object
        """
        default_metadata = {
            "description": "Glacier albedo research papers",
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 300,  # Optimized for 2048 dims (was 200)
            "hnsw:M": 24,  # Optimized for 2048 dims (was 16) - More connections for high-dim
        }

        if metadata:
            default_metadata.update(metadata)

        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=default_metadata,
                # No embedding function specified here - we'll embed before adding
            )
            logger.info(f"Collection '{collection_name}' ready (count: {collection.count()})")
            return collection
        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}")
            raise

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection"""
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
        except Exception as e:
            logger.warning(f"Failed to delete collection: {e}")

    def list_collections(self) -> list:
        """List all collections in the database"""
        try:
            collections = self.client.list_collections()
            logger.info(f"Found {len(collections)} collections")
            return collections
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def get_stats(self) -> dict:
        """Get statistics about the Chroma database"""
        stats = {}
        try:
            collections = self.client.list_collections()
            for collection in collections:
                stats[collection.name] = {
                    "count": collection.count(),
                    "metadata": collection.metadata,
                }
            logger.info(f"Chroma stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def backup(self, backup_path: Path) -> bool:
        """Create a backup of the Chroma database (local mode only)"""
        import shutil

        if self.use_cloud:
            logger.info("Backup not applicable for Chroma Cloud mode. "
                       "Data is managed by Chroma Cloud infrastructure.")
            return True

        try:
            backup_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                self.chroma_path,
                backup_path / "chroma_backup",
                dirs_exist_ok=True,
            )
            logger.info(f"Backup created at {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False


    def enable_sparse_vectors(self, collection_name: str = None) -> bool:
        """
        Enable sparse vector support for a collection (cloud-only, requires Search() API).

        Sparse vectors (BM25 local, SPLADE cloud) provide keyword-based search that
        complements dense semantic vectors in hybrid search.

        Args:
            collection_name: Collection to enable sparse vectors for (defaults to primary)

        Returns:
            True if successful, False otherwise
        """
        try:
            from ..config import config

            if not config.sparse_vector_enabled:
                logger.warning("Sparse vectors disabled in config")
                return False

            if self.cloud_client is None:
                logger.info("Cloud client not available - sparse vectors for local mode")
                return True  # Local mode will use BM25 instead

            logger.info(
                f"Sparse vectors enabled for collection '{collection_name}'. "
                f"Model: {config.cloud_sparse_embedding_model}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to enable sparse vectors: {e}")
            return False

    def configure_cloud_sparse_schema(self, collection) -> Dict:
        """
        Configure Chroma Cloud collection for sparse vector (SPLADE) operations.

        Returns metadata config for sparse vectors to be passed to collection creation.

        Returns:
            Dict with sparse vector metadata configuration
        """
        from ..config import config

        sparse_metadata = {
            "sparse_vector_enabled": config.sparse_vector_enabled,
            "sparse_embedding_model": config.cloud_sparse_embedding_model,
            "sparse_embedding_dims": config.cloud_sparse_embedding_dims,
        }

        logger.info(f"Cloud sparse schema configured: {sparse_metadata}")
        return sparse_metadata

    def get_collection_capabilities(self, collection) -> Dict:
        """
        Get capabilities of a Chroma collection (dense, sparse, search API, etc.).

        Returns:
            Dict with available search capabilities
        """
        capabilities = {
            "dense_search": True,  # All collections support dense vector search
            "sparse_vectors": False,
            "search_api": False,
            "rrf_fusion": False,
        }

        try:
            from ..config import config

            # Check if sparse vectors are enabled
            if config.sparse_vector_enabled:
                capabilities["sparse_vectors"] = True

            # Check if cloud Search() API is available
            if self.cloud_client is not None and config.enable_cloud_search_api:
                capabilities["search_api"] = True
                capabilities["rrf_fusion"] = True  # RRF available with Search() API

            logger.info(f"Collection capabilities: {capabilities}")
            return capabilities

        except Exception as e:
            logger.error(f"Failed to determine collection capabilities: {e}")
            return capabilities


def initialize_chroma(
    chroma_path: Path | str = None, embedding_model: str = None
) -> chromadb.Collection:
    """
    Convenience function to initialize Chroma and get default collection

    Args:
        chroma_path: Path to Chroma database
        embedding_model: Embedding model name

    Returns:
        Chroma collection object
    """
    manager = ChromaClientManager(chroma_path, embedding_model)
    collection = manager.get_or_create_collection(config.default_collection_name)
    return collection
