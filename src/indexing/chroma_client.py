"""Chroma DB client initialization and management"""

from pathlib import Path
from typing import Optional
import logging

import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction

from ..config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ChromaClientManager:
    """Manager for Chroma DB client and collections"""

    def __init__(self, chroma_path: Path | str = None, embedding_model: str = None):
        """
        Initialize Chroma client

        Args:
            chroma_path: Path to persist Chroma database
            embedding_model: Embedding model name
        """
        self.chroma_path = Path(chroma_path) if chroma_path else config.chroma_path
        self.embedding_model = embedding_model or config.embedding_model

        # Create path if needed
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing Chroma client at {self.chroma_path}")

        # Initialize Chroma client with persistence
        self.client = self._initialize_client()

    def _initialize_client(self) -> chromadb.PersistentClient:
        """Initialize Chroma persistent client"""
        try:
            client = chromadb.PersistentClient(
                path=str(self.chroma_path),  # Must be string for Windows
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True,
                ),
            )
            logger.info("Chroma client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Chroma client: {e}")
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
        """Create a backup of the Chroma database"""
        import shutil

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
    collection = manager.get_or_create_collection("scientific_papers")
    return collection
