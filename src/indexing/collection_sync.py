"""Collection synchronization manager for hybrid local + cloud indexing"""

from typing import Optional, List, Dict, Any, Union, Literal
from datetime import datetime
import logging

from ..config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class SyncFailure:
    """Record of a failed sync attempt"""

    def __init__(self, timestamp: datetime, chunk_ids: List[str], error: str):
        self.timestamp = timestamp
        self.chunk_ids = chunk_ids
        self.error = error


class CollectionSyncManager:
    """
    Manages hybrid local + cloud collection indexing with fail-soft error handling.

    Architecture:
    - LOCAL collection is always primary (never fails)
    - CLOUD collection is optional (sync can be disabled or fail gracefully)
    - Indexing: Write to local first, then sync to cloud if enabled
    - Searching: Route to local or cloud based on source parameter

    Usage:
        manager = CollectionSyncManager(local_collection, cloud_collection_or_none)
        # Index documents (local only, or local + cloud if enabled)
        manager.add(ids, documents, embeddings, metadatas)
        # Search local
        results = manager.query(query_embedding, source='local', top_k=10)
        # Search cloud
        results = manager.query(query_embedding, source='cloud', top_k=10)
    """

    def __init__(
        self,
        local_collection,
        cloud_collection: Optional[object] = None,
        sync_to_cloud: bool = None,
    ):
        """
        Initialize collection sync manager

        Args:
            local_collection: Chroma local collection (PersistentClient)
            cloud_collection: Chroma cloud collection (CloudClient) or None
            sync_to_cloud: Whether to sync additions to cloud (default from config)
        """
        self.local_collection = local_collection
        self.cloud_collection = cloud_collection
        self.sync_to_cloud = (
            sync_to_cloud
            if sync_to_cloud is not None
            else config.sync_to_cloud
        )

        # Sparse vector support (BM25 local, SPLADE cloud)
        self.sparse_vectors_enabled = config.sparse_vector_enabled

        # Track failed syncs for optional manual retry
        self.sync_failures: List[SyncFailure] = []

        logger.info("CollectionSyncManager initialized")
        logger.info(f"  Local collection: {local_collection.name}")
        logger.info(f"  Cloud collection: {cloud_collection.name if cloud_collection else 'disabled'}")
        logger.info(f"  Sync to cloud: {self.sync_to_cloud}")

    def add(
        self,
        ids: List[str],
        documents: Union[List[str], List[Dict]],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        sparse_vectors: Optional[List[Dict[int, float]]] = None) -> bool:
        """
        Add documents to collections (local always, cloud if enabled)

        Args:
            ids: Document IDs
            documents: Document texts or multimodal dicts
            embeddings: Embedding vectors
            metadatas: Metadata dictionaries

            sparse_vectors: Optional sparse vectors (BM25 local or SPLADE cloud)
        Returns:
            True if local indexing succeeded (cloud sync optional)
        """
        try:
            # Step 1: Add to local collection (primary, must succeed)
            logger.debug(f"Adding {len(ids)} documents to local collection...")
            self.local_collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.debug(f"Successfully added {len(ids)} documents to local collection")

            # Step 2: Sync to cloud if enabled (optional, fail-soft)
            if self.sync_to_cloud and self.cloud_collection:
                self._sync_to_cloud(ids, documents, embeddings, metadatas, sparse_vectors=sparse_vectors)

            return True

        except Exception as e:
            logger.error(f"Error adding documents to local collection: {e}", exc_info=True)
            return False

    def _sync_to_cloud(
        self,
        ids: List[str],
        documents: Union[List[str], List[Dict]],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        sparse_vectors: Optional[List[Dict[int, float]]] = None) -> bool:
        """
        Sync documents to cloud collection (fail-soft)

        Args:
            ids: Document IDs
            documents: Document texts or multimodal dicts
            embeddings: Embedding vectors
            metadatas: Metadata dictionaries

            sparse_vectors: Optional sparse vectors (ignored for cloud, cloud uses SPLADE instead)
        Returns:
            True if sync succeeded, False otherwise
        """
        try:
            logger.debug(f"Syncing {len(ids)} documents to cloud collection...")
            # NOTE: Cloud collections don't support sparse_vectors parameter (only local PersistentClient does)
            # Cloud uses SPLADE embeddings instead of BM25
            self.cloud_collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas)
            logger.debug(f"Successfully synced {len(ids)} documents to cloud")
            return True

        except Exception as e:
            # Fail-soft: log error but don't raise
            logger.warning(
                f"Failed to sync {len(ids)} documents to cloud: {e}. "
                f"Documents remain in local collection. Will retry on next index."
            )

            # Record failure for potential manual retry
            failure = SyncFailure(
                timestamp=datetime.now(),
                chunk_ids=ids,
                error=str(e),
            )
            self.sync_failures.append(failure)

            return False

    def query(
        self,
        query_embedding: List[float],
        where: Optional[Dict] = None,
        top_k: int = 10,
        source: Literal["local", "cloud", "default"] = "default",
    ) -> Dict[str, Any]:
        """
        Query collection with source selection

        Args:
            query_embedding: Query embedding vector
            where: Optional filter conditions
            top_k: Number of results to return
            source: 'local' (fast), 'cloud' (remote), or 'default' (use config)

        Returns:
            Query results from selected source
        """
        # Resolve source
        if source == "default":
            source = config.default_search_source

        if source not in ["local", "cloud"]:
            logger.warning(
                f"Invalid search source '{source}', falling back to local"
            )
            source = "local"

        logger.debug(f"Querying {source} collection (top_k={top_k})")

        try:
            if source == "cloud":
                if not self.cloud_collection:
                    logger.warning(
                        "Cloud search requested but cloud collection not available. "
                        "Falling back to local."
                    )
                    return self.local_collection.query(
                        query_embeddings=[query_embedding],
                        where=where,
                        n_results=top_k,
                    )
                return self.cloud_collection.query(
                    query_embeddings=[query_embedding],
                    where=where,
                    n_results=top_k,
                )
            else:
                return self.local_collection.query(
                    query_embeddings=[query_embedding],
                    where=where,
                    n_results=top_k,
                )

        except Exception as e:
            logger.error(f"Error querying {source} collection: {e}", exc_info=True)
            raise

    def _generate_sparse_vectors_for_sync(
        self, documents: Union[List[str], List[Dict]]
    ) -> Optional[List[Dict[int, float]]]:
        """
        Generate sparse vectors for documents during sync (if available)

        Uses SPLADE if available for cloud sync, falls back to None if unavailable.
        Local BM25 sparse vectors are handled automatically by Chroma.

        Args:
            documents: Documents to generate sparse vectors for

        Returns:
            List of sparse vectors or None if generation unavailable
        """
        if not self.sparse_vectors_enabled or not self.cloud_collection:
            return None

        try:
            # Try to get SPLADE client from hybrid search engine
            from .hybrid_search import HybridSearchEngine
            engine = HybridSearchEngine()

            if hasattr(engine, '_generate_sparse_vector'):
                sparse_vecs = []
                for doc in documents:
                    # Handle both string and dict documents
                    text = doc if isinstance(doc, str) else doc.get('text', '')
                    sparse_vec = engine._generate_sparse_vector(text)
                    sparse_vecs.append(sparse_vec if sparse_vec else {})

                if any(sparse_vecs):  # If any non-empty sparse vectors
                    logger.debug(f"Generated sparse vectors for {len(documents)} documents")
                    return sparse_vecs

        except Exception as e:
            logger.debug(f"Could not generate sparse vectors: {e}")

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about both collections"""
        stats = {
            "local": {"count": self.local_collection.count()},
            "cloud": {"count": self.cloud_collection.count() if self.cloud_collection else None},
            "sync_enabled": self.sync_to_cloud,
            "sync_failures": len(self.sync_failures),
            "failed_chunks": sum(len(f.chunk_ids) for f in self.sync_failures),
        }

        if self.sync_failures:
            stats["last_failure"] = {
                "timestamp": self.sync_failures[-1].timestamp.isoformat(),
                "error": self.sync_failures[-1].error,
            }

        return stats

    def retry_failed_syncs(self) -> int:
        """
        Retry previously failed cloud syncs

        Returns:
            Number of successfully retried chunks
        """
        if not self.sync_failures or not self.cloud_collection:
            return 0

        logger.info(f"Retrying {len(self.sync_failures)} failed sync batches...")
        retried_count = 0

        for failure in self.sync_failures[:]:  # Iterate over copy
            try:
                chunk_ids = failure.chunk_ids
                # Query local collection to get documents and embeddings
                results = self.local_collection.get(ids=chunk_ids)

                if results and "ids" in results:
                    # NOTE: Don't pass sparse_vectors to cloud - only local PersistentClient supports it
                    self.cloud_collection.add(
                        ids=results["ids"],
                        documents=results.get("documents"),
                        embeddings=results.get("embeddings"),
                        metadatas=results.get("metadatas"))
                    retried_count += len(chunk_ids)
                    self.sync_failures.remove(failure)
                    logger.info(f"Successfully retried {len(chunk_ids)} chunks")

            except Exception as e:
                logger.error(f"Failed to retry sync for batch: {e}")

        return retried_count

    def clear_sync_failure_log(self) -> None:
        """Clear the sync failure log"""
        count = len(self.sync_failures)
        self.sync_failures.clear()
        logger.info(f"Cleared {count} sync failure records")
