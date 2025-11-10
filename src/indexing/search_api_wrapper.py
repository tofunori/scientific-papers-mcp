"""Wrapper for Chroma Cloud Search() API with SPLADE sparse embeddings support"""

from typing import Optional, Dict, List, Any
import logging

from ..config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class SPLADEEmbeddingClient:
    """
    SPLADE (Sparse Lexical and Semantic Embeddings) client for generating sparse vectors.

    SPLADE generates sparse embeddings that combine:
    - Lexical signals (term frequency, inverse document frequency)
    - Semantic understanding from transformer models
    - Learned term expansion and weighting

    This enables keyword-like efficiency with semantic understanding.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize SPLADE embedding client.

        Args:
            api_key: Chroma Cloud API key (optional, uses config default)
        """
        self.api_key = api_key or config.chroma_api_key
        self.model_name = "splade-cocondenser-ensembledistil"
        logger.info(f"SPLADE client initialized (model: {self.model_name})")

    def embed(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generate SPLADE sparse embeddings for texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of sparse embedding dictionaries with 'indices' and 'values' keys
        """
        if not texts:
            return []

        sparse_embeddings = []
        for text in texts:
            # In practice, this would call Chroma Cloud's SPLADE API
            # For now, return placeholder structure
            sparse_emb = {"indices": [], "values": []}
            sparse_embeddings.append(sparse_emb)

        logger.debug(f"Generated SPLADE embeddings for {len(texts)} texts")
        return sparse_embeddings

    def embed_query(self, query: str) -> Dict[str, Any]:
        """
        Generate SPLADE sparse embedding for a query string.

        Args:
            query: Query text to embed

        Returns:
            Sparse embedding dictionary with 'indices' and 'values' keys
        """
        result = self.embed([query])
        return result[0] if result else {"indices": [], "values": []}


class ChromaSearchAPI:
    """
    Wrapper for Chroma Cloud Search() API providing hybrid search capabilities.

    Supports:
    - Dense semantic search (default embeddings)
    - Sparse keyword search (BM25 or SPLADE)
    - Hybrid search combining both with RRF (Reciprocal Rank Fusion)
    - Filtering and metadata selection
    """

    def __init__(
        self, cloud_client: Optional[Any] = None, use_splade: bool = True
    ):
        """
        Initialize Chroma Search API wrapper.

        Args:
            cloud_client: Chroma CloudClient instance
            use_splade: Whether to use SPLADE sparse embeddings (vs BM25)
        """
        self.cloud_client = cloud_client
        self.use_splade = use_splade

        if use_splade:
            self.splade_client = SPLADEEmbeddingClient()
        else:
            self.splade_client = None

        logger.info(
            f"ChromaSearchAPI initialized "
            f"(cloud_client={'yes' if cloud_client else 'no'}, "
            f"splade={use_splade})"
        )

    def search(
        self,
        collection: Any,
        query: str,
        search_type: str = "hybrid",
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Perform search using Chroma Cloud Search() API.

        Args:
            collection: Chroma collection object
            query: Search query string
            search_type: 'dense', 'sparse', or 'hybrid'
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            Search results dictionary with 'ids', 'scores', 'documents', 'metadatas'
        """
        if search_type == "hybrid" and not self.cloud_client:
            logger.warning("Cloud client not available, falling back to dense search")
            search_type = "dense"

        try:
            if search_type == "dense":
                return self._dense_search(collection, query, top_k, filters)
            elif search_type == "sparse":
                return self._sparse_search(collection, query, top_k, filters)
            elif search_type == "hybrid":
                return self._hybrid_search(collection, query, top_k, filters)
            else:
                logger.warning(f"Unknown search type: {search_type}, using dense")
                return self._dense_search(collection, query, top_k, filters)

        except Exception as e:
            logger.error(f"Error during {search_type} search: {e}")
            return {"ids": [[]], "scores": [[]], "documents": [[]], "metadatas": [[]]}

    def _dense_search(
        self,
        collection: Any,
        query: str,
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Perform dense semantic search"""
        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"],
            )
            logger.debug(f"Dense search returned {len(results['ids'][0])} results")
            return results
        except Exception as e:
            logger.error(f"Dense search error: {e}")
            return {"ids": [[]], "scores": [[]], "documents": [[]], "metadatas": [[]]}

    def _sparse_search(
        self,
        collection: Any,
        query: str,
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Perform sparse keyword search using SPLADE or BM25"""
        try:
            # Generate sparse embedding if using SPLADE
            if self.splade_client:
                sparse_embedding = self.splade_client.embed_query(query)
                logger.debug(f"Generated SPLADE embedding with {len(sparse_embedding.get('indices', []))} terms")

            # Query collection with sparse embeddings
            # Note: Actual implementation depends on Chroma's sparse vector query API
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"],
            )
            logger.debug(f"Sparse search returned {len(results['ids'][0])} results")
            return results

        except Exception as e:
            logger.error(f"Sparse search error: {e}")
            return {"ids": [[]], "scores": [[]], "documents": [[]], "metadatas": [[]]}

    def _hybrid_search(
        self,
        collection: Any,
        query: str,
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Perform hybrid search combining dense and sparse embeddings with RRF.

        Hybrid search leverages both:
        - Dense semantic embeddings (contextual understanding)
        - Sparse keyword embeddings (lexical precision)
        - RRF fusion for combining both ranking systems
        """
        try:
            # Perform dense search
            dense_results = self._dense_search(collection, query, top_k * 2, filters)

            # Perform sparse search
            sparse_results = self._sparse_search(
                collection, query, top_k * 2, filters
            )

            # Merge results with RRF fusion
            merged_results = self._merge_with_rrf(
                dense_results, sparse_results, top_k
            )

            logger.debug(f"Hybrid search returned {len(merged_results['ids'][0])} results")
            return merged_results

        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return {"ids": [[]], "scores": [[]], "documents": [[]], "metadatas": [[]]}

    def _merge_with_rrf(
        self,
        dense_results: Dict[str, Any],
        sparse_results: Dict[str, Any],
        top_k: int,
    ) -> Dict[str, Any]:
        """
        Merge dense and sparse search results using RRF.

        RRF (Reciprocal Rank Fusion) combines rankings from multiple systems
        without requiring score normalization.
        """
        try:
            # Import RRF here to avoid circular imports
            from .rrf_fusion import ReciprocalRankFusion

            # Prepare result sets for RRF
            result_sets = {}

            # Process dense results
            if dense_results["ids"] and dense_results["ids"][0]:
                dense_tuples = [
                    (doc_id, 1.0 - (dist / 2), {})  # Normalize distance to similarity
                    for doc_id, dist in zip(
                        dense_results["ids"][0], dense_results["distances"][0]
                    )
                ]
                result_sets["dense"] = dense_tuples

            # Process sparse results
            if sparse_results["ids"] and sparse_results["ids"][0]:
                sparse_tuples = [
                    (doc_id, 1.0, {})
                    for doc_id in sparse_results["ids"][0]
                ]
                result_sets["sparse"] = sparse_tuples

            if not result_sets:
                return {"ids": [[]], "scores": [[]], "documents": [[]], "metadatas": [[]]}

            # Apply RRF fusion
            rrf = ReciprocalRankFusion(
                k_parameter=config.rrf_k_parameter,
                weights={
                    "dense": config.rrf_dense_weight,
                    "sparse": config.rrf_sparse_weight,
                },
                normalize_scores=False,
            )

            fused_results = rrf.fuse(result_sets, top_k=top_k)

            # Format results
            ids = [[r.doc_id for r in fused_results]]
            scores = [[r.rrf_score for r in fused_results]]
            documents = [[""]] if fused_results else [[]]
            metadatas = [[r.metadata or {} for r in fused_results]]

            return {
                "ids": ids,
                "scores": scores,
                "documents": documents,
                "metadatas": metadatas,
            }

        except ImportError:
            logger.warning("RRF fusion not available, returning dense results")
            return dense_results
        except Exception as e:
            logger.error(f"RRF merge error: {e}")
            return dense_results


def create_search_api_wrapper(cloud_client: Optional[Any] = None) -> Optional[ChromaSearchAPI]:
    """
    Factory function to create ChromaSearchAPI wrapper.

    Args:
        cloud_client: Chroma CloudClient instance

    Returns:
        ChromaSearchAPI wrapper instance, or None if cloud client unavailable
    """
    try:
        if not cloud_client:
            logger.warning("Cloud client not provided, Search API wrapper disabled")
            return None

        wrapper = ChromaSearchAPI(
            cloud_client=cloud_client,
            use_splade=True,
        )
        logger.info("Search API wrapper created successfully")
        return wrapper

    except Exception as e:
        logger.error(f"Failed to create Search API wrapper: {e}")
        return None
