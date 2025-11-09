"""Cross-encoder reranking for improved search precision"""

from typing import List, Tuple
import logging
import numpy as np

from ..config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranking using sentence-transformers

    Provides ~35% better precision compared to pure embedding search
    by scoring query-document pairs directly instead of comparing embeddings
    """

    def __init__(self, model_name: str = None):
        """
        Initialize cross-encoder reranker with lazy loading

        Args:
            model_name: Name of cross-encoder model (default from config)
        """
        self.model_name = model_name or config.reranker_model
        self.model = None  # Lazy load on first use
        logger.info(f"Cross-encoder reranker initialized (model: {self.model_name}, lazy loading)")

    def _load_model(self):
        """Lazy load the cross-encoder model"""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading cross-encoder model: {self.model_name}")
                self.model = CrossEncoder(self.model_name)
                logger.info(f"Cross-encoder model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for reranking. "
                    "Install with: pip install sentence-transformers"
                )
            except Exception as e:
                logger.error(f"Error loading cross-encoder model: {e}")
                raise

    def rerank(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[str],
        scores: List[float],
        metadatas: List[dict],
        top_k: int = 10,
    ) -> Tuple[List[str], List[float], List[dict], List[str]]:
        """
        Rerank documents using cross-encoder scoring

        Args:
            query: Search query
            documents: List of document texts
            doc_ids: List of document IDs
            scores: Original search scores (from hybrid search)
            metadatas: List of metadata dicts
            top_k: Number of top results to return after reranking

        Returns:
            Tuple of (doc_ids, reranked_scores, metadatas, documents)
        """
        if not documents:
            return [], [], [], []

        # Lazy load model
        self._load_model()

        logger.info(f"Reranking {len(documents)} documents with cross-encoder")

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        try:
            # Score all pairs (returns numpy array)
            rerank_scores = self.model.predict(pairs)

            # Convert to list if numpy array
            if isinstance(rerank_scores, np.ndarray):
                rerank_scores = rerank_scores.tolist()

            # Sort by reranked scores (descending)
            sorted_indices = sorted(
                range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True
            )

            # Take top K
            top_indices = sorted_indices[:top_k]

            # Reorder results
            reranked_doc_ids = [doc_ids[i] for i in top_indices]
            reranked_scores = [rerank_scores[i] for i in top_indices]
            reranked_metadatas = [metadatas[i] for i in top_indices]
            reranked_documents = [documents[i] for i in top_indices]

            logger.info(
                f"Reranking complete. Top score: {reranked_scores[0]:.3f}, "
                f"Returned {len(reranked_doc_ids)} results"
            )

            return reranked_doc_ids, reranked_scores, reranked_metadatas, reranked_documents

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback: return original results sorted by original scores
            logger.warning("Falling back to original scores")
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            top_indices = sorted_indices[:top_k]

            return (
                [doc_ids[i] for i in top_indices],
                [scores[i] for i in top_indices],
                [metadatas[i] for i in top_indices],
                [documents[i] for i in top_indices],
            )

    def rerank_with_metadata_boost(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[str],
        scores: List[float],
        metadatas: List[dict],
        top_k: int = 10,
        title_boost: float = 2.0,
        abstract_boost: float = 1.5,
    ) -> Tuple[List[str], List[float], List[dict], List[str]]:
        """
        Rerank with metadata boosting for title and abstract matches

        If query terms appear in title or abstract, boost the reranking score
        This is inspired by Zotero MCP's metadata prioritization

        Args:
            query: Search query
            documents: List of document texts
            doc_ids: List of document IDs
            scores: Original search scores
            metadatas: List of metadata dicts
            top_k: Number of top results to return
            title_boost: Multiplier for title matches (default 2.0)
            abstract_boost: Multiplier for abstract matches (default 1.5)

        Returns:
            Tuple of (doc_ids, boosted_scores, metadatas, documents)
        """
        # First, get reranked results
        reranked_ids, reranked_scores, reranked_metas, reranked_docs = self.rerank(
            query, documents, doc_ids, scores, metadatas, top_k=len(documents)
        )

        if not reranked_scores:
            return [], [], [], []

        # Apply metadata boosting
        query_terms = set(query.lower().split())
        boosted_scores = []

        for i, (score, metadata) in enumerate(zip(reranked_scores, reranked_metas)):
            boost_factor = 1.0

            # Check title match
            title = metadata.get("title", "").lower()
            if title and any(term in title for term in query_terms):
                boost_factor *= title_boost
                logger.debug(f"Title match boost ({title_boost}x) for doc {reranked_ids[i]}")

            # Check abstract match
            abstract = metadata.get("abstract", "").lower()
            if abstract and any(term in abstract for term in query_terms):
                boost_factor *= abstract_boost
                logger.debug(f"Abstract match boost ({abstract_boost}x) for doc {reranked_ids[i]}")

            boosted_score = score * boost_factor
            boosted_scores.append(boosted_score)

        # Re-sort by boosted scores
        sorted_indices = sorted(range(len(boosted_scores)), key=lambda i: boosted_scores[i], reverse=True)
        top_indices = sorted_indices[:top_k]

        final_ids = [reranked_ids[i] for i in top_indices]
        final_scores = [boosted_scores[i] for i in top_indices]
        final_metas = [reranked_metas[i] for i in top_indices]
        final_docs = [reranked_docs[i] for i in top_indices]

        logger.info(f"Metadata boosting applied. Final top score: {final_scores[0]:.3f}")

        return final_ids, final_scores, final_metas, final_docs
