"""Cohere Rerank API integration for improved search precision"""

from typing import List, Tuple, Optional
import logging

from ..config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class CohereReranker:
    """
    Cohere Rerank API integration for document reranking

    Provides state-of-the-art reranking using Cohere's proprietary models
    which are specifically trained for ranking query-document relevance
    """

    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize Cohere reranker with API credentials

        Args:
            model_name: Name of Cohere rerank model (default from config)
            api_key: Cohere API key (default from config)
        """
        self.model_name = model_name or config.cohere_model
        self.api_key = api_key or config.cohere_api_key
        self.client = None  # Lazy load on first use
        logger.info(f"Cohere reranker initialized (model: {self.model_name}, lazy loading)")

    def _load_client(self):
        """Lazy load the Cohere client"""
        if self.client is None:
            try:
                import cohere

                if not self.api_key:
                    logger.warning("Cohere API key not configured. Fallback to local reranking will be used.")
                    return False

                logger.info(f"Initializing Cohere client with model: {self.model_name}")
                self.client = cohere.ClientV2(api_key=self.api_key)
                logger.info(f"Cohere client initialized successfully")
                return True
            except ImportError:
                logger.warning(
                    "cohere package is required for Cohere reranking. "
                    "Install with: pip install cohere. Falling back to local reranking."
                )
                return False
            except Exception as e:
                logger.error(f"Error initializing Cohere client: {e}")
                logger.warning("Falling back to local reranking")
                return False

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
        Rerank documents using Cohere Rerank API

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

        # Try to load Cohere client
        if not self._load_client() or self.client is None:
            logger.warning("Cohere client unavailable, returning original scores")
            # Fallback: return original results sorted by original scores
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            top_indices = sorted_indices[:top_k]
            return (
                [doc_ids[i] for i in top_indices],
                [scores[i] for i in top_indices],
                [metadatas[i] for i in top_indices],
                [documents[i] for i in top_indices],
            )

        logger.info(f"Reranking {len(documents)} documents with Cohere {self.model_name}")

        try:
            # Call Cohere Rerank API
            response = self.client.rerank(
                model=self.model_name,
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
                rank_fields=["text"],
            )

            # Extract reranked indices from response
            reranked_indices = [result.index for result in response.results]
            reranked_scores = [result.relevance_score for result in response.results]

            # Reorder results based on Cohere ranking
            reranked_doc_ids = [doc_ids[i] for i in reranked_indices]
            reranked_metadatas = [metadatas[i] for i in reranked_indices]
            reranked_documents = [documents[i] for i in reranked_indices]

            logger.info(
                f"Reranking complete. Top score: {reranked_scores[0]:.3f}, "
                f"Returned {len(reranked_doc_ids)} results"
            )

            return reranked_doc_ids, reranked_scores, reranked_metadatas, reranked_documents

        except Exception as e:
            logger.error(f"Error during Cohere reranking: {e}")
            logger.warning("Falling back to original scores")
            # Fallback: return original results sorted by original scores
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
        # First, get reranked results from Cohere
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
