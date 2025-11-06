"""Hybrid search engine combining semantic and keyword-based search"""

from typing import List, Dict, Tuple, Optional
import logging
import numpy as np

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from ..config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class HybridSearchEngine:
    """
    Combines dense vector search (semantic) with sparse BM25 search (keyword)
    for high-quality information retrieval
    """

    def __init__(self, chroma_collection, embedding_model: str = None):
        """
        Initialize hybrid search engine

        Args:
            chroma_collection: Chroma collection object
            embedding_model: Name of embedding model to use
        """
        self.collection = chroma_collection
        self.embedding_model_name = embedding_model or config.embedding_model

        # Load embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # BM25 index (will be built during indexing)
        self.bm25_index = None
        self.documents = []  # Tokenized documents for BM25
        self.doc_ids = []  # Corresponding document IDs
        self.doc_texts = {}  # Full text of documents

        # Load existing data from Chroma
        self._load_from_chroma()

        logger.info("Hybrid search engine initialized")

    def _load_from_chroma(self) -> None:
        """Load existing documents from Chroma into the BM25 index"""
        try:
            logger.info("Chargeant les documents existants de Chroma vers l'index BM25...")

            # Get all documents from Chroma
            all_data = self.collection.get(include=["documents", "metadatas"])

            if not all_data.get("ids"):
                logger.info("Aucun document trouvé dans Chroma")
                return

            # Load each document into BM25 index
            for doc_id, document, metadata in zip(
                all_data["ids"],
                all_data.get("documents", []),
                all_data.get("metadatas", [])
            ):
                if document and document.strip():
                    # Tokenize for BM25
                    tokenized = document.lower().split()
                    self.documents.append(tokenized)
                    self.doc_ids.append(doc_id)
                    self.doc_texts[doc_id] = document

            # Rebuild BM25 index
            if self.documents:
                self.bm25_index = BM25Okapi(self.documents)
                logger.info(f"Chargé {len(self.doc_ids)} documents dans l'index BM25")
            else:
                logger.info("Aucun document à indexer en BM25")

        except Exception as e:
            logger.warning(f"Erreur lors du chargement de Chroma: {e}")

    def index_document(
        self, doc_id: str, text: str, metadata: dict = None
    ) -> None:
        """
        Index a document in both Chroma (dense) and BM25 (sparse) indices

        Args:
            doc_id: Unique document identifier
            text: Full text of the document
            metadata: Optional metadata dictionary
        """
        if not text or not text.strip():
            logger.warning(f"Skipping empty document: {doc_id}")
            return

        try:
            # 1. Dense embedding for Chroma
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)

            self.collection.add(
                ids=[doc_id],
                documents=[text],
                embeddings=[embedding.tolist()],
                metadatas=[metadata] if metadata else None,
            )

            # 2. Sparse indexing for BM25
            tokenized = text.lower().split()
            self.documents.append(tokenized)
            self.doc_ids.append(doc_id)
            self.doc_texts[doc_id] = text

            # Rebuild BM25 index
            if self.documents:
                self.bm25_index = BM25Okapi(self.documents)

            logger.debug(f"Indexed document: {doc_id}")

        except Exception as e:
            logger.error(f"Error indexing document {doc_id}: {e}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        where_filter: Optional[Dict] = None,
    ) -> Tuple[List[str], List[float], List[Dict], List[str]]:
        """
        Perform hybrid search combining semantic and keyword matching

        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Balance between semantic (1.0) and keyword (0.0) search
                  0.0 = pure keyword, 1.0 = pure semantic, 0.5 = balanced
            where_filter: Optional metadata filter for Chroma

        Returns:
            Tuple of (document_ids, scores, metadatas, documents)
        """
        if not query.strip():
            return [], [], [], []

        logger.info(f"Search query: '{query}' (alpha={alpha}, top_k={top_k})")

        try:
            # Step 1: Dense semantic search
            dense_results = self._semantic_search(
                query, top_k=top_k * 2, where_filter=where_filter
            )

            # Step 2: Sparse keyword search (BM25)
            sparse_results = self._keyword_search(query, top_k=top_k * 2)

            # Step 3: Combine and rank results
            final_ids, final_scores, final_metadatas, final_documents = self._combine_results(
                dense_results, sparse_results, alpha, top_k
            )

            logger.info(f"Found {len(final_ids)} results")
            return final_ids, final_scores, final_metadatas, final_documents

        except Exception as e:
            logger.error(f"Error during search: {e}")
            return [], [], [], []

    def _semantic_search(
        self, query: str, top_k: int = 20, where_filter: Optional[Dict] = None
    ) -> Dict:
        """
        Semantic search using dense vectors

        Args:
            query: Search query
            top_k: Number of results
            where_filter: Optional metadata filter

        Returns:
            Results dictionary from Chroma
        """
        try:
            # Encode query
            query_embedding = self.embedding_model.encode(
                query, convert_to_tensor=False
            )

            # Query Chroma
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            return results
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

    def _keyword_search(self, query: str, top_k: int = 20) -> Dict:
        """
        Keyword search using BM25

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Results dictionary compatible with semantic search format
        """
        if not self.bm25_index or not self.doc_ids:
            return {"ids": [[]], "scores": []}

        try:
            # Tokenize query same as documents
            tokenized_query = query.lower().split()

            # Get BM25 scores
            bm25_scores = self.bm25_index.get_scores(tokenized_query)

            # Get top-k
            top_indices = np.argsort(bm25_scores)[::-1][:top_k]

            result_ids = [[self.doc_ids[i] for i in top_indices]]
            result_scores = [[float(bm25_scores[i]) for i in top_indices]]

            return {"ids": result_ids, "scores": result_scores}
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return {"ids": [[]], "scores": []}

    def _combine_results(
        self,
        dense_results: Dict,
        sparse_results: Dict,
        alpha: float,
        top_k: int,
    ) -> Tuple[List[str], List[float], List[Dict], List[str]]:
        """
        Combine semantic and keyword search results

        Args:
            dense_results: Results from semantic search (with distances and documents)
            sparse_results: Results from keyword search (with scores)
            alpha: Weight for semantic (1-alpha for keyword)
            top_k: Final number of results

        Returns:
            Tuple of (doc_ids, combined_scores, metadatas, documents)
        """
        combined_scores = {}

        # Add dense search scores (convert distance to similarity)
        if dense_results["ids"] and dense_results["ids"][0]:
            dense_ids = dense_results["ids"][0]
            dense_distances = dense_results["distances"][0]
            dense_metadatas = dense_results["metadatas"][0] if dense_results.get("metadatas") else [None] * len(dense_ids)
            dense_documents = dense_results["documents"][0] if dense_results.get("documents") else [None] * len(dense_ids)

            # Convert distances to similarities (closer = higher score)
            for doc_id, distance, metadata, document in zip(dense_ids, dense_distances, dense_metadatas, dense_documents):
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                combined_scores[doc_id] = alpha * similarity
                if not hasattr(self, "_doc_metadatas"):
                    self._doc_metadatas = {}
                if not hasattr(self, "_doc_documents"):
                    self._doc_documents = {}
                self._doc_metadatas[doc_id] = metadata
                self._doc_documents[doc_id] = document

        # Add sparse search scores
        if sparse_results["ids"] and sparse_results["ids"][0]:
            sparse_ids = sparse_results["ids"][0]
            sparse_scores = sparse_results["scores"][0]

            # Normalize sparse scores
            if sparse_scores:
                max_sparse = max(sparse_scores)
                min_sparse = min(sparse_scores)
                sparse_range = max_sparse - min_sparse if max_sparse > min_sparse else 1

                for doc_id, score in zip(sparse_ids, sparse_scores):
                    normalized_score = (score - min_sparse) / sparse_range if sparse_range > 0 else 0

                    if doc_id in combined_scores:
                        combined_scores[doc_id] += (1 - alpha) * normalized_score
                    else:
                        combined_scores[doc_id] = (1 - alpha) * normalized_score

        # Sort by combined score
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        final_ids = [doc_id for doc_id, _ in ranked]
        final_scores = [score for _, score in ranked]

        # Get metadatas and documents
        final_metadatas = []
        final_documents = []
        if hasattr(self, "_doc_metadatas"):
            final_metadatas = [
                self._doc_metadatas.get(doc_id, {}) for doc_id in final_ids
            ]
        if hasattr(self, "_doc_documents"):
            final_documents = [
                self._doc_documents.get(doc_id, "") for doc_id in final_ids
            ]

        return final_ids, final_scores, final_metadatas, final_documents

    def get_document_text(self, doc_id: str) -> str:
        """Get full text of a document"""
        return self.doc_texts.get(doc_id, "")

    def get_metadata(self, doc_id: str) -> Dict:
        """Get metadata for a document"""
        try:
            result = self.collection.get(ids=[doc_id], include=["metadatas"])
            if result["metadatas"]:
                return result["metadatas"][0]
        except:
            pass
        return {}
