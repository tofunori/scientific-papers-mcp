"""Reciprocal Rank Fusion (RRF) implementation for combining multiple ranking systems"""

from typing import List, Dict, Tuple, Optional, NamedTuple
import logging
from collections import defaultdict

from ..utils.logger import setup_logger
from ..config import config

logger = setup_logger(__name__)


class RRFScorer(NamedTuple):
    """Result from RRF fusion containing document ID and RRF score"""

    doc_id: str
    rrf_score: float
    metadata: Optional[Dict] = None


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) algorithm for combining multiple ranking systems.

    RRF formula: score = sum(weight_i / (k + rank_i)) for each ranking system
    where:
    - weight_i: weight for ranking system i
    - k: smoothing parameter (prevents rank inflation, typically 60)
    - rank_i: rank position in system i (1-indexed)

    RRF is effective at combining dense semantic and sparse keyword rankings without
    requiring normalized scores, making it robust across different ranking systems.
    """

    def __init__(
        self,
        k_parameter: int = 60,
        weights: Optional[Dict[str, float]] = None,
        normalize_scores: bool = False,
    ):
        """
        Initialize RRF fusion engine.

        Args:
            k_parameter: Smoothing parameter (default: 60, empirically optimal)
                        Higher k values reduce rank sensitivity
            weights: Dictionary of weights for each ranking system
                    e.g., {"dense": 0.7, "sparse": 0.3}
                    If None, equal weights assigned to all systems
            normalize_scores: Whether to normalize final scores to [0, 1] range
        """
        self.k_parameter = k_parameter
        self.weights = weights or {}
        self.normalize_scores = normalize_scores
        logger.info(
            f"RRF initialized: k={k_parameter}, "
            f"weights={self.weights}, normalize={normalize_scores}"
        )

    def fuse(
        self, result_sets: Dict[str, List[Tuple[str, float, Dict]]], top_k: int = 10
    ) -> List[RRFScorer]:
        """
        Fuse multiple ranking result sets using Reciprocal Rank Fusion.

        Args:
            result_sets: Dictionary where keys are ranking system names (e.g., 'dense', 'sparse')
                        and values are lists of (doc_id, score, metadata) tuples sorted by rank
            top_k: Number of top results to return

        Returns:
            List of RRFScorer results sorted by RRF score (highest first)
        """
        if not result_sets:
            logger.warning("No result sets provided for RRF fusion")
            return []

        # Calculate total weight for normalization
        system_names = list(result_sets.keys())
        if not self.weights:
            # Equal weights if not specified
            total_weight = len(system_names)
            weights = {name: 1.0 for name in system_names}
        else:
            weights = self.weights
            total_weight = sum(weights.values())

        # Accumulate RRF scores
        rrf_scores = defaultdict(float)
        doc_metadatas = {}
        all_docs = set()

        for system_name, results in result_sets.items():
            if not results:
                continue

            weight = weights.get(system_name, 1.0)

            for rank, (doc_id, score, metadata) in enumerate(results, start=1):
                # RRF formula: contribute weight / (k + rank)
                rrf_scores[doc_id] += weight / (self.k_parameter + rank)
                all_docs.add(doc_id)
                if doc_id not in doc_metadatas:
                    doc_metadatas[doc_id] = metadata

        # Normalize scores if requested
        if self.normalize_scores and rrf_scores:
            max_score = max(rrf_scores.values())
            if max_score > 0:
                rrf_scores = {
                    doc_id: score / max_score for doc_id, score in rrf_scores.items()
                }

        # Create result objects and sort by RRF score
        fused_results = [
            RRFScorer(doc_id=doc_id, rrf_score=score, metadata=doc_metadatas.get(doc_id))
            for doc_id, score in rrf_scores.items()
        ]

        # Sort by RRF score (descending) and return top-k
        fused_results.sort(key=lambda x: x.rrf_score, reverse=True)
        top_results = fused_results[:top_k]

        top_score_str = f"{top_results[0].rrf_score:.6f}" if top_results else "N/A"
        logger.debug(
            f"RRF fusion complete: combined {len(system_names)} systems, "
            f"{len(all_docs)} unique documents, returned {len(top_results)} results, "
            f"top score: {top_score_str}"
        )

        return top_results


class RRFConfig:
    """Configuration container for RRF parameters"""

    def __init__(self):
        """Initialize with values from config"""
        self.k_parameter = config.rrf_k_parameter
        self.dense_weight = config.rrf_dense_weight
        self.sparse_weight = config.rrf_sparse_weight

    def to_weights_dict(self) -> Dict[str, float]:
        """Convert to weights dictionary for RRF"""
        return {"dense": self.dense_weight, "sparse": self.sparse_weight}
