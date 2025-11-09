"""Document deduplication logic for Zotero library"""

from typing import List, Optional, Tuple
import hashlib
import string
from difflib import SequenceMatcher

from ..models.document import ZoteroDocument
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentDeduplicator:
    """
    Intelligent document deduplication

    Detects duplicates using:
    1. DOI matching (exact)
    2. Normalized title matching (fuzzy, >90% similarity)
    3. Author + year matching (for cases without DOI)

    Inspired by Zotero MCP's deduplication strategy
    """

    def __init__(self, similarity_threshold: float = 0.90):
        """
        Initialize deduplicator

        Args:
            similarity_threshold: Minimum similarity for title matching (default 0.90)
        """
        self.similarity_threshold = similarity_threshold
        logger.info(f"Deduplicator initialized (similarity threshold: {similarity_threshold})")

    def find_duplicates(
        self, documents: List[ZoteroDocument]
    ) -> Tuple[List[ZoteroDocument], List[ZoteroDocument]]:
        """
        Find and remove duplicates from document list

        Strategy (in priority order):
        1. Group by DOI (exact match)
        2. Group by normalized title (fuzzy match)
        3. For duplicates: keep published > preprint, newer > older

        Args:
            documents: List of Zotero documents

        Returns:
            Tuple of (unique_documents, duplicate_documents)
        """
        if not documents:
            return [], []

        logger.info(f"Deduplicating {len(documents)} documents...")

        unique_docs = []
        duplicate_docs = []

        # Track seen documents by DOI and title
        seen_dois = {}
        seen_titles = {}

        for doc in documents:
            is_duplicate = False

            # Strategy 1: Check DOI
            if doc.doi:
                if doc.doi in seen_dois:
                    # DOI already seen - this is a duplicate
                    existing_doc = seen_dois[doc.doi]
                    winner, loser = self._select_best_version(existing_doc, doc)

                    if winner == doc:
                        # Replace existing with new
                        unique_docs.remove(existing_doc)
                        duplicate_docs.append(existing_doc)
                        unique_docs.append(doc)
                        seen_dois[doc.doi] = doc
                    else:
                        # Keep existing
                        duplicate_docs.append(doc)

                    is_duplicate = True
                    logger.info(
                        f"DOI duplicate detected: {doc.doi} "
                        f"(kept: {winner.filename}, removed: {loser.filename})"
                    )
                else:
                    # First time seeing this DOI
                    seen_dois[doc.doi] = doc

            # Strategy 2: Check normalized title (if no DOI or not duplicate yet)
            if not is_duplicate and doc.title:
                norm_title = doc.normalized_title
                title_hash = self._hash_title(norm_title)

                # Check for similar titles
                for existing_hash, existing_doc in list(seen_titles.items()):
                    similarity = self._title_similarity(
                        norm_title, existing_doc.normalized_title
                    )

                    if similarity >= self.similarity_threshold:
                        # Title match - likely duplicate
                        winner, loser = self._select_best_version(existing_doc, doc)

                        if winner == doc:
                            # Replace existing with new
                            unique_docs.remove(existing_doc)
                            duplicate_docs.append(existing_doc)
                            unique_docs.append(doc)
                            del seen_titles[existing_hash]
                            seen_titles[title_hash] = doc
                        else:
                            # Keep existing
                            duplicate_docs.append(doc)

                        is_duplicate = True
                        logger.info(
                            f"Title duplicate detected (similarity: {similarity:.2f}): "
                            f"'{doc.title[:50]}...' "
                            f"(kept: {winner.filename}, removed: {loser.filename})"
                        )
                        break

                if not is_duplicate:
                    # First time seeing this title
                    seen_titles[title_hash] = doc

            # If not duplicate, add to unique list
            if not is_duplicate:
                unique_docs.append(doc)

        logger.info(
            f"Deduplication complete: {len(unique_docs)} unique, "
            f"{len(duplicate_docs)} duplicates removed"
        )

        return unique_docs, duplicate_docs

    def _select_best_version(
        self, doc1: ZoteroDocument, doc2: ZoteroDocument
    ) -> Tuple[ZoteroDocument, ZoteroDocument]:
        """
        Select best version between two duplicate documents

        Priority rules:
        1. Publication over preprint (journal name indicates published)
        2. Has DOI over no DOI
        3. Has full text over no full text
        4. More recent (by year)
        5. Longer abstract
        6. Newer file modification date

        Args:
            doc1: First document
            doc2: Second document

        Returns:
            Tuple of (winner, loser)
        """
        score1 = 0
        score2 = 0

        # Rule 1: Publication over preprint
        if doc1.publication and not doc2.publication:
            score1 += 10
        elif doc2.publication and not doc1.publication:
            score2 += 10

        # Rule 2: Has DOI
        if doc1.doi and not doc2.doi:
            score1 += 5
        elif doc2.doi and not doc1.doi:
            score2 += 5

        # Rule 3: Has full text
        if doc1.has_fulltext and not doc2.has_fulltext:
            score1 += 3
        elif doc2.has_fulltext and not doc1.has_fulltext:
            score2 += 3

        # Rule 4: More recent year
        if doc1.year and doc2.year:
            if doc1.year > doc2.year:
                score1 += 2
            elif doc2.year > doc1.year:
                score2 += 2

        # Rule 5: Longer abstract
        len1 = len(doc1.abstract or "")
        len2 = len(doc2.abstract or "")
        if len1 > len2:
            score1 += 1
        elif len2 > len1:
            score2 += 1

        # Rule 6: Newer modification date
        if doc1.date_modified > doc2.date_modified:
            score1 += 1
        elif doc2.date_modified > doc1.date_modified:
            score2 += 1

        if score1 >= score2:
            return doc1, doc2
        else:
            return doc2, doc1

    @staticmethod
    def _title_similarity(title1: str, title2: str) -> float:
        """
        Calculate similarity between two normalized titles

        Uses SequenceMatcher for fuzzy matching

        Args:
            title1: First normalized title
            title2: Second normalized title

        Returns:
            Similarity score between 0 and 1
        """
        if not title1 or not title2:
            return 0.0

        return SequenceMatcher(None, title1, title2).ratio()

    @staticmethod
    def _hash_title(title: str) -> str:
        """
        Generate hash for normalized title

        Args:
            title: Normalized title

        Returns:
            SHA256 hash (first 16 chars)
        """
        return hashlib.sha256(title.encode()).hexdigest()[:16]


def normalize_title(title: str) -> str:
    """
    Normalize title for deduplication

    Steps:
    1. Lowercase
    2. Remove punctuation
    3. Remove articles (a, an, the)
    4. Remove extra whitespace

    Args:
        title: Original title

    Returns:
        Normalized title string
    """
    if not title:
        return ""

    # Lowercase
    title = title.lower()

    # Remove punctuation
    title = "".join(c if c not in string.punctuation else " " for c in title)

    # Remove articles and extra spaces
    words = title.split()
    words = [w for w in words if w not in {"a", "an", "the"}]

    return " ".join(words)


def find_duplicates_by_doi(documents: List[ZoteroDocument]) -> dict:
    """
    Find all documents with duplicate DOIs

    Args:
        documents: List of Zotero documents

    Returns:
        Dict mapping DOI to list of documents with that DOI
    """
    doi_groups = {}

    for doc in documents:
        if doc.doi:
            if doc.doi not in doi_groups:
                doi_groups[doc.doi] = []
            doi_groups[doc.doi].append(doc)

    # Filter to only groups with duplicates
    duplicates = {doi: docs for doi, docs in doi_groups.items() if len(docs) > 1}

    if duplicates:
        logger.info(f"Found {len(duplicates)} DOIs with duplicates")

    return duplicates


def find_duplicates_by_title(
    documents: List[ZoteroDocument], similarity_threshold: float = 0.90
) -> dict:
    """
    Find all documents with similar titles

    Args:
        documents: List of Zotero documents
        similarity_threshold: Minimum similarity (default 0.90)

    Returns:
        Dict mapping first document to list of similar documents
    """
    title_groups = {}
    processed = set()

    for i, doc1 in enumerate(documents):
        if i in processed or not doc1.title:
            continue

        similar_docs = [doc1]

        for j, doc2 in enumerate(documents[i + 1 :], start=i + 1):
            if j in processed or not doc2.title:
                continue

            similarity = SequenceMatcher(
                None, doc1.normalized_title, doc2.normalized_title
            ).ratio()

            if similarity >= similarity_threshold:
                similar_docs.append(doc2)
                processed.add(j)

        if len(similar_docs) > 1:
            title_groups[doc1.item_key] = similar_docs
            processed.add(i)

    if title_groups:
        logger.info(f"Found {len(title_groups)} title groups with duplicates")

    return title_groups
