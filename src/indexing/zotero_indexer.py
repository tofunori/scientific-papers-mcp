"""Zotero library indexer with incremental updates and deduplication"""

from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass
import logging
from tqdm import tqdm

from ..models.document import ZoteroDocument
from ..extractors.pdf_extractor import extract_text_from_pdf, extract_metadata_from_pdf
from ..config import config
from ..utils.logger import setup_logger
from ..utils.text_chunker import DocumentChunker
from .indexing_state import IndexingStateManager
from .deduplicator import DocumentDeduplicator

# Marker extractors (optional, will be loaded on demand)
MARKER_API_AVAILABLE = False
MARKER_LOCAL_AVAILABLE = False

try:
    from ..extractors.marker_api_extractor import MarkerAPIExtractor
    MARKER_API_AVAILABLE = True
except ImportError:
    pass

try:
    from ..extractors.marker_local_extractor import MarkerLocalExtractor
    MARKER_LOCAL_AVAILABLE = MarkerLocalExtractor.is_available()
except ImportError:
    pass

logger = setup_logger(__name__)


@dataclass
class IndexingStatistics:
    """Statistics for indexing operation"""

    total_scanned: int = 0
    added: int = 0
    updated: int = 0
    skipped: int = 0
    duplicates: int = 0
    errors: int = 0
    error_files: List[str] = None

    def __post_init__(self):
        if self.error_files is None:
            self.error_files = []

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "total_scanned": self.total_scanned,
            "added": self.added,
            "updated": self.updated,
            "skipped": self.skipped,
            "duplicates": self.duplicates,
            "errors": self.errors,
            "error_count": len(self.error_files),
            "error_files": self.error_files[:10],  # Limit to first 10
        }


class ZoteroLibraryIndexer:
    """
    Indexes Zotero library with intelligent features:
    - Incremental updates (only reindex changed files)
    - Deduplication (DOI and title matching)
    - Batch processing with progress tracking
    - Error recovery and statistics

    Inspired by Zotero MCP's indexing strategy
    """

    def __init__(
        self,
        search_engine,
        library_path: Path = None,
        state_manager: IndexingStateManager = None,
        deduplicator: DocumentDeduplicator = None,
    ):
        """
        Initialize Zotero library indexer

        Args:
            search_engine: Hybrid search engine instance
            library_path: Path to Zotero storage (default from config)
            state_manager: Indexing state manager (default creates new)
            deduplicator: Document deduplicator (default creates new)
        """
        self.search_engine = search_engine
        self.library_path = library_path or config.documents_path
        self.state_manager = state_manager or IndexingStateManager()
        self.deduplicator = deduplicator or DocumentDeduplicator()

        # Initialize document chunker for splitting papers
        # Use markdown-aware chunking if using Marker extraction
        use_markdown = self.pdf_extraction_method in ["marker_api", "marker_local"]
        self.chunker = DocumentChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            use_markdown_separators=use_markdown,
        )

        self.statistics = IndexingStatistics()

        # Initialize PDF extractors based on config
        self.pdf_extraction_method = config.pdf_extraction_method
        self.marker_api_extractor = None
        self.marker_local_extractor = None

        # Initialize Marker API extractor if configured
        if self.pdf_extraction_method == "marker_api" and MARKER_API_AVAILABLE:
            if config.marker_api_key:
                self.marker_api_extractor = MarkerAPIExtractor(
                    api_key=config.marker_api_key,
                    timeout=config.marker_api_timeout,
                )
                logger.info("Marker API extractor initialized")
            else:
                logger.warning("Marker API selected but no API key found. Falling back to PyMuPDF")
                self.pdf_extraction_method = "pymupdf"

        # Initialize Marker Local extractor if configured
        elif self.pdf_extraction_method == "marker_local" and MARKER_LOCAL_AVAILABLE:
            try:
                self.marker_local_extractor = MarkerLocalExtractor(
                    use_llm=config.marker_local_use_llm,
                    llm_provider=config.marker_local_llm_provider or None,
                    llm_model=config.marker_local_llm_model or None,
                    batch_multiplier=config.marker_local_batch_multiplier,
                )
                logger.info("Marker Local extractor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Marker Local: {e}. Falling back to PyMuPDF")
                self.pdf_extraction_method = "pymupdf"

        logger.info(f"Zotero library indexer initialized (path: {self.library_path})")
        logger.info(f"PDF extraction method: {self.pdf_extraction_method}")
        logger.info(f"Document chunker initialized (chunk_size={config.chunk_size}, overlap={config.chunk_overlap})")

    def scan_zotero_library(
        self,
        limit: Optional[int] = None,
        file_types: List[str] = None,
    ) -> List[Path]:
        """
        Scan Zotero library for documents

        Zotero storage structure: Each item in a folder like ABCDEF12/
        containing PDFs and metadata

        Args:
            limit: Maximum number of documents to scan (for testing)
            file_types: File extensions to scan (default: ["pdf"])

        Returns:
            List of PDF file paths
        """
        if file_types is None:
            file_types = ["pdf"]  # Focus on PDFs for now

        logger.info(f"Scanning Zotero library: {self.library_path}")

        all_files = []

        try:
            # Get list of item directories for progress tracking
            item_dirs = list(self.library_path.iterdir())

            # Scan all subdirectories in Zotero storage with progress bar
            for item_dir in tqdm(item_dirs, desc="Scanning Zotero folders", unit="folder"):
                if not item_dir.is_dir():
                    continue

                # Look for PDFs in each item directory
                for file_type in file_types:
                    pattern = f"*.{file_type}"
                    files = list(item_dir.glob(pattern))
                    all_files.extend(files)

                    if limit and len(all_files) >= limit:
                        break

                if limit and len(all_files) >= limit:
                    break

            if limit:
                all_files = all_files[:limit]

            logger.info(f"Found {len(all_files)} documents in Zotero library")
            return all_files

        except Exception as e:
            logger.error(f"Error scanning Zotero library: {e}")
            return []

    def extract_document(self, file_path: Path) -> Optional[ZoteroDocument]:
        """
        Extract metadata and content from a document file

        Supports multiple extraction methods:
        - pymupdf: Fast, lightweight (default)
        - marker_api: High quality, cloud-based (Datalab API)
        - marker_local: High quality, local processing

        Args:
            file_path: Path to PDF file

        Returns:
            ZoteroDocument or None if extraction failed
        """
        try:
            logger.debug(f"Extracting: {file_path.name} (method: {self.pdf_extraction_method})")

            # Initialize variables
            full_text = ""
            metadata = {}
            images = []
            is_scanned = False
            extraction_source = self.pdf_extraction_method

            # Try Marker API extraction
            if self.pdf_extraction_method == "marker_api" and self.marker_api_extractor:
                try:
                    markdown_text, marker_metadata, images = self.marker_api_extractor.extract_text_from_pdf(
                        file_path,
                        use_llm=config.marker_use_llm,
                        force_ocr=config.marker_force_ocr,
                        extract_images=True,
                    )
                    # Extract enhanced metadata from markdown
                    metadata = self.marker_api_extractor.extract_metadata_from_markdown(
                        markdown_text, marker_metadata
                    )
                    full_text = markdown_text
                    is_scanned = marker_metadata.get("is_scanned", False)
                    logger.debug(f"Successfully extracted with Marker API: {len(markdown_text)} chars")

                except Exception as e:
                    logger.warning(f"Marker API failed for {file_path.name}: {e}")
                    if config.marker_fallback_to_pymupdf:
                        logger.info(f"Falling back to PyMuPDF for {file_path.name}")
                        extraction_source = "pymupdf_fallback"
                    else:
                        raise

            # Try Marker Local extraction
            elif self.pdf_extraction_method == "marker_local" and self.marker_local_extractor:
                try:
                    markdown_text, marker_metadata, images = self.marker_local_extractor.extract_text_from_pdf(
                        file_path,
                        extract_images=True,
                    )
                    # Extract enhanced metadata from markdown
                    metadata = self.marker_local_extractor.extract_metadata_from_markdown(
                        markdown_text, marker_metadata
                    )
                    full_text = markdown_text
                    is_scanned = marker_metadata.get("is_scanned", False)
                    logger.debug(f"Successfully extracted with Marker Local: {len(markdown_text)} chars")

                except Exception as e:
                    logger.warning(f"Marker Local failed for {file_path.name}: {e}")
                    if config.marker_fallback_to_pymupdf:
                        logger.info(f"Falling back to PyMuPDF for {file_path.name}")
                        extraction_source = "pymupdf_fallback"
                    else:
                        raise

            # Use PyMuPDF extraction (default or fallback)
            if not full_text or extraction_source in ["pymupdf", "pymupdf_fallback"]:
                # Extract PDF metadata
                metadata = extract_metadata_from_pdf(file_path)

                # Extract PDF text content AND images for multimodal embedding
                full_text, is_scanned, images = extract_text_from_pdf(file_path, extract_images=True)

                logger.debug(f"Extracted with PyMuPDF: {len(full_text)} chars")

            # Get file modification time
            file_modified = datetime.fromtimestamp(file_path.stat().st_mtime)

            # Intelligent author fallback: if metadata extraction fails, try text extraction
            authors = metadata.get("authors", [])
            if not authors and full_text:
                logger.debug(f"No authors in metadata for {file_path.name}, trying text extraction...")
                # Import the text extraction function
                from ..extractors.pdf_extractor import _extract_authors_from_text
                authors = _extract_authors_from_text(full_text[:2000])  # First 2000 chars usually contain authors

                if authors:
                    logger.debug(f"Successfully extracted {len(authors)} authors from text: {authors[:3]}")
                else:
                    logger.warning(f"No authors found for {file_path.name} - using placeholder")
                    authors = ["Unknown"]  # Placeholder to enable fulltext search

            # Create ZoteroDocument
            doc = ZoteroDocument(
                item_key="",  # Will be auto-generated from file_path
                file_path=file_path,
                filename=file_path.name,
                title=metadata.get("title", file_path.stem),
                authors=authors,
                year=metadata.get("year"),
                doi=metadata.get("doi"),
                publication=metadata.get("journal"),
                abstract=metadata.get("abstract"),
                keywords=metadata.get("keywords", []),
                tags=[],  # Could be extracted from Zotero metadata files
                collections=[],  # Could be extracted from Zotero database
                date_modified=file_modified,
                has_fulltext=bool(full_text),
                fulltext_source="pdf" if not is_scanned else "ocr",
                file_type="pdf",
                is_scanned=is_scanned,
                page_count=metadata.get("page_count"),
                full_text=full_text,
                images=images,  # NEW: Store images for multimodal embedding
            )

            # Validation: Log warning if authors are still missing
            if not doc.authors or doc.authors == ["Unknown"]:
                self.statistics.missing_authors = getattr(self.statistics, 'missing_authors', 0) + 1
                logger.warning(
                    f"Missing authors for {file_path.name} - "
                    f"Title: {doc.title[:50]}... "
                    f"Total papers without authors: {self.statistics.missing_authors}"
                )

            logger.debug(
                f"Extracted: {doc.title[:50]}... "
                f"(Authors: {len(doc.authors)}, DOI: {doc.doi or 'N/A'}, pages: {doc.page_count})"
            )

            return doc

        except Exception as e:
            logger.error(f"Error extracting {file_path.name}: {e}")
            self.statistics.errors += 1
            self.statistics.error_files.append(str(file_path))
            return None

    def check_incremental_updates(
        self, documents: List[ZoteroDocument], force_rebuild: bool = False
    ) -> List[ZoteroDocument]:
        """
        Filter documents to only those needing indexing

        Args:
            documents: List of all documents
            force_rebuild: If True, index all documents

        Returns:
            List of documents that need indexing
        """
        if not config.enable_incremental_indexing or force_rebuild:
            logger.info("Incremental indexing disabled or force rebuild")
            return documents

        needs_indexing = []

        for doc in documents:
            if self.state_manager.should_skip_file(doc.file_path, force_rebuild):
                self.statistics.skipped += 1
                logger.debug(f"Skipping unchanged: {doc.filename}")
            else:
                needs_indexing.append(doc)

        logger.info(
            f"Incremental check: {len(needs_indexing)} need indexing, "
            f"{self.statistics.skipped} skipped"
        )

        return needs_indexing

    def deduplicate_documents(
        self, documents: List[ZoteroDocument]
    ) -> List[ZoteroDocument]:
        """
        Remove duplicate documents

        Args:
            documents: List of documents

        Returns:
            List of unique documents
        """
        if not config.enable_deduplication:
            logger.info("Deduplication disabled")
            return documents

        unique_docs, duplicate_docs = self.deduplicator.find_duplicates(documents)

        self.statistics.duplicates = len(duplicate_docs)

        # Mark duplicates in state
        for dup_doc in duplicate_docs:
            if dup_doc.doi:
                self.state_manager.mark_as_duplicate(dup_doc.doi, dup_doc.file_path)
            else:
                # Use normalized title hash as key
                title_hash = dup_doc.item_key
                self.state_manager.mark_as_duplicate(title_hash, dup_doc.file_path)

        logger.info(
            f"Deduplication: {len(unique_docs)} unique, {len(duplicate_docs)} duplicates"
        )

        return unique_docs

    def index_batch(
        self,
        documents: List[ZoteroDocument],
        batch_size: int = None,
    ) -> None:
        """
        Index documents in batches with chunking and progress tracking

        NEW: Each document is split into multiple chunks for better retrieval.
        Expected: 15-20 chunks per scientific paper.

        Args:
            documents: List of documents to index
            batch_size: Batch size for ChromaDB insertion (default from config)
        """
        if not documents:
            logger.info("No documents to index")
            return

        batch_size = batch_size or config.batch_indexing_size

        logger.info(
            f"Indexing {len(documents)} documents with chunking "
            f"(batch_size={batch_size})..."
        )

        # Collect ALL chunks from ALL documents first
        all_chunk_ids = []
        all_chunk_texts = []
        all_chunk_metadatas = []
        total_chunks = 0

        for doc_idx, doc in enumerate(tqdm(documents, desc="Chunking documents", unit="doc")):
            try:
                # Generate text-only chunks from document
                chunks = doc.to_chunks(self.chunker, include_fulltext=True)

                # Add all text chunks to the collection
                for chunk_text, chunk_meta in chunks:
                    all_chunk_ids.append(chunk_meta["chunk_id"])
                    all_chunk_texts.append(chunk_text)
                    all_chunk_metadatas.append(chunk_meta)

                total_chunks += len(chunks)

                # NEW: Add multimodal chunks for documents with images
                if doc.images:
                    # Create one multimodal chunk per image (limit to first 3 important images)
                    for img_idx, img in enumerate(doc.images[:3]):
                        # Create multimodal input dict
                        mm_chunk = {
                            "text": doc.to_embedding_text(include_fulltext=False),  # Metadata only
                            "image": img["image_base64"]
                        }

                        # Create unique ID for this multimodal chunk
                        mm_chunk_id = f"{doc.item_key}_mm_{img_idx}"

                        # Add metadata
                        mm_metadata = doc.to_metadata_dict()
                        mm_metadata["chunk_id"] = mm_chunk_id
                        mm_metadata["chunk_index"] = f"MM{img_idx}"
                        mm_metadata["is_multimodal"] = "True"
                        mm_metadata["image_page"] = str(img["page_num"] + 1)

                        all_chunk_ids.append(mm_chunk_id)
                        all_chunk_texts.append(mm_chunk)  # Dict, not string
                        all_chunk_metadatas.append(mm_metadata)

                        total_chunks += 1

                    logger.debug(f"Added {min(len(doc.images), 3)} multimodal chunks for {doc.filename}")

                # Track indexing for state (per document, not per chunk)
                existing_info = self.state_manager.get_file_info(doc.file_path)
                if existing_info:
                    self.statistics.updated += 1
                else:
                    self.statistics.added += 1

                # Mark document as indexed in state
                self.state_manager.mark_as_indexed(
                    file_path=doc.file_path, doc_id=doc.item_key, doi=doc.doi
                )

            except Exception as e:
                logger.error(f"Error chunking {doc.filename}: {e}")
                self.statistics.errors += 1
                self.statistics.error_files.append(str(doc.file_path))
                continue

        if not all_chunk_ids:
            logger.warning("No chunks generated from documents")
            return

        logger.info(
            f"Generated {total_chunks} chunks from {len(documents)} documents "
            f"(avg {total_chunks/len(documents):.1f} chunks/doc)"
        )

        # Index all chunks in batches
        logger.info(f"Indexing {len(all_chunk_ids)} chunks in batches...")

        for batch_idx in tqdm(range(0, len(all_chunk_ids), batch_size), desc="Indexing chunks", unit="batch"):
            batch_ids = all_chunk_ids[batch_idx : batch_idx + batch_size]
            batch_texts = all_chunk_texts[batch_idx : batch_idx + batch_size]
            batch_metas = all_chunk_metadatas[batch_idx : batch_idx + batch_size]

            try:
                self.search_engine.index_documents_batch(
                    batch_ids, batch_texts, batch_metas
                )

            except Exception as e:
                logger.error(f"Error indexing chunk batch: {e}")
                self.statistics.errors += len(batch_ids)

        logger.info(
            f"Batch indexing complete: {total_chunks} chunks from "
            f"{self.statistics.added + self.statistics.updated} documents"
        )

    def index_library(
        self,
        force_rebuild: bool = False,
        limit: Optional[int] = None,
        save_state: bool = True,
    ) -> Dict:
        """
        Main indexing workflow

        Steps:
        1. Scan Zotero library for files
        2. Extract metadata and content
        3. Check incremental updates (skip unchanged)
        4. Deduplicate documents
        5. Batch index with progress tracking
        6. Save state

        Args:
            force_rebuild: Force reindex all files (ignore state)
            limit: Limit number of files to process (for testing)
            save_state: Save indexing state after completion

        Returns:
            Statistics dictionary
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Starting Zotero library indexing")
        logger.info(f"Force rebuild: {force_rebuild}, Limit: {limit or 'None'}")
        logger.info("=" * 60)

        try:
            # Step 1: Scan library
            file_paths = self.scan_zotero_library(limit=limit)
            self.statistics.total_scanned = len(file_paths)

            if not file_paths:
                logger.warning("No files found in Zotero library")
                return self.statistics.to_dict()

            # Step 2: Extract documents
            logger.info("Extracting metadata and content...")
            documents = []
            for file_path in tqdm(file_paths, desc="Extracting documents", unit="doc"):
                doc = self.extract_document(file_path)
                if doc:
                    documents.append(doc)

            logger.info(f"Successfully extracted {len(documents)} documents")

            # Step 3: Incremental updates
            documents = self.check_incremental_updates(documents, force_rebuild)

            if not documents:
                logger.info("All documents up to date, nothing to index")
                return self.statistics.to_dict()

            # Step 4: Deduplication
            documents = self.deduplicate_documents(documents)

            # Step 5: Batch indexing
            self.index_batch(documents)

            # Step 6: Save state
            if save_state:
                if force_rebuild:
                    self.state_manager.mark_full_reindex()
                self.state_manager.save_state()
                logger.info("Indexing state saved")

            # Summary
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info("=" * 60)
            logger.info("Indexing Complete!")
            logger.info(f"Time elapsed: {elapsed:.1f}s")
            logger.info(f"Total scanned: {self.statistics.total_scanned}")
            logger.info(f"Added: {self.statistics.added}")
            logger.info(f"Updated: {self.statistics.updated}")
            logger.info(f"Skipped (unchanged): {self.statistics.skipped}")
            logger.info(f"Duplicates removed: {self.statistics.duplicates}")
            logger.info(f"Errors: {self.statistics.errors}")
            logger.info("=" * 60)

            return self.statistics.to_dict()

        except Exception as e:
            logger.error(f"Error during library indexing: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return self.statistics.to_dict()
