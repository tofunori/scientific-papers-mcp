"""State management for incremental indexing"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Set
import logging

from ..config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class IndexingStateManager:
    """
    Manages indexing state for incremental updates

    Tracks which files have been indexed and when they were last modified
    to avoid reindexing unchanged documents (inspired by Zotero MCP)
    """

    def __init__(self, state_file_path: Path = None):
        """
        Initialize indexing state manager

        Args:
            state_file_path: Path to state file (default from config)
        """
        self.state_file = state_file_path or config.indexing_state_path
        self.state = self._load_state()
        logger.info(f"Indexing state loaded from {self.state_file}")

    def _load_state(self) -> dict:
        """
        Load indexing state from JSON file

        Returns:
            State dictionary with structure:
            {
                "indexed_files": {
                    "file_path": {
                        "date_modified": "ISO timestamp",
                        "doc_id": "document ID",
                        "doi": "10.xxxx/yyyy",
                        "date_indexed": "ISO timestamp"
                    }
                },
                "deduplicated_files": {
                    "doi_or_title_hash": ["file_path1", "file_path2"]
                },
                "statistics": {
                    "total_indexed": 0,
                    "last_full_reindex": "ISO timestamp",
                    "last_incremental_update": "ISO timestamp"
                }
            }
        """
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    logger.info(
                        f"Loaded state with {len(state.get('indexed_files', {}))} indexed files"
                    )
                    return state
            except Exception as e:
                logger.error(f"Error loading state file: {e}")
                logger.warning("Creating new state")

        # Create new state structure
        return {
            "indexed_files": {},
            "deduplicated_files": {},
            "statistics": {
                "total_indexed": 0,
                "last_full_reindex": None,
                "last_incremental_update": None,
            },
        }

    def save_state(self) -> None:
        """Save current state to JSON file"""
        try:
            # Ensure directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)

            logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Error saving state file: {e}")

    def should_skip_file(
        self, file_path: Path, force_rebuild: bool = False
    ) -> bool:
        """
        Check if file should be skipped (already indexed and unchanged)

        Args:
            file_path: Path to file
            force_rebuild: If True, never skip (force reindexing)

        Returns:
            True if file should be skipped, False if needs indexing
        """
        if force_rebuild:
            return False

        if not config.enable_incremental_indexing:
            return False

        file_path_str = str(file_path.resolve())

        # Check if file is in indexed files
        if file_path_str not in self.state["indexed_files"]:
            return False

        # Check if file has been modified since last indexing
        try:
            current_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            indexed_info = self.state["indexed_files"][file_path_str]
            last_modified = datetime.fromisoformat(indexed_info["date_modified"])

            if current_modified > last_modified:
                logger.debug(f"File modified since last index: {file_path.name}")
                return False

            # File is unchanged, can skip
            logger.debug(f"Skipping unchanged file: {file_path.name}")
            return True

        except Exception as e:
            logger.warning(f"Error checking file modification time: {e}")
            return False  # If error, reindex to be safe

    def mark_as_indexed(
        self,
        file_path: Path,
        doc_id: str,
        doi: Optional[str] = None,
    ) -> None:
        """
        Mark file as indexed with current timestamp

        Args:
            file_path: Path to indexed file
            doc_id: Document ID in ChromaDB
            doi: Optional DOI for duplicate tracking
        """
        file_path_str = str(file_path.resolve())

        try:
            file_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
        except Exception as e:
            logger.warning(f"Error getting file modification time: {e}")
            file_modified = datetime.now()

        self.state["indexed_files"][file_path_str] = {
            "date_modified": file_modified.isoformat(),
            "doc_id": doc_id,
            "doi": doi or "",
            "date_indexed": datetime.now().isoformat(),
        }

        # Update statistics
        self.state["statistics"]["total_indexed"] = len(self.state["indexed_files"])
        self.state["statistics"]["last_incremental_update"] = datetime.now().isoformat()

    def mark_as_duplicate(self, key: str, file_path: Path) -> None:
        """
        Mark file as duplicate by DOI or title hash

        Args:
            key: DOI or normalized title hash
            file_path: Path to duplicate file
        """
        file_path_str = str(file_path.resolve())

        if key not in self.state["deduplicated_files"]:
            self.state["deduplicated_files"][key] = []

        if file_path_str not in self.state["deduplicated_files"][key]:
            self.state["deduplicated_files"][key].append(file_path_str)
            logger.info(f"Marked as duplicate (key={key}): {file_path.name}")

    def get_duplicate_files(self, key: str) -> list:
        """
        Get list of files marked as duplicates for given key

        Args:
            key: DOI or normalized title hash

        Returns:
            List of file paths
        """
        return self.state["deduplicated_files"].get(key, [])

    def get_indexed_files(self) -> Set[str]:
        """
        Get set of all indexed file paths

        Returns:
            Set of file path strings
        """
        return set(self.state["indexed_files"].keys())

    def get_file_info(self, file_path: Path) -> Optional[dict]:
        """
        Get indexing info for a file

        Args:
            file_path: Path to file

        Returns:
            Dict with indexing info or None if not indexed
        """
        file_path_str = str(file_path.resolve())
        return self.state["indexed_files"].get(file_path_str)

    def remove_file(self, file_path: Path) -> None:
        """
        Remove file from indexed state (e.g., if file deleted)

        Args:
            file_path: Path to file
        """
        file_path_str = str(file_path.resolve())

        if file_path_str in self.state["indexed_files"]:
            del self.state["indexed_files"][file_path_str]
            logger.debug(f"Removed from state: {file_path.name}")

    def mark_full_reindex(self) -> None:
        """Mark that a full reindex was performed"""
        self.state["statistics"]["last_full_reindex"] = datetime.now().isoformat()
        logger.info("Marked full reindex completion")

    def get_statistics(self) -> dict:
        """Get indexing statistics"""
        return self.state["statistics"]

    def clear_state(self) -> None:
        """Clear all state (for fresh start)"""
        self.state = {
            "indexed_files": {},
            "deduplicated_files": {},
            "statistics": {
                "total_indexed": 0,
                "last_full_reindex": None,
                "last_incremental_update": None,
            },
        }
        logger.warning("State cleared")
        self.save_state()
