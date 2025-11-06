"""Configuration module for Scientific Papers MCP Server"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Application configuration loaded from environment variables"""

    # Paths
    documents_path: Path = Path("D:/Github/Revue-de-litterature---Maitrise/Articles")
    chroma_path: Path = Path("D:/Claude Code/scientific-papers-mcp/data/chroma")

    # Search settings
    default_top_k: int = 10
    default_alpha: float = 0.5  # Hybrid search balance (0=keyword, 1=semantic)

    # Chunking settings
    max_chunk_size: int = 1000  # tokens per chunk
    chunk_overlap: int = 50  # token overlap between chunks

    # Embedding model
    embedding_model: str = "intfloat/multilingual-e5-large"

    # Indexing
    auto_index_on_start: bool = False
    watch_directory: bool = True

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def validate_paths(self) -> None:
        """Validate that required paths exist"""
        if not self.documents_path.exists():
            raise FileNotFoundError(
                f"Documents path does not exist: {self.documents_path}"
            )

        # Create chroma path if it doesn't exist
        self.chroma_path.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
