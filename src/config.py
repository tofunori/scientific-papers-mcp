"""Configuration module for Scientific Papers MCP Server"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

# Load .env file and set environment variables BEFORE defining Config class
from dotenv import dotenv_values

# Find and load .env file from project root
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    env_values = dotenv_values(env_file)
    # Set all env vars explicitly (avoids any encoding issues)
    for key, value in env_values.items():
        if value is not None:
            os.environ[key] = str(value)


class Config(BaseSettings):
    """Application configuration loaded from environment variables"""

    # Paths
    documents_path: Path = Path("C:/Users/thier/Zotero/storage")
    chroma_path: Path = Path("D:/Claude Code/scientific-papers-mcp/data/chroma")

    # Collection management
    default_collection_name: str = "scientific_papers_voyage"  # Default ChromaDB collection (Voyage AI)

    # Chroma Cloud configuration (hybrid mode: local + cloud)
    use_chroma_cloud: bool = False  # Toggle between local PersistentClient and CloudClient
    chroma_tenant: str = "61d5e32b-2b82-44c9-8dec-a8aff28ea279"  # Chroma Cloud tenant ID
    chroma_database: str = "Scientific"  # Chroma Cloud database name
    chroma_api_key: str = ""  # Chroma Cloud API key (from CHROMA_API_KEY env var)

    # Search settings
    default_top_k: int = 10
    default_alpha: float = 0.5  # Hybrid search balance (0=keyword, 1=semantic)

    # RRF (Reciprocal Rank Fusion) settings for improved hybrid search
    use_rrf: bool = True  # Use proper RRF formula instead of linear combination
    rrf_k_parameter: int = 60  # RRF smoothing parameter (higher = more uniform ranking)
    rrf_dense_weight: float = 0.7  # Weight for dense semantic search results
    rrf_sparse_weight: float = 0.3  # Weight for sparse keyword search results

    # Sparse vector / Cloud Search API settings
    use_cloud_search_api: bool = False  # Use Chroma Cloud Search() API with native sparse vectors
    sparse_vector_enabled: bool = True  # Enable sparse vectors (BM25 local, SPLADE cloud)

    # Cloud sparse vector configuration (SPLADE - Sparse Lexical and Semantic Embeddings)
    cloud_sparse_embedding_model: str = "splade-cocondenser-ensembledistil"  # SPLADE model for cloud
    cloud_sparse_embedding_dims: int = 30522  # Vocabulary size for SPLADE (BERT vocab)
    enable_cloud_search_api: bool = False  # Enable Chroma Cloud Search() API (requires Chroma v0.4.0+)

    # Hybrid local + cloud sync settings
    sync_to_cloud: bool = True  # Automatically sync indexed documents to Chroma Cloud
    default_search_source: str = "local"  # Default search source: 'local' (fast) or 'cloud' (remote)

    # Chunking settings (Optimized for contextual embedding models)
    # - 1024 tokens optimal for scientific papers with voyage-context-3
    # - voyage-context-3 has 32K context window, reduces chunking strategy sensitivity
    # - Overlap depends on embedding model:
    #   * Voyage AI: 100 tokens overlap (captures transitions between sections)
    #   * Jina-v4: 100 tokens (~20%) (late_chunking preserves context)
    #   * Local models: 100 tokens (~20%) (no contextual features)
    chunking_enabled: bool = True
    chunk_size: int = 1024  # tokens per chunk (optimal for scientific papers + voyage-context-3)
    chunk_overlap: int = 100  # token overlap for transitions
    chunk_encoding: str = "cl100k_base"  # Token encoding (GPT tokenizer)

    # Contextual chunking: Group chunks by document for contextual embeddings
    # When True, Jina uses late_chunking and Voyage uses contextualized_embed
    enable_contextual_chunking: bool = True

    # Jina API settings
    use_jina_api: bool = False  # Toggle between API and local models
    jina_api_key: str = ""
    jina_model: str = "jina-embeddings-v4"

    # Voyage AI settings (hybrid: context-3 for text, multimodal-3 for images)
    use_voyage_api: bool = True  # Toggle to use Voyage AI instead of Jina/local
    voyage_api_key: str = ""
    voyage_text_model: str = "voyage-context-3"  # For text-only chunks
    voyage_multimodal_model: str = "voyage-multimodal-3"  # For multimodal chunks

    # Embedding model (used when use_jina_api=False and use_voyage_api=False)
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dimensions: int = 1024  # Qina v4 and Qwen3: 1024 dimensions
    batch_size: int = 12  # Batch size for embedding (Jina: 12×512=6,144 tokens < 8,192 limit)

    # Reranking model
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 50  # Retrieve 50 candidates before reranking

    # Cohere Rerank Configuration
    use_cohere_rerank: bool = True  # Use Cohere Rerank API instead of local cross-encoder
    cohere_api_key: str = ""
    cohere_model: str = "rerank-v3.5"
    cohere_top_k: int = 10  # Number of results to return from Cohere

    # Indexing
    auto_index_on_start: bool = False
    watch_directory: bool = True

    # Zotero indexing settings
    enable_incremental_indexing: bool = True
    enable_deduplication: bool = True
    track_file_modifications: bool = True
    indexing_state_path: Path = Path("D:/Claude Code/scientific-papers-mcp/data/indexing_state.json")
    batch_indexing_size: int = 50  # Process 50 documents at once

    # Logging
    log_level: str = "INFO"

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def validate_paths(self) -> None:
        """Validate that required paths exist and cloud credentials if needed"""
        # Validate local paths (always needed for Zotero documents)
        if not self.documents_path.exists():
            raise FileNotFoundError(
                f"Documents path does not exist: {self.documents_path}"
            )

        # Cloud mode: validate Chroma Cloud credentials
        if self.use_chroma_cloud:
            if not self.chroma_api_key:
                raise ValueError(
                    "Chroma Cloud API key is required when use_chroma_cloud=True. "
                    "Set CHROMA_API_KEY environment variable or configure in .env file."
                )
            if not self.chroma_tenant:
                raise ValueError(
                    "Chroma Cloud tenant is required when use_chroma_cloud=True. "
                    "Set CHROMA_TENANT environment variable or configure in .env file."
                )
            if not self.chroma_database:
                raise ValueError(
                    "Chroma Cloud database is required when use_chroma_cloud=True. "
                    "Set CHROMA_DATABASE environment variable or configure in .env file."
                )
        else:
            # Local mode: create chroma path if it doesn't exist
            self.chroma_path.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
