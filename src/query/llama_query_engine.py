"""LlamaIndex Query Engine - Advanced RAG capabilities on existing ChromaDB

Integrates LlamaIndex with your existing ChromaDB collection for advanced queries:
- Sub-question decomposition
- Multi-document comparison
- Citation tracking
- Router query engine

Uses your existing:
- ChromaDB vector store
- Voyage AI embeddings
- Cohere reranking
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Try to import LlamaIndex
try:
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.query_engine import SubQuestionQueryEngine, RouterQueryEngine
    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.voyageai import VoyageEmbedding
    from llama_index.core.postprocessor import CohereRerank
    from llama_index.core.response_synthesizers import ResponseMode
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    logger.warning(
        "LlamaIndex not installed. Install with: pip install llama-index llama-index-vector-stores-chroma "
        "llama-index-embeddings-voyageai llama-index-postprocessor-cohere"
    )


class LlamaIndexQueryEngine:
    """
    Advanced query engine using LlamaIndex on your existing ChromaDB

    Features:
    - Sub-question decomposition for complex queries
    - Multi-document comparison
    - Citation tracking with source nodes
    - Router query engine for different query types

    Connects to your existing:
    - ChromaDB collection (scientific_papers_voyage)
    - Voyage AI embeddings
    - Cohere reranking
    """

    def __init__(
        self,
        chroma_client,
        collection_name: str,
        voyage_api_key: str,
        voyage_model: str = "voyage-context-3",
        cohere_api_key: Optional[str] = None,
        cohere_model: str = "rerank-v3.5",
        similarity_top_k: int = 50,
        rerank_top_n: int = 10,
    ):
        """
        Initialize LlamaIndex query engine on existing ChromaDB

        Args:
            chroma_client: Your existing ChromaDB client
            collection_name: Collection name (e.g., "scientific_papers_voyage")
            voyage_api_key: Voyage AI API key
            voyage_model: Voyage model name
            cohere_api_key: Cohere API key (optional, for reranking)
            cohere_model: Cohere rerank model
            similarity_top_k: Number of results to retrieve
            rerank_top_n: Number of results after reranking
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex not installed. Install with:\n"
                "pip install llama-index llama-index-vector-stores-chroma "
                "llama-index-embeddings-voyageai llama-index-postprocessor-cohere"
            )

        self.chroma_client = chroma_client
        self.collection_name = collection_name
        self.similarity_top_k = similarity_top_k
        self.rerank_top_n = rerank_top_n

        # Setup Voyage AI embeddings (same as your custom stack)
        logger.info(f"Setting up Voyage AI embeddings: {voyage_model}")
        embed_model = VoyageEmbedding(
            model_name=voyage_model,
            voyage_api_key=voyage_api_key,
        )

        # Configure global LlamaIndex settings
        Settings.embed_model = embed_model
        Settings.chunk_size = 1024  # Same as your config
        Settings.chunk_overlap = 100

        # Connect to existing ChromaDB collection
        logger.info(f"Connecting to ChromaDB collection: {collection_name}")
        chroma_collection = chroma_client.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Create index from existing vector store
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

        # Setup postprocessors
        self.postprocessors = []
        if cohere_api_key:
            logger.info(f"Setting up Cohere reranking: {cohere_model}")
            self.postprocessors.append(
                CohereRerank(
                    api_key=cohere_api_key,
                    model=cohere_model,
                    top_n=rerank_top_n,
                )
            )

        # Create base query engine
        self.base_query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            node_postprocessors=self.postprocessors,
            response_mode=ResponseMode.COMPACT,  # Compact with citations
        )

        # Create sub-question query engine (for complex queries)
        query_engine_tools = [
            QueryEngineTool(
                query_engine=self.base_query_engine,
                metadata=ToolMetadata(
                    name="scientific_papers",
                    description=(
                        "Search scientific papers about climate change, glacier monitoring, "
                        "remote sensing, and machine learning applications. "
                        "Useful for answering questions about research methods, findings, "
                        "comparisons between studies, and technical approaches."
                    ),
                ),
            )
        ]

        self.sub_question_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools,
            verbose=True,
        )

        logger.info("LlamaIndex query engine initialized successfully")

    def query(
        self,
        question: str,
        use_sub_questions: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Query with LlamaIndex

        Args:
            question: User question
            use_sub_questions: Use sub-question decomposition for complex queries
            verbose: Show detailed reasoning

        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"Querying: {question[:100]}...")

        # Choose query engine
        if use_sub_questions:
            logger.info("Using sub-question query engine")
            query_engine = self.sub_question_engine
        else:
            query_engine = self.base_query_engine

        # Execute query
        response = query_engine.query(question)

        # Extract sources with scores and citations
        sources = []
        for node in response.source_nodes:
            sources.append({
                "text": node.text,
                "score": node.score if hasattr(node, 'score') else None,
                "metadata": node.metadata,
                "page_label": node.metadata.get("page_label"),
                "file_name": node.metadata.get("file_name"),
            })

        result = {
            "answer": str(response),
            "sources": sources,
            "num_sources": len(sources),
            "metadata": {
                "use_sub_questions": use_sub_questions,
                "query_length": len(question),
            }
        }

        # Add sub-questions if available
        if hasattr(response, 'metadata') and 'sub_questions' in response.metadata:
            result["sub_questions"] = response.metadata['sub_questions']

        logger.info(f"Query completed: {len(sources)} sources retrieved")

        return result

    def compare_papers(
        self,
        paper_titles_or_dois: List[str],
        comparison_aspect: str,
    ) -> Dict[str, Any]:
        """
        Compare multiple papers on a specific aspect

        Args:
            paper_titles_or_dois: List of paper titles or DOIs to compare
            comparison_aspect: What to compare (e.g., "methodology", "results", "datasets")

        Returns:
            Comparison results with citations
        """
        # Build comparison query
        papers_str = ", ".join(f'"{p}"' for p in paper_titles_or_dois)
        question = (
            f"Compare the {comparison_aspect} in the following papers: {papers_str}. "
            f"Highlight similarities, differences, and unique contributions of each paper."
        )

        # Use sub-question engine for complex comparison
        return self.query(question, use_sub_questions=True)

    def find_related_papers(
        self,
        paper_title_or_doi: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Find papers related to a given paper

        Args:
            paper_title_or_doi: Title or DOI of reference paper
            top_k: Number of related papers to find

        Returns:
            List of related papers with similarity scores
        """
        question = (
            f"Find papers related to '{paper_title_or_doi}'. "
            f"Focus on papers with similar methods, research questions, or findings."
        )

        # Adjust similarity_top_k for this query
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k * 2,  # Retrieve more, rerank to top_k
            node_postprocessors=self.postprocessors,
            response_mode=ResponseMode.TREE_SUMMARIZE,
        )

        response = query_engine.query(question)

        return {
            "reference_paper": paper_title_or_doi,
            "related_papers": [
                {
                    "title": node.metadata.get("title", "Unknown"),
                    "authors": node.metadata.get("authors", []),
                    "year": node.metadata.get("year"),
                    "similarity_score": node.score if hasattr(node, 'score') else None,
                    "excerpt": node.text[:200] + "...",
                }
                for node in response.source_nodes[:top_k]
            ],
            "num_related": len(response.source_nodes[:top_k]),
        }

    @staticmethod
    def is_available() -> bool:
        """Check if LlamaIndex is available"""
        return LLAMAINDEX_AVAILABLE
