import logging
from typing import List, Tuple
from rag.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Retriever for semantic search over document chunks."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    async def retrieve(
        self,
        query: str,
        conn,
        user_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        query_embedding = await self.embedding_service.embed_text(query)

        rows = await conn.fetch("""
            SELECT dc.chunk_text, 
                1 - (dc.embedding <=> $1::vector) as similarity
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE d.user_id = $2
            AND 1 - (dc.embedding <=> $1::vector) >= $3
            ORDER BY dc.embedding <=> $1::vector
            LIMIT $4
        """, f"[{','.join(map(str, query_embedding))}]", user_id, similarity_threshold, top_k)

        return [(row["chunk_text"], row["similarity"]) for row in rows]

    def format_context(self, chunks: List[Tuple[str, float]]) -> str:
        """
        Format retrieved chunks into context string.

        Args:
            chunks: List of (chunk_text, similarity_score) tuples

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant documents found."

        context = "Relevant information from documents:\n\n"
        for i, (chunk, score) in enumerate(chunks, 1):
            context += f"[Document {i} - Confidence: {score:.2f}]\n{chunk}\n\n"

        return context
