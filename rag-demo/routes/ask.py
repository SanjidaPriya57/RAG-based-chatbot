import logging
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
import asyncpg
from openai import AsyncOpenAI

from db import get_db
from rag.embeddings import EmbeddingService
from rag.retriever import DocumentRetriever
from memory.cache import get_cache, set_cache


async def get_conversation_history(conn, user_id: str, limit: int = 10) -> list:
    rows = await conn.fetch("""
        SELECT message, response FROM conversations
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT $2
    """, user_id, limit)
    return list(reversed(rows))  # oldest first

logger = logging.getLogger(__name__)
router = APIRouter()


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, user_id: str = Header(...), conn: asyncpg.Connection = Depends(get_db)):
    """
    Ask a question over the indexed documents using RAG.

    Args:
        request: Question and retrieval parameters
        conn: Database connection

    Returns:
        Answer and source documents
    """
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(
                status_code=400, detail="Query cannot be empty")

        # Check cache first
        cache_key = f"answer:{user_id}:{query}"
        cached_answer = await get_cache(cache_key)
        if cached_answer:
            logger.info(f"Cache hit for query: {query}")
            return AskResponse(**cached_answer)

        # Retrieve relevant chunks
        embedding_service = EmbeddingService()
        retriever = DocumentRetriever(embedding_service)

        retrieved_chunks = await retriever.retrieve(
            query=query,
            conn=conn,
            user_id=user_id,
            top_k=request.top_k
        )

        if not retrieved_chunks:
            return AskResponse(
                query=query,
                answer="No relevant information found in the indexed documents.",
                sources=[]
            )

        context = retriever.format_context(retrieved_chunks)

        prompt = f"""You are a helpful finance assistant. Answer based on the context below.

        Context:
        {context}

        Question: {query}"""

        # Format context from retrieved chunks
        history = await get_conversation_history(conn, user_id)
        messages = []
        for row in history:
            messages.append({"role": "user", "content": row["message"]})
            messages.append({"role": "assistant", "content": row["response"]})
        messages.append({"role": "user", "content": prompt})

        client = AsyncOpenAI()
        gpt_response = await client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=messages
        )
        answer = gpt_response.choices[0].message.content

        await conn.execute(
            "INSERT INTO conversations (user_id, message, response) VALUES ($1, $2, $3)",
            user_id, query, answer
        )

        sources = [chunk for chunk, _ in retrieved_chunks]

        response = AskResponse(query=query, answer=answer, sources=sources)

        # Cache the answer
        await set_cache(cache_key, response.model_dump(), ttl=300)

        logger.info(f"Answer generated for query: {query}")
        return response

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process question: {str(e)}")
