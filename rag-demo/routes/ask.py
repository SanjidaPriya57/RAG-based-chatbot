import logging
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
import asyncpg
from openai import AsyncOpenAI
from s3_client import get_image_url

from db import get_db
from rag.embeddings import EmbeddingService
from rag.retriever import DocumentRetriever
from memory.cache import get_cache, set_cache
from fastapi.responses import StreamingResponse

tools = [
    {
        "type": "function",
        "function": {
            "name": "create_budget_alert",
            "description": "Create a budget alert for the user when they want to be notified about spending in a category",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Spending category, e.g. food, transport"},
                    "amount": {"type": "number", "description": "Alert threshold amount"}
                },
                "required": ["category", "amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_budget_alerts",
            "description": "List all budget alerts the user has set up",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


async def get_conversation_history(conn, user_id: str, session_id: str, limit: int = 10) -> list:
    rows = await conn.fetch("""
        SELECT message, response FROM conversations
        WHERE user_id = $1 AND session_id = $2
        ORDER BY created_at DESC
        LIMIT $3
    """, user_id, session_id, limit)
    return list(reversed(rows))

logger = logging.getLogger(__name__)
router = APIRouter()


class AskRequest(BaseModel):
    query: str
    session_id: str
    top_k: int = 5


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, user_id: str = Header(..., convert_underscores=False), conn: asyncpg.Connection = Depends(get_db)):
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
        cache_key = f"answer:{user_id}:{request.session_id}:{query}"
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

        # if not retrieved_chunks:
        #     return AskResponse(
        #         query=query,
        #         answer="I don't have anything on that yet — feel free to upload a document or just ask me directly!",
        #         sources=[]
        #     )

        context = retriever.format_context(retrieved_chunks)

        sources = [chunk for chunk, _, _, _ in retrieved_chunks]
        image_keys = list({img_key for _, _, img_key,
                           is_img in retrieved_chunks if is_img and img_key})

        prompt = f"""You are a personal finance assistant embedded in a finance app. 
        You have access to the user's financial documents, transactions, and history.
        Answer naturally and conversationally like ChatGPT would.
        Use the context below to answer — but don't mention "documents" or "context" to the user.

        Context:
        {context}

        User's question: {query}"""

        # Format context from retrieved chunks
        history = await get_conversation_history(conn, user_id, request.session_id)
        messages = []
        for row in history:
            messages.append({"role": "user", "content": row["message"]})
            messages.append({"role": "assistant", "content": row["response"]})
        messages.append({"role": "user", "content": prompt})

        if image_keys:
            image_urls = [get_image_url(key) for key in image_keys]
            vision_content = [{"type": "text", "text": prompt}]
            for url in image_urls:
                vision_content.append(
                    {"type": "image_url", "image_url": {"url": url}})
            messages[-1] = {"role": "user", "content": vision_content}

        client = AsyncOpenAI()
        gpt_response = await client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=messages,
            tools=tools
        )
        response_message = gpt_response.choices[0].message

        if response_message.tool_calls:
            import json
            tool_call = response_message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)

            if tool_call.function.name == "create_budget_alert":
                await conn.execute(
                    "INSERT INTO budget_alerts (user_id, category, amount) VALUES ($1, $2, $3)",
                    user_id, args["category"], args["amount"]
                )
                answer = f"Got it — I'll alert you if your spending on {args['category']} goes over {args['amount']}."

            elif tool_call.function.name == "list_budget_alerts":
                alerts = await conn.fetch(
                    "SELECT category, amount FROM budget_alerts WHERE user_id = $1",
                    user_id
                )
                if alerts:
                    lines = [
                        f"- {a['category']}: {a['amount']}" for a in alerts]
                    answer = "Here are your current budget alerts:\n" + \
                        "\n".join(lines)
                else:
                    answer = "You don't have any budget alerts set up yet."

            else:
                answer = "Sorry, I couldn't complete that action."
        else:
            answer = response_message.content

        await conn.execute(
            "INSERT INTO conversations (user_id, session_id, message, response) VALUES ($1, $2, $3, $4)",
            user_id, request.session_id, query, answer
        )

        response = AskResponse(query=query, answer=answer, sources=sources)

        # Cache the answer
        await set_cache(cache_key, response.model_dump(), ttl=300)

        logger.info(f"Answer generated for query: {query}")
        return response

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process question: {str(e)}")


@router.post("/ask-stream")
async def ask_question_stream(request: AskRequest, user_id: str = Header(..., convert_underscores=False)):
    async def generate():
        client = AsyncOpenAI()
        stream = await client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{"role": "user", "content": request.query}],
            stream=True
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return StreamingResponse(generate(), media_type="text/plain")
