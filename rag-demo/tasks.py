from celery_app import celery_app
import asyncio
import asyncpg
import os

from rag.chunker import DocumentChunker
from rag.embeddings import EmbeddingService


@celery_app.task(name="process_document")
def process_document_task(doc_id: int, text_content: str):
    asyncio.run(_process_document(doc_id, text_content))


async def _process_document(doc_id: int, text_content: str):
    db_url = os.getenv("DATABASE_URL")
    conn = await asyncpg.connect(db_url)

    try:
        chunker = DocumentChunker()
        chunks = chunker.chunk_pdf_text(text_content)

        embedding_service = EmbeddingService()
        embeddings = await embedding_service.embed_batch(chunks)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            await conn.execute(
                """INSERT INTO document_chunks (document_id, chunk_text, chunk_index, embedding)
                VALUES ($1, $2, $3, $4)""",
                doc_id,
                chunk,
                i,
                f"[{','.join(map(str, embedding))}]"
            )
    finally:
        await conn.close()
