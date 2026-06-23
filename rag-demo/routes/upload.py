import io
import logging
from fastapi import APIRouter, File, Header, UploadFile, Depends, HTTPException
import asyncpg
import pypdf

from db import get_db
from rag.chunker import DocumentChunker
from rag.embeddings import EmbeddingService
from memory.cache import set_cache
import base64
from openai import AsyncOpenAI
import pandas as pd

logger = logging.getLogger(__name__)
router = APIRouter()


async def extract_text_from_image(image_bytes: bytes, media_type: str) -> str:
    client = AsyncOpenAI()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": "Extract all financial data, numbers, dates, and text from this image. Return plain text only."
                }
            ]
        }]
    )
    return response.choices[0].message.content


async def extract_text_from_excel(file_bytes: bytes) -> str:
    df = pd.read_excel(io.BytesIO(file_bytes))
    return df.to_string(index=False)


async def extract_text_from_csv(file_bytes: bytes) -> str:
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df.to_string(index=False)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), user_id: str = Header(...), conn: asyncpg.Connection = Depends(get_db)):
    """
    Upload and ingest a PDF document.

    Args:
        file: PDF file to upload
        conn: Database connection

    Returns:
        Document metadata and chunk count
    """
    try:
        if not file.filename.lower().endswith((".pdf", ".jpg", ".jpeg", ".png", ".heic", ".webp", ".xlsx", ".csv")):
            raise HTTPException(
                status_code=400, detail="Only PDF, image, and Excel/CSV files are supported")
        content = await file.read()

        # Extract text from PDF
        filename = file.filename.lower()

        if filename.endswith(".pdf"):
            pdf_reader = pypdf.PdfReader(io.BytesIO(content))
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
        elif filename.endswith((".jpg", ".jpeg")):
            text_content = await extract_text_from_image(content, "image/jpeg")
        elif filename.endswith(".png"):
            text_content = await extract_text_from_image(content, "image/png")
        elif filename.endswith(".heic"):
            text_content = await extract_text_from_image(content, "image/heic")
        elif filename.endswith(".webp"):
            text_content = await extract_text_from_image(content, "image/webp")
        elif filename.endswith(".xlsx"):
            text_content = await extract_text_from_excel(content)
        elif filename.endswith(".csv"):
            text_content = await extract_text_from_csv(content)

        if not text_content.strip():
            raise HTTPException(
                status_code=400, detail="No content found in file")

        # Store document metadata
        doc_id = await conn.fetchval(
            "INSERT INTO documents (user_id, filename, content) VALUES ($1, $2, $3) RETURNING id",
            user_id,
            file.filename,
            text_content
        )

        # Chunk document
        chunker = DocumentChunker()
        chunks = chunker.chunk_pdf_text(text_content)

        # Generate embeddings
        embedding_service = EmbeddingService()
        embeddings = await embedding_service.embed_batch(chunks)

        # Store chunks with embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            await conn.execute(
                """INSERT INTO document_chunks (document_id, chunk_text, chunk_index, embedding)
                VALUES ($1, $2, $3, $4)""",
                doc_id,
                chunk,
                i,
                f"[{','.join(map(str, embedding))}]"
            )

        # Cache document info
        await set_cache(
            f"doc:{doc_id}",
            {
                "id": doc_id,
                "filename": file.filename,
                "chunks": len(chunks),
                "status": "indexed"
            },
            ttl=3600
        )

        logger.info(
            f"Document {file.filename} uploaded with ID {doc_id}, {len(chunks)} chunks created")

        return {
            "status": "success",
            "document_id": doc_id,
            "filename": file.filename,
            "chunks": len(chunks),
            "message": f"Document uploaded and indexed with {len(chunks)} chunks"
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/documents")
async def list_documents(user_id: str = Header(...), conn: asyncpg.Connection = Depends(get_db)):
    documents = await conn.fetch(
        "SELECT id, filename, uploaded_at FROM documents WHERE user_id = $1 ORDER BY uploaded_at DESC",
        user_id
    )
    return {"documents": [dict(doc) for doc in documents]}
