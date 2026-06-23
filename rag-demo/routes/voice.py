import logging
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Header
import asyncpg
from openai import AsyncOpenAI

from db import get_db


async def transcribe_audio(audio_bytes: bytes) -> str:
    client = AsyncOpenAI()

    response = await client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.mp3", audio_bytes, "audio/mpeg")
    )
    return response.text

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/voice")
async def voice_question(
    file: UploadFile = File(...),
    user_id: str = Header(...),
    conn: asyncpg.Connection = Depends(get_db)
):
    try:
        if not file.filename.lower().endswith(".mp3"):
            raise HTTPException(
                status_code=400, detail="Only MP3 files are supported")

        audio_bytes = await file.read()
        text = await transcribe_audio(audio_bytes)
        logger.info(f"Transcribed: {text}")

        return {"transcribed_text": text}

    except Exception as e:
        logger.error(f"Voice error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Voice processing failed: {str(e)}")
