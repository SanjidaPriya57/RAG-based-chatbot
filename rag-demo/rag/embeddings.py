import os
import logging
from typing import List
from openai import AsyncOpenAI
import numpy as np

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "text-embedding-3-small"  # 1536 dimensions, cheap and fast

    async def embed_text(self, text: str) -> List[float]:
        response = await self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        embeddings = [item.embedding for item in response.data]
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return embeddings

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:

        arr1, arr2 = np.array(vec1), np.array(vec2)
        dot = np.dot(arr1, arr2)
        mag = np.linalg.norm(arr1) * np.linalg.norm(arr2)
        return float(dot / mag) if mag != 0 else 0.0
