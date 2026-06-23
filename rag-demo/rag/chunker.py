import logging
from typing import List

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Utility for splitting documents into chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            # Move start position by chunk_size minus overlap
            start = end - self.overlap

        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    def chunk_pdf_text(self, text: str) -> List[str]:
        """
        Chunk extracted PDF text by paragraphs and length.

        Args:
            text: Extracted PDF text

        Returns:
            List of text chunks
        """
        # Split by double newlines (paragraphs)
        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        logger.info(f"Split PDF into {len(chunks)} chunks")
        return chunks
