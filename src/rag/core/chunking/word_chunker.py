from src.rag.core.chunking.base_chunker import BaseChunker


class WordChunker(BaseChunker):
    """
    Splits text into fixed-size word windows that overlap.
    Example: size 150, overlap 30 -> each new chunk starts 120 words later.
    """

    def __init__(self, chunk_size=150, overlap=30):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be between 0 and chunk_size - 1")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text):
        words = text.split()
        step = self.chunk_size - self.overlap
        pieces = []

        for start in range(0, len(words), step):
            end = start + self.chunk_size
            pieces.append(" ".join(words[start:end]))

            # stop once this window reached the end of the text
            if end >= len(words):
                break

        return pieces
