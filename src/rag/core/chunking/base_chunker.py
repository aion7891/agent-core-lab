"""
Base class for every chunking strategy.

To add a new strategy (sentence, semantic, recursive, ...), inherit from
BaseChunker and implement split_text(). The id/source handling is shared.
"""


class BaseChunker:
    def chunk(self, documents):
        """Turn a list of document dicts into a list of chunk dicts."""
        chunks = []

        for document in documents:
            for text in self.split_text(document["text"]):
                chunks.append({
                    "id": f"chunk_{len(chunks)}",
                    "source": document["source"],
                    "text": text,
                })

        return chunks

    def split_text(self, text):
        """Return a list of text pieces."""
        raise NotImplementedError
