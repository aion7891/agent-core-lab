"""
First step of retrieval: turn the question into a vector
and find the most similar chunks in the vector store.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store

    def has_documents(self):
        return self.store.count() > 0

    def retrieve(self, query, top_k=5):
        if not self.has_documents():
            logger.warning("Vector store is empty, nothing to retrieve")
            return []

        query_vector = self.embedder.embed(query)
        matches = self.store.search(query_vector, top_k=top_k)

        for match in matches:
            logger.debug("Retrieved %s (score %.4f)", match["id"], match["score"])

        return matches
