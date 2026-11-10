"""
Simple RAG:
    Question -> Retrieve -> Rerank -> Generate
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SimpleRAG:
    def __init__(self, retriever, reranker, generator, top_k=5, top_n=2):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.top_k = top_k
        self.top_n = top_n

    def answer(self, query):
        query = query.strip()

        if not query:
            return "Please enter a question."

        if not self.retriever.has_documents():
            return "Please index your documents first."

        matches = self.retriever.retrieve(query, top_k=self.top_k)
        best_chunks = self.reranker.rerank(query, matches, top_n=self.top_n)

        for chunk in best_chunks:
            logger.info("Selected %s from %s (rerank %.4f)",
                        chunk["id"], chunk["source"], chunk["rerank_score"])

        return self.generator.generate(query, best_chunks)
