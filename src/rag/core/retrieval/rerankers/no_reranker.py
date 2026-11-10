from src.rag.core.retrieval.rerankers.base_reranker import BaseReranker


class NoReranker(BaseReranker):
    """Keeps the retriever's order. Useful to compare against real rerankers."""

    def score(self, query, match):
        return match["score"]
