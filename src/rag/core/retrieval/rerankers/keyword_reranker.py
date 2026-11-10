from src.rag.core.retrieval.rerankers.base_reranker import BaseReranker

PUNCTUATION = str.maketrans("", "", ".,?!:;()\"'")


class KeywordReranker(BaseReranker):
    """
    Adds a small bonus for every word shared by the question and the chunk.
    final score = semantic score + keyword_weight * shared words
    """

    def __init__(self, keyword_weight=0.05):
        self.keyword_weight = keyword_weight

    def score(self, query, match):
        shared_words = self._words(query) & self._words(match["text"])
        return match["score"] + self.keyword_weight * len(shared_words)

    def _words(self, text):
        return set(text.lower().translate(PUNCTUATION).split())
