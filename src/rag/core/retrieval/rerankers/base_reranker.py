"""
Base class for every reranker (keyword, cross-encoder, LLM, Cohere, ...).

To add one, inherit from BaseReranker and implement score().
rerank() takes care of copying, sorting and cutting to top_n.
"""


class BaseReranker:
    def rerank(self, query, matches, top_n=3):
        """
        matches: list of dicts from the retriever (each has "score").
        Returns the best top_n matches, each with a "rerank_score" key.
        """
        reranked = []

        for match in matches:
            # copy so the retriever's results are never changed
            new_match = dict(match)
            new_match["rerank_score"] = self.score(query, match)
            reranked.append(new_match)

        reranked.sort(key=lambda match: match["rerank_score"], reverse=True)
        return reranked[:top_n]

    def score(self, query, match):
        """Return a number. Higher means more relevant."""
        raise NotImplementedError
