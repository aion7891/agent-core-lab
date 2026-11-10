from src.rag.prompts import QUERY_REWRITE_PROMPT


class QueryRewriter:
    """Rewrites a question into a better search query."""

    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, query):
        rewritten = self.llm.ask(QUERY_REWRITE_PROMPT, query)
        return rewritten or query
