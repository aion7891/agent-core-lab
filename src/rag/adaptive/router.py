from src.rag.prompts import ROUTER_PROMPT

ROUTES = ["direct", "simple", "complex"]


class Router:
    """Decides how much work a question needs: direct, simple or complex."""

    def __init__(self, llm, default_route="simple"):
        self.llm = llm
        self.default_route = default_route

    def route(self, query):
        reply = self.llm.ask(ROUTER_PROMPT, query)
        label = reply.strip().lower().strip(".!\"' ")

        # small models sometimes add extra words, so fall back safely
        if label not in ROUTES:
            return self.default_route

        return label
