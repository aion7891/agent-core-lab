"""
Fake LLM and embedder so tests run fast and without Ollama.
"""

import numpy as np

from src.llm.base_llm import BaseLLM
from src.rag.core.embeddings.base_embedder import BaseEmbedder


class FakeLLM(BaseLLM):
    """Returns the prepared replies in order, and records what it was asked."""

    def __init__(self, replies):
        super().__init__("fake")
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        reply = self.replies.pop(0)

        # a reply can be plain text or a dict with tool_calls
        if isinstance(reply, str):
            reply = {"content": reply, "tool_calls": []}

        return {**reply, "message": {"role": "assistant", "content": reply["content"]}}

    def tool_result_message(self, tool_call, result):
        return {"role": "tool", "tool_name": tool_call["name"], "content": str(result)}


class FakeEmbedder(BaseEmbedder):
    """Counts a few keywords, so texts with the same words get similar vectors."""

    KEYWORDS = ["remote", "work", "office", "cat", "dog"]

    def __init__(self):
        super().__init__("fake")

    def embed_many(self, texts):
        vectors = []
        for text in texts:
            words = text.lower().split()
            vectors.append(np.array([words.count(k) for k in self.KEYWORDS], dtype=np.float32))
        return vectors
