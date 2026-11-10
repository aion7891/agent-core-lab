import numpy as np
import ollama

from src.rag.core.embeddings.base_embedder import BaseEmbedder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model="nomic-embed-text", host=None, batch_size=32):
        super().__init__(model)
        self.batch_size = batch_size
        self.client = ollama.Client(host=host)

    def embed_many(self, texts):
        vectors = []

        # send texts in small batches so large documents do not overload Ollama
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
            logger.debug("Embedded %d / %d texts", len(vectors), len(texts))

        return vectors

    def _embed_batch(self, batch):
        try:
            response = self.client.embed(model=self.model, input=batch)
        except Exception as error:
            raise RuntimeError(
                f"Ollama embedding failed for model '{self.model}': {error}"
            ) from error

        return [np.array(vector, dtype=np.float32) for vector in response["embeddings"]]
