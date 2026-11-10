"""
Offline indexing, shared by every RAG type:
    load files -> chunk -> embed -> store
"""

from src.rag.core.loaders.loader_registry import load_document
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IndexingPipeline:
    def __init__(self, chunker, embedder, store):
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    def run(self, file_paths, reset=True):
        """Index the files. Returns the number of chunks stored."""
        documents = self.load_documents(file_paths)

        chunks = self.chunker.chunk(documents)
        logger.info("Created %d chunks", len(chunks))

        if not chunks:
            logger.warning("No chunks created, nothing to index")
            return 0

        vectors = self.embedder.embed_many([chunk["text"] for chunk in chunks])

        if reset:
            self.store.clear()

        self.store.add(chunks, vectors)
        logger.info("Stored %d chunks in the vector store", self.store.count())

        return len(chunks)

    def load_documents(self, file_paths):
        documents = []

        for path in file_paths:
            document = load_document(path)

            if not document["text"].strip():
                logger.warning("Skipping empty document: %s", document["source"])
                continue

            word_count = len(document["text"].split())
            logger.info("Loaded %s (%d words)", document["source"], word_count)
            documents.append(document)

        return documents
