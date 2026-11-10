"""
The only place that knows which class to use for each setting.

Adding a new provider or strategy:
    1. create its class in the right folder
    2. add one `if name == ...` line in the matching build_* function
    3. change the setting in .env
"""

from src.llm.ollama_llm import OllamaLLM
from src.rag.core.chunking.word_chunker import WordChunker
from src.rag.core.embeddings.ollama_embedder import OllamaEmbedder
from src.rag.core.generation.grounded_generator import GroundedGenerator
from src.rag.core.retrieval.rerankers.keyword_reranker import KeywordReranker
from src.rag.core.retrieval.rerankers.no_reranker import NoReranker
from src.rag.core.retrieval.retriever import Retriever
from src.rag.core.vector_stores.memory_store import MemoryStore
from src.rag.pipelines.indexing_pipeline import IndexingPipeline


def build_llm(settings, model=None, think=False):
    name = settings["llm_provider"]
    model = model or settings["llm_model"]

    if name == "ollama":
        return OllamaLLM(model, host=settings["ollama_host"], think=think)

    raise ValueError(f"Unknown LLM provider: {name}")


def build_agent_llm(settings):
    return build_llm(settings, model=settings["agent_model"], think=settings["agent_think"])


def build_embedder(settings):
    name = settings["embed_provider"]

    if name == "ollama":
        return OllamaEmbedder(settings["embed_model"], host=settings["ollama_host"])

    raise ValueError(f"Unknown embedding provider: {name}")


def build_chunker(settings):
    name = settings["chunker"]

    if name == "word":
        return WordChunker(settings["chunk_size"], settings["chunk_overlap"])

    raise ValueError(f"Unknown chunker: {name}")


def build_store(settings):
    name = settings["vector_store"]

    if name == "memory":
        return MemoryStore()

    raise ValueError(f"Unknown vector store: {name}")


def build_reranker(settings, llm=None):
    name = settings["reranker"]

    if name == "keyword":
        return KeywordReranker()
    if name == "none":
        return NoReranker()

    raise ValueError(f"Unknown reranker: {name}")


def build_generator(settings, llm):
    name = settings["generator"]

    if name == "grounded":
        return GroundedGenerator(llm)

    raise ValueError(f"Unknown generator: {name}")


def build_rag_parts(settings):
    """Build every RAG building block, sharing one embedder and one store."""
    llm = build_llm(settings)
    embedder = build_embedder(settings)
    store = build_store(settings)

    return {
        "llm": llm,
        "pipeline": IndexingPipeline(build_chunker(settings), embedder, store),
        "retriever": Retriever(embedder, store),
        "reranker": build_reranker(settings, llm),
        "generator": build_generator(settings, llm),
    }
