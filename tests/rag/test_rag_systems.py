from src.rag.adaptive.adaptive_rag import AdaptiveRAG
from src.rag.core.chunking.word_chunker import WordChunker
from src.rag.core.generation.grounded_generator import GroundedGenerator
from src.rag.core.retrieval.rerankers.keyword_reranker import KeywordReranker
from src.rag.core.retrieval.retriever import Retriever
from src.rag.core.vector_stores.memory_store import MemoryStore
from src.rag.prompts import NO_ANSWER
from src.rag.simple_rag import SimpleRAG
from tests.fakes import FakeEmbedder, FakeLLM


def make_retriever():
    embedder = FakeEmbedder()
    store = MemoryStore()
    documents = [{"source": "doc.pdf", "text": "remote work is the future of office life"}]
    chunks = WordChunker(chunk_size=50, overlap=5).chunk(documents)
    store.add(chunks, embedder.embed_many([c["text"] for c in chunks]))
    return Retriever(embedder, store)


def make_adaptive(replies):
    llm = FakeLLM(replies)
    rag = AdaptiveRAG(llm, make_retriever(), KeywordReranker(), GroundedGenerator(llm))
    return rag, llm


def test_simple_rag_answers():
    llm = FakeLLM(["The future is remote [chunk_0]."])
    rag = SimpleRAG(make_retriever(), KeywordReranker(), GroundedGenerator(llm))
    assert rag.answer("future of remote work?") == "The future is remote [chunk_0]."


def test_simple_rag_without_documents():
    rag = SimpleRAG(Retriever(FakeEmbedder(), MemoryStore()), KeywordReranker(), None)
    assert rag.answer("hi") == "Please index your documents first."


def test_adaptive_direct_route_skips_retrieval():
    rag, llm = make_adaptive(["direct", "Hello there!"])
    assert rag.answer("Hello") == "Hello there!"
    assert len(llm.calls) == 2


def test_adaptive_answers_when_evidence_is_enough():
    rag, _ = make_adaptive(["simple", "yes", "Remote [chunk_0]."])
    assert rag.answer("future of remote work?") == "Remote [chunk_0]."


def test_adaptive_retries_once_then_gives_up():
    # route, check(no), rewrite, check(no) -> no extra rewrite after last attempt
    rag, llm = make_adaptive(["Complex.", "no", "remote work future", "no"])
    assert rag.answer("future of remote work?") == NO_ANSWER
    assert len(llm.calls) == 4


def test_adaptive_unknown_route_falls_back_to_simple():
    rag, _ = make_adaptive(["something odd", "Yes.", "ok [chunk_0]"])
    assert rag.answer("remote work?") == "ok [chunk_0]"
