from src.rag.core.retrieval.rerankers.keyword_reranker import KeywordReranker
from src.rag.core.retrieval.rerankers.no_reranker import NoReranker
from src.rag.core.retrieval.retriever import Retriever
from src.rag.core.vector_stores.memory_store import MemoryStore
from tests.fakes import FakeEmbedder


def make_matches():
    return [
        {"id": "chunk_0", "source": "a", "text": "cats sleep a lot", "score": 0.80},
        {"id": "chunk_1", "source": "a", "text": "Remote work: the future!", "score": 0.78},
    ]


def test_keyword_reranker_rewards_shared_words():
    best = KeywordReranker().rerank("What is the future of remote work?", make_matches(), top_n=1)

    assert best[0]["id"] == "chunk_1"
    assert best[0]["rerank_score"] > 0.78


def test_reranker_does_not_change_input():
    matches = make_matches()
    KeywordReranker().rerank("remote work", matches, top_n=2)
    assert "rerank_score" not in matches[0]


def test_no_reranker_keeps_order():
    best = NoReranker().rerank("remote work", make_matches(), top_n=2)
    assert [m["id"] for m in best] == ["chunk_0", "chunk_1"]


def test_retriever_empty_store_returns_nothing():
    retriever = Retriever(FakeEmbedder(), MemoryStore())
    assert retriever.has_documents() is False
    assert retriever.retrieve("remote work") == []


def test_retriever_finds_similar_chunk():
    embedder = FakeEmbedder()
    store = MemoryStore()
    chunks = [
        {"id": "chunk_0", "source": "a", "text": "cat and dog"},
        {"id": "chunk_1", "source": "a", "text": "remote work from office"},
    ]
    store.add(chunks, embedder.embed_many([c["text"] for c in chunks]))

    results = Retriever(embedder, store).retrieve("remote work", top_k=1)
    assert results[0]["id"] == "chunk_1"
