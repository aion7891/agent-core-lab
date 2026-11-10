import numpy as np
import pytest

from src.rag.core.vector_stores.memory_store import MemoryStore


def make_store():
    store = MemoryStore()
    chunks = [
        {"id": "chunk_0", "source": "a", "text": "x"},
        {"id": "chunk_1", "source": "a", "text": "y"},
        {"id": "chunk_2", "source": "a", "text": "zero"},
    ]
    vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])]
    store.add(chunks, vectors)
    return store


def test_search_returns_best_first_with_score():
    results = make_store().search(np.array([1.0, 0.1]), top_k=2)

    assert [r["id"] for r in results] == ["chunk_0", "chunk_1"]
    assert results[0]["score"] > results[1]["score"]
    assert "vector" not in results[0]


def test_zero_vector_scores_zero():
    results = make_store().search(np.array([1.0, 0.0]), top_k=3)
    assert results[-1] == {"id": "chunk_2", "source": "a", "text": "zero", "score": 0.0}


def test_count_and_clear():
    store = make_store()
    assert store.count() == 3
    store.clear()
    assert store.count() == 0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        MemoryStore().add([{"id": "a", "source": "s", "text": "t"}], [])
