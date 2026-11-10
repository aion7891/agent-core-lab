import pytest

from src.rag.core.chunking.word_chunker import WordChunker


def test_chunks_overlap_and_cover_all_words():
    chunker = WordChunker(chunk_size=4, overlap=2)
    pieces = chunker.split_text("a b c d e f g")

    assert pieces == ["a b c d", "c d e f", "e f g"]


def test_short_text_gives_one_chunk():
    chunker = WordChunker(chunk_size=10, overlap=2)
    assert chunker.split_text("only three words") == ["only three words"]


def test_empty_text_gives_no_chunks():
    assert WordChunker(chunk_size=5, overlap=1).split_text("   ") == []


def test_chunk_ids_and_sources():
    documents = [
        {"source": "a.pdf", "text": "one two three"},
        {"source": "b.pdf", "text": "four five six"},
    ]
    chunks = WordChunker(chunk_size=3, overlap=1).chunk(documents)

    assert [c["id"] for c in chunks] == ["chunk_0", "chunk_1"]
    assert [c["source"] for c in chunks] == ["a.pdf", "b.pdf"]


@pytest.mark.parametrize("size, overlap", [(0, 0), (5, 5), (5, -1)])
def test_bad_settings_raise(size, overlap):
    with pytest.raises(ValueError):
        WordChunker(chunk_size=size, overlap=overlap)
