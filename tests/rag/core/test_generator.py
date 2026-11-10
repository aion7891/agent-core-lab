from src.rag.core.generation.grounded_generator import GroundedGenerator
from src.rag.prompts import NO_ANSWER
from tests.fakes import FakeLLM

CHUNKS = [{"id": "chunk_3", "source": "doc.pdf", "text": "Remote work is growing."}]


def test_generator_sends_context_and_question():
    llm = FakeLLM(["It is growing [chunk_3]."])
    answer = GroundedGenerator(llm).generate("Is remote work growing?", CHUNKS)

    user_message = llm.calls[0][1]["content"]
    assert answer == "It is growing [chunk_3]."
    assert "[chunk_3] Source: doc.pdf" in user_message
    assert "Is remote work growing?" in user_message


def test_no_chunks_skips_the_llm():
    llm = FakeLLM([])
    assert GroundedGenerator(llm).generate("anything", []) == NO_ANSWER
    assert llm.calls == []


def test_empty_llm_reply_falls_back():
    assert GroundedGenerator(FakeLLM([""])).generate("q", CHUNKS) == NO_ANSWER
