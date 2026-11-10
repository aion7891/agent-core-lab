"""
Run from the project root:
    uv run python -m scripts.run_simple_rag
"""

from pathlib import Path

from src.config import get_settings
from src.factory import build_rag_parts
from src.rag.simple_rag import SimpleRAG

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FILES = [DATA_DIR / "sample-doc.pdf", DATA_DIR / "sample_doc2.pdf"]


def main():
    parts = build_rag_parts(get_settings())

    # Phase 1: index the documents once
    if parts["pipeline"].run(FILES) == 0:
        print("No chunks found. Check the files in the data folder.")
        return

    # Phase 2: answer questions
    rag = SimpleRAG(parts["retriever"], parts["reranker"], parts["generator"])

    while True:
        question = input("\nAsk a question (or type exit): ")
        if question.strip().lower() == "exit":
            break
        print("\n================ ANSWER ================")
        print(rag.answer(question))


if __name__ == "__main__":
    main()
