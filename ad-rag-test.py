from IPython.terminal.shortcuts.auto_match import braces

from src.adaptive_rag.ingestion import DocumentIngestion
from src.adaptive_rag.vector_store import VectorStore
from src.adaptive_rag.rag import AdaptiveRAG
from pathlib import Path

project_dir = Path(__file__).resolve().parent

# =====================================================================
# PHASE 1: PREPARE THE DOCUMENTS ONCE
# =====================================================================

file_paths = [project_dir / "sample-doc.pdf", project_dir / "sample_doc2.pdf"]

# step1
ingestion = DocumentIngestion()
documents = ingestion.ingest_files(file_paths)

# step2 : create chunks

chunks = ingestion.chunk_documents(documents,chunk_size=150,overlaps=30)

if not chunks:
    print("No chunks found")
    exit()

# step3 :
vector_store = VectorStore()
vector_store.index_chunks(chunks)


# # step4:
# query = "what is the future of remote work?"
#
# matches = vector_store.retrieve(query,top_k=5)
#
# # step 5 : rerank
#
# best_chunk = vector_store.rerank(query,matches,top_n=2)

# print("\n =========== selected chunks====================")
#
# for chunk in best_chunk:
#     print(f"\nID: {chunk['id']}")
#     print(f"Source: {chunk['source']}")
#     print(f"Similarity: {chunk['semantic_similarity']:.4f}")
#     print(f"Text: {chunk['text']}")


# =====================================================================
# PHASE 2: RUN ADAPTIVE RAG FOR EACH QUESTION
# =====================================================================
print(""" 
# =====================================================================
# PHASE 2: RUN ADAPTIVE RAG FOR EACH QUESTION
# =====================================================================
\n
""")

adaptive_rag = AdaptiveRAG(vector_store)
while True:

    question  = input("\n Ask a question or type exit: ")

    if question.strip().lower() == "exit":
        break

    answer = adaptive_rag.answer(question)
    print("\n================ ANSWER ================")
    print(answer)
