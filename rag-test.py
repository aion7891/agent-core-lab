from src.rag_system import SimpleRAG
import os


rag = SimpleRAG()

#1. run the Multi-Format RAG Pipeline
docs = rag.ingest_files(['sample-doc.pdf','sample_doc2.pdf'])


#step2 chunk all normalised doc
chunks = rag.chunk_text([docs[1]['text']],chunk_size=25,overlap=5)

# step3 :
rag.index_chunk(chunks)



print(""" 

    # =====================================================================
    # PHASE 2: ONLINE RETRIEVAL & GENERATION (Real-Time)
    # =====================================================================

""")
# Phase 2
user_query = "what is the future of remote work ?"

# retrival
raw_matches = rag.retrieve(user_query,top_k=5)

print('\n RAW matches: \n')
print(raw_matches)

# rerank
best_context = rag.rerank(user_query,raw_matches,top_n=2)

print('\n BEST context: ')
print(best_context)

# build augment  and generate the answer
final_prompt = rag.build_augmented_prompt(user_query,best_context)

# print(final_prompt)

answer = rag.generate_answer(final_prompt)

print("\n================ FINAL GROUNDED RESPONSE ================")
print(answer)

