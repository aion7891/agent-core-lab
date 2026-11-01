from src.doc_processor import  MultiFormatParser
import os
import ollama
import numpy as np

class SimpleRAG:
    def __init__(self, embed_model = 'nomic-embed-text' , llm_model = 'gemma3:4b'):
        self.embed_model = embed_model
        self.llm_model = llm_model
        # In-memory "Vector Database" (stores dicts with id, text, and vector)
        self.vector_db = []
        self.parser = MultiFormatParser()

    # =====================================================================
    # PHASE 1: OFFLINE INDEXING (Data Preparation)
    # =====================================================================
    # --- STEP 1: Ingest Multiple File Formats ---
    def ingest_files(self,file_paths):
        print("\n--- [Step 1] Ingesting & Normalizing Multi-Format Files ---")
        normalised_docs = []
        for file_path in file_paths:
            doc_obj = self.parser.parse(file_path)
            normalised_docs.append(doc_obj)
            print(f"  Parsed '{doc_obj['source']}' ({doc_obj['file_type']}) -> {len(doc_obj['text'].split())} words")
        return normalised_docs

    def chunk_text(self, documents , chunk_size=15, overlap=5):
        print(f"\n--- [Step 2] Chunking (size={chunk_size} words, overlap={overlap} words) ---")
        chunks = []

        for doc in documents:
            words = doc.split()
            step = chunk_size - overlap

            for i in range (0, len(words), step):
                chunk = " ".join(words[i:i+chunk_size])
                if chunk :
                    chunks.append(chunk)

        for idx ,c in enumerate(chunks):
            print(f"  Chunk {idx}: '{c}'")
        return chunks

    def get_embedding(self, text):
        print("\n--- [Step 3] Getting Emedding ---")
        response = ollama.embed(model=self.embed_model, input=text)
        vector = np.array(response["embeddings"][0], dtype=np.float32)
        return vector

    def index_chunk(self, chunks):
        print("\n--- [Step 4] Indexing Chunks ---")
        for idx, chunk in enumerate(chunks):
            print(f"   Indexing the Chunk {idx}: '{chunk}'.........")
            vector = self.get_embedding(chunk)
            self.vector_db.append({
                "id": f"chunk-{idx}",
                "text": chunk,
                "vector": vector,
            })
        print(f"Stored {len(self.vector_db)} chunks in Vector DB (Vector dimension: {len(vector)}).")

    # =====================================================================
    # PHASE 2: ONLINE RETRIEVAL & GENERATION (Real-Time)
    # =====================================================================

    def _cosine_similarity(self, vector1, vector2):
        """ helper function to calculate cosine similarity between two vectors """

        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        return float(dot_product / (norm_vector1 * norm_vector2))


    def retrieve(self,query:str, top_k:int=5) -> list[dict]:

        print(f"\n--- [Step 4 & 5] Embedding Query & Searching Vector DB (Top-{top_k}) ---")
        query_vector = self.get_embedding(query)

        scored_chunks = []
        for record in self.vector_db:
            score = self._cosine_similarity(query_vector, record["vector"])
            scored_chunks.append({
                'id' : record["id"],
                'text' : record["text"],
                'semantic_score': score,
            })

        scored_chunks.sort(key=lambda x: x['semantic_score'], reverse=True)
        top_matches = scored_chunks[:top_k]

        for match in top_matches:
            print(f"  [{match['id']}] Cosine Score: {match['semantic_score']:.4f} | '{match['text']}'")

        return top_matches

    def rerank(self, query:str, retrieved_chunks:list[dict] , top_n: int =2 ) -> list[dict]:
        print(f"\n--- [Step 6] Reranking Candidates (Selecting Top-{top_n}) ---")

        query_words = set(query.lower().replace("?","").split())

        for item in retrieved_chunks:
            chunk_words = set(item["text"].lower().replace(".","").split())
            keyword_overlap = len(query_words.intersection(chunk_words))

            item['rerank_score'] = item["semantic_score"] + (0.05 * keyword_overlap)

        retrieved_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        return retrieved_chunks[:top_n]


    def build_augmented_prompt(self,query,selected_chunks):
        print("\n--- [Step 7a] Augmenting Prompt with Retrieved Context ---")

        context_block = "\n".join(
            [f"[{c['id']}]: {c['text']}" for c in selected_chunks]
        )

        RAG_PROMPT = f"""You are a precise assistant. Answer the user's question using ONLY the context provided below.
            Always cite the chunk ID (e.g., [chunk_0]) at the end of the sentence. If the answer is not in the context, say "I don't know."
            
            CONTEXT:
            {context_block}
            
            USER QUESTION:
            {query}
            
            ANSWER:"""

        print(RAG_PROMPT)

        return RAG_PROMPT


    def generate_answer(self,augmented_prompt:str) -> str:
        print("\n--- [Step 7b] Generating Grounded Response from Ollama LLM ---")

        response = ollama.chat(
            model = self.llm_model,
            messages = [
                {"role":"user", "content":augmented_prompt},
            ],
        )

        # critique
        # this will check the response :
        return response['message']['content']





