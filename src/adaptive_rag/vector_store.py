'''
1. Convert each chunk into an embedding.
2. Store the chunk and its embedding.
3. Find chunks similar to a question.
4. Rerank the results.
'''

import numpy as np
import ollama

class VectorStore:
    def __init__(self,embed_model='nomic-embed-text'):
        # Use teh embedding models
        self.embed_model = embed_model

        # in-memory vector db
        self.records = []

    # =====================================================================
    # STEP 1: CONVERT TEXT INTO AN EMBEDDING
    # =====================================================================

    def get_embedding(self,text):

        response = ollama.embed(
            model = self.embed_model,
            input = text,
        )
        return np.array(response['embeddings'][0])


    # =====================================================================
    # STEP 2: EMBED AND STORE THE CHUNKS
    # =====================================================================

    def index_chunks(self,chunks):

        new_records = []

        for chunk in chunks:
            vector = self.get_embedding(chunk['text'])

            new_records.append({
                'id': chunk['id'],
                'source': chunk['source'],
                'text': chunk['text'],
                'vector': vector,
            })

            print(f"Indexed {chunk['id']} form {chunk['source']}")

        # replace the db with the newly indexed chunks

        self.records = new_records

        print(f"\n Stored {len(self.records)} chunks ")


    # =====================================================================
    # STEP 3: CALCULATE COSINE SIMILARITY
    # =====================================================================

    def cosine_similarity(self,vector1,vector2):

        dot_product = np.dot(vector1,vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)

        denominator = norm_vector1 * norm_vector2

        if denominator == 0:
            return 0.0

        return float( dot_product / denominator)

    # =====================================================================
    # STEP 4: RETRIEVE THE MOST SIMILAR CHUNKS
    # =====================================================================

    def retrieve(self,query, top_k=5):

        if not self.records:
            return []

        # conver query to embedding
        query_embedding = self.get_embedding(query)

        matches = []

        for record in self.records:

            score = self.cosine_similarity(query_embedding,record['vector'])

            matches.append({
                'id': record['id'],
                'source': record['source'],
                'text': record['text'],
                'semantic_similarity': score,
            })


        # put the highest similarity score first.

        matches.sort(key=lambda x: x['semantic_similarity'], reverse=True)

        return matches[:top_k]


    # =====================================================================
    # STEP 5: RERANK THE RETRIEVED CHUNKS
    # =====================================================================

    def rerank(self,query, matches, top_n=2):

        # remove punctuation before comparing
        punctuation = str.maketrans("", "", ".,?!:;()\"'")

        query_words = set(query.lower().translate(punctuation).split())

        for match in matches:

            chunk_words = set(match['text'].lower().translate(punctuation).split())
            common_words = query_words.intersection(chunk_words)

            match['rerank_score'] = (match['semantic_similarity']  + 0.05*len(common_words))


        matches.sort(key=lambda x: x['rerank_score'], reverse=True)
        return matches[:top_n]