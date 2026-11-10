from src.doc_processor import MultiFormatParser



class DocumentIngestion:

    def __init__(self):
        # doc parser
        self.parser = MultiFormatParser()

    # =====================================================================
    # STEP 1: READ DOCUMENTS and extract the text
    # =====================================================================

    def ingest_files(self,file_paths):
        documents = []
        for file_path in file_paths:

            document = self.parser.parse(file_path)

            if not document['text'].strip():
                print(f"Skipping empty document : {document['source']}")
                continue
            documents.append(document)

            print(f" Read : {document['source']} : \n {len(document['text'])} words")

        return documents

    # =====================================================================
    # STEP 2: SPLIT DOCUMENTS INTO OVERLAPPING CHUNKS
    # =====================================================================

    def chunk_documents(self, documents, chunk_size =150, overlaps =30):

        if chunk_size <= 0:
            raise ValueError("Chunk size must be > 0")

        if overlaps < 0 or overlaps >= chunk_size:
            raise ValueError("Chunk size must be between 0 and chunk_size -1")


        chunks = []
        # example : size 150 and overlap 30 , step 120
        step = chunk_size - overlaps

        for document in documents:
            print(f"Chunking : {document['source']}")

            words = document['text'].lower().split()

            for start in range(0, len(words), step):
                end = start + chunk_size

                chunks.append({
                    "id" : f"chunk_{len(chunks)}",
                    "source" : document['source'],
                    'text' : " ".join(words[start:end]),
                })
                # stop after reaching the end of this doc
                if end >= len(words):
                    break
        print(f"Created {len(chunks)} chunks")
        return chunks