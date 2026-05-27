""""
This file implements the RAG component of the project. The RAG_Service class creates a datastore with chromadb, embeds the documents with all-mpnet-base-v2, 
and returns the 3 most similar examples to the user's query.
"""

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class RAG_Service:
    def __init__(self, data_path: str, collection_name: str = "sciq_examples", similarity_threshold=0.3):
        print(f"Initializing RAG Service with data from: {data_path}")
        self.similarity_threshold = similarity_threshold # only examples with more than 70% similarity are used
        self.data_path = data_path
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

        if self.collection.count() == 0:
            self._ingest_data()
        else:
            print(f"Collection '{collection_name}' already exists with {self.collection.count()} entries.")

    def _ingest_data(self):
        """Reads CSV, embeds support text, and stores metadata."""
        print("Ingesting data into ChromaDB...")
        try:
            df = pd.read_csv(self.data_path)
            df = df.dropna(subset=['support', 'question', 'correct_answer'])

            documents = []
            metadatas = []
            ids = []

            for idx, row in df.iterrows():
                documents.append(row["support"])
                metadatas.append({
                    "question_id": str(row.get("question_id", idx)),
                    "question": row["question"],
                    "correct_answer": row["correct_answer"],
                    "distractor1": row["distractor1"],
                    "distractor2": row["distractor2"],
                    "distractor3": row["distractor3"]
                })
                ids.append(str(idx))

            embeddings = self.embedding_model.encode(documents).tolist()  # why do we need to transform them into a list?

            self.collection.add(
                embeddings=embeddings,
                documents=documents, 
                metadatas=metadatas,
                ids=ids
            )

            print(f"Successfully ingested {len(ids)} documents.")

        except Exception as e:
            print(f"Error during ingestion: {e}")

    def retrieve_examples(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:  # why this datatype
        """Finds the top_n most similar examples to the query_text."""
        query_embedding = self.embedding_model.encode([query_text]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        examples = []
        if results["documents"]:
            for i in range(len(results["documents"][0])):
                distance = results["distances"][0][i]
                if distance >= self.similarity_threshold:  # filtering out the examples that are not similar enough, therefore not relevant
                    continue
                doc = results["documents"][0][i]
                meta = results["metadatas"][0][i]

                example = {
                    "support": doc,
                    "question": meta["question"],
                    "correct_answer": meta["correct_answer"],
                    "distractors": [
                        meta["distractor1"],
                        meta["distractor2"],
                        meta["distractor3"]
                    ]
                }

                examples.append(example)
        
        return examples




# underline the methods of collection
# check what these methods return

# don't we need the ids of the question?