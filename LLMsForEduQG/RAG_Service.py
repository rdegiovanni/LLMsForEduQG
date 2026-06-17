""""
This file implements the RAG component of the project. The RAG_Service class creates a datastore with chromadb, embeds the documents with all-mpnet-base-v2, 
and returns the 3 most similar examples to the user's query.
"""

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class RAG_Service:
    def __init__(self, data_path: str, collection_name: str = "sciq_examples", similarity_threshold=1.0, embedding_target: str = "support"):
        """
        embedding_target: The text that will be embedded. Accepted values: 
            - support -> to embed the support text in the datastore
            - qa -> to embed a question-answer pair. no distractors

        """
        print(f"Initializing RAG Service with data from: {data_path}")
        self.similarity_threshold = similarity_threshold # only examples with more than 70% similarity are used
        self.data_path = data_path
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        self.embedding_target = embedding_target
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

        if self.collection.count() == 0:
            self._ingest_data()
        else:
            print(f"Collection '{collection_name}' already exists with {self.collection.count()} entries.")

    def _ingest_data(self):
        """Reads CSV, embeds text, and stores metadata in batches."""
        print("Ingesting data into ChromaDB...")
        try:
            df = pd.read_csv(self.data_path)
            df = df.dropna(subset=['support', 'question', 'correct_answer'])

            documents = []
            metadatas = []
            ids = []

            if self.embedding_target == "support": 
                for idx, row in df.iterrows():
                    documents.append(row["support"])
                    metadatas.append({
                        "question_id":    str(row.get("question_id", idx)),
                        "question":       row["question"],
                        "correct_answer": row["correct_answer"],
                        "distractor1":    row["distractor1"],
                        "distractor2":    row["distractor2"],
                        "distractor3":    row["distractor3"]
                    })
                    ids.append(str(idx))
            elif self.embedding_target == "qa":
                for idx, row in df.iterrows():
                    documents.append(f"{row['question']} {row['correct_answer']}")
                    metadatas.append({
                        "question_id":    str(row.get("question_id", idx)),
                        "question":       row["question"],
                        "correct_answer": row["correct_answer"],
                        "support":        row["support"],
                        "distractor1":    row["distractor1"],
                        "distractor2":    row["distractor2"],
                        "distractor3":    row["distractor3"]
                    })
                    ids.append(str(idx))


            embeddings = self.embedding_model.encode(
                documents,
                batch_size=64,
                show_progress_bar=True
            ).tolist()

            # chunk into batches of 5000
            batch_size = 5000
            for start in range(0, len(ids), batch_size):
                end = min(start + batch_size, len(ids))
                print(f"Adding batch {start}:{end} to ChromaDB...")
                self.collection.add(
                    embeddings=embeddings[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                    ids=ids[start:end]
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
                    "question_id": meta["question_id"],
                    "support": meta.get("support", doc),
                    "question": meta["question"],
                    "correct_answer": meta["correct_answer"],
                    "distractors": [
                        meta["distractor1"],
                        meta["distractor2"],
                        meta["distractor3"]
                    ],
                    "distance": distance
                }

                examples.append(example)
        
        return examples
    

