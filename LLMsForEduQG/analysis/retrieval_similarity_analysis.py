"""
When RAG finds a similar context, does it also find a similar question?
i.e. is support similarity a good proxy for question relevance?
We embed the support text of the training set into the datastore. We query the datastore with each support text of the test set.
We get 3 similar support texts from the datastore. We use the retrieved texts to get their associated questions. 
We also measure the distance between the question in the test set (associated with the query support text) and the ones associated to the retrieved support texts.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import scipy.stats as ss
import os

sys.path.append("/home/ldap/coccia@private.list.lu/oat_2024/RAGForEduQG/LLMsForEduQG")
from RAG_Service import RAG_Service

TEST_PATH   = "/home/ldap/coccia@private.list.lu/oat_2024/RAGForEduQG/datasets/SciQ_test.csv"
TRAIN_PATH  = "/home/ldap/coccia@private.list.lu/oat_2024/RAGForEduQG/datasets/SciQ_train_clean.csv"
OUTPUT_PATH = "/home/ldap/coccia@private.list.lu/oat_2024/RAGForEduQG/datasets/retrieval_similarity_analysis.csv"
MODEL_NAME  = "sentence-transformers/all-mpnet-base-v2"


class RetrievalSimilarityAnalyzer:

    def __init__(self, test_path: str, train_path: str, output_path: str):
        self.df_test = pd.read_csv(test_path, dtype={"question_id": str, "support": str, "question": str})
        self.test_path   = test_path
        self.train_path  = train_path
        self.output_path = output_path
        self.model       = SentenceTransformer(MODEL_NAME)
        self.rag_service = RAG_Service(data_path=train_path)
        self.df_test     = pd.read_csv(test_path, dtype={"question_id": str})

    def _cosine_sim(self, text_a: str, text_b: str) -> float:
        """Computes cosine similarity between two texts."""
        emb = self.model.encode([text_a, text_b])
        return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

    def analyze(self) -> pd.DataFrame:
        rows = []
        total = len(self.df_test)

        for i, row in self.df_test.iterrows():
            print(f"Processing {i+1}/{total} (question_id={row['question_id']})")

            gt_question = row["question"]
            gt_support  = row["support"]

            if pd.isna(gt_support) or str(gt_support).strip() == "":
                continue

            examples = self.rag_service.retrieve_examples(
                query_text=gt_support, n_results=3
            )

            for rank, ex in enumerate(examples, start=1):
                q_sim = self._cosine_sim(gt_question, ex["question"])
                rows.append({
                    "gt_question_id":       row["question_id"],
                    "gt_question":          gt_question,
                    "gt_support":           gt_support,
                    "retrieved_rank":       rank,
                    "retrieved_question":   ex["question"],
                    "retrieved_support":    ex["support"],
                    "support_distance":     ex["support_distance"],  # from ChromaDB
                    "support_similarity":   1 - ex["support_distance"],  # convert to similarity
                    "question_similarity":  q_sim,
                })

        self.df_results = pd.DataFrame(rows)
        return self.df_results

    def report(self) -> None:
        if not hasattr(self, "df_results"):
            print("Run analyze() first.")
            return

        support_sim  = self.df_results["support_similarity"]
        question_sim = self.df_results["question_similarity"]

        pearson_r,  pearson_p  = ss.pearsonr(support_sim, question_sim)
        spearman_r, spearman_p = ss.spearmanr(support_sim, question_sim)
        kendall_t,  kendall_p  = ss.kendalltau(support_sim, question_sim)

        print("\n--- Similarity Report ---")
        print(f"Support similarity  — mean: {support_sim.mean():.4f} | median: {support_sim.median():.4f}")
        print(f"Question similarity — mean: {question_sim.mean():.4f} | median: {question_sim.median():.4f}")
        print("\n--- Correlation: support_similarity vs question_similarity ---")
        print(f"Pearson  r = {pearson_r:.4f}  (p={pearson_p:.4f})")
        print(f"Spearman r = {spearman_r:.4f}  (p={spearman_p:.4f})")
        print(f"Kendall  τ = {kendall_t:.4f}  (p={kendall_p:.4f})")

    def save(self) -> None:
        if not hasattr(self, "df_results"):
            print("Run analyze() first.")
            return
        self.df_results.to_csv(self.output_path, index=False)
        print(f"Saved to: {self.output_path}")


if __name__ == "__main__":
    analyzer = RetrievalSimilarityAnalyzer(TEST_PATH, TRAIN_PATH, OUTPUT_PATH)
    analyzer.analyze()
    analyzer.report()
    analyzer.save()