"""
This script compares the question-answer pairs retrieved using by 
quering vector databases created by embedding different parts of the training set.
The first vector db contains the embeddings for the support text.
The second vector db contains the embeddings for the question-answer pairs
"""

from RAG_Service import RAG_Service
import pandas as pd # pyright: ignore[reportMissingModuleSource] # type: Ignore

DB_INPUT_PATH = r"/home/ldap/coccia@private.list.lu/oat_2024/RAGForEduQG/datasets/SciQ_train_clean.csv"
QUERY_INPUT_PATH = r"/home/ldap/coccia@private.list.lu/oat_2024/RAGForEduQG/datasets/SciQ_test.csv"
OUTPUT_PATH = r"/home/ldap/coccia@private.list.lu/oat_2024/RAGForEduQG/datasets/examples_overlap.csv"
SUMMARY_PATH = r"/home/ldap/coccia@private.list.lu/oat_2024/RAGForEduQG/datasets/examples_overlap_summary.csv"

class RAGOverlapExperiment:
    def __init__(self, db_input_path, query_input_path, output_path, summary_path):
        self.db_input_path = db_input_path
        self.query_input_path = query_input_path
        self.output_path = output_path
        self.summary_path = summary_path
        
        self.embed_support_rag = RAG_Service(  
            data_path=self.db_input_path,
            collection_name="sciq_examples_from_support", 
            similarity_threshold=0.35, 
            embedding_target="support"
        )

        self.embed_qa_rag = RAG_Service(
            data_path=self.db_input_path,
            collection_name="sciq_examples_from_qa", 
            similarity_threshold=0.45, 
            embedding_target="qa"
        )

    def _get_examples(self, query_text):
        if not isinstance(query_text, str) or query_text.strip() == "":
            return [], [], [], []
        ex_sup = self.embed_support_rag.retrieve_examples(query_text=query_text, n_results=3)
        ids_sup = [item["question_id"] for item in ex_sup]
                    
        ex_qa = self.embed_qa_rag.retrieve_examples(query_text=query_text, n_results=3)
        ids_qa = [item["question_id"] for item in ex_qa] 
        
        return ex_sup, ex_qa, ids_sup, ids_qa
    
    def _calculate_jaccard(self, ids_sup, ids_qa):
        set_sup = set(ids_sup)
        set_qa = set(ids_qa)

        intersection = set_sup & set_qa
        union = set_sup | set_qa

        jaccard = len(intersection) / len(union) if union else 0.0
        return jaccard

    def _save_results(self, results):
        results_df = pd.DataFrame(results)
        summary_df = pd.DataFrame([self._summarize(results_df)])
        
        results_df.to_csv(self.output_path, index=False)
        summary_df.to_csv(self.summary_path)

        return results_df
    
    def _summarize(self, results_df):
        empty_examples_filter = (results_df["retrieved_ids_support"] == "[]") & (results_df["retrieved_ids_qa"] == "[]")
        filtered_df = results_df[~empty_examples_filter]
        mean_jaccard = filtered_df["jaccard"].mean()
        
        no_overlap = filtered_df[filtered_df["jaccard"] == 0.0]
        overlap = (len(filtered_df) - len(no_overlap)) / len(filtered_df)
        
        empty_support_filter = results_df["retrieved_ids_support"].apply(lambda x: len(x) == 0)
        empty_qa_filter = results_df["retrieved_ids_qa"].apply(lambda x: len(x) == 0)

        pct_empty_support = len(results_df[empty_support_filter]) / len(results_df)
        pct_empty_qa = len(results_df[empty_qa_filter]) / len(results_df)

        summary_dict = {
            "mean_jaccard": mean_jaccard,
            "overlap_percentage": overlap,
            "no_examples_support": pct_empty_support,
            "no_examples_qa": pct_empty_qa
        }
        return summary_dict
    
    def run(self):
        query_data = pd.read_csv(self.query_input_path)
        ex_from_support = []
        ex_from_qa = []
        results = []
        for i, row in query_data.iterrows():

            print(f"Processing query {i+1}/{len(query_data)}")

            ex_sup, ex_qa, ids_sup, ids_qa = self._get_examples(row["support"])
            ex_from_support.append(ex_sup)
            ex_from_qa.append(ex_qa)
            
            jaccard = self._calculate_jaccard(ids_sup, ids_qa)
            
            results_row = {
            "query_id": row["question_id"],
            "retrieved_ids_support": ids_sup,
            "retrieved_ids_qa": ids_qa,
            "jaccard": jaccard
            }
            
            results.append(results_row)
        report = self._save_results(results)
        
        return report

            
if __name__ == "__main__":
    example_overlap_calculator = RAGOverlapExperiment(DB_INPUT_PATH, QUERY_INPUT_PATH, OUTPUT_PATH, SUMMARY_PATH)
    report_df = example_overlap_calculator.run()