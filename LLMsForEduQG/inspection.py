
# test_rag_threshold.py
import pandas as pd
from RAG_Service import RAG_Service

FULL_DATA_PATH = r"C:\Users\coccia\Documents\Projects\LLMsForEduQG\datasets\SciQ_valid.csv"
EVAL_DATA_PATH = r"C:\Users\coccia\Documents\Projects\LLMsForEduQG\datasets\SciQ_100_valid.csv"
df = pd.read_csv(EVAL_DATA_PATH).head(5)

# init RAG with your threshold
rag = RAG_Service(
    data_path=FULL_DATA_PATH,
    similarity_threshold=0.30
)

# run retrieval for each of the 5 questions
for _, row in df.iterrows():
    examples = rag.retrieve_examples(query_text=row["support"], n_results=3)
    print(f"QID {row['question_id']} → {len(examples)} example(s) retrieved")