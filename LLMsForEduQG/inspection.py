import pandas as pd

df = pd.read_csv("results/phase4_full_nebius/clean_generated_questions.csv",
                 keep_default_na=False)

llama = df[df["model_id"] == "NEB_Llama3370Instruct"]
perfect = llama[llama["f1"] == 1.0]

print(f"N perfect-F1 questions: {len(perfect)}")
print("\nMean copy ratios for perfect-F1 Llama questions:")
print(perfect[["copy_ratio_support", "copy_ratio_reference"]].mean().round(3))

print("\nCompared to Llama overall:")
print(llama[["copy_ratio_support", "copy_ratio_reference"]].mean().round(3))

print("##### DATASET INSPECTION #####")
# add to inspect_copy.py
gt = pd.read_csv("../datasets/SciQ_valid.csv", keep_default_na=False)
from Metrics import Metrics
m = Metrics()
ratios = [m.longest_common_substring_ratio(r["question"], r["support"])
          for _, r in gt.iterrows() if r["support"]]
import numpy as np
print(f"\nSciQ reference questions: mean support-copy = {np.mean(ratios):.3f}")

import pandas as pd
from RAG_Service import RAG_Service

SCIQ_PATH = "../datasets/SciQ_valid.csv"
rag = RAG_Service(data_path=SCIQ_PATH)
df = pd.read_csv(SCIQ_PATH, keep_default_na=False)

n_check = 20
leaks = 0
for _, row in df.head(n_check).iterrows():
    qid = row["question_id"]
    examples = rag.retrieve_examples(query_text=row["support"], n_results=3)
    example_qids = [ex.get("question_id") or ex.get("question") for ex in examples]
    # check if the query's own question appears in its retrieved examples
    if any(ex["question"] == row["question"] for ex in examples):
        leaks += 1
        print(f"LEAK on qid {qid}: own question retrieved as FewShot example")

print(f"\nLeakage: {leaks}/{n_check} queries had self-retrieval")