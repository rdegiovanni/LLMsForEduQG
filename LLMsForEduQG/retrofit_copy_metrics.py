import pandas as pd
from pathlib import Path
from Metrics import Metrics
from Statistics import Statistics

RESULTS_DIR = "results/phase4_full_nebius"
SCIQ_PATH   = "../datasets/SciQ_valid.csv"

m = Metrics()
gt = pd.read_csv(SCIQ_PATH, keep_default_na=False)[["question_id", "question", "support"]]
gt = gt.rename(columns={"question": "ref_question", "support": "ref_support"})

def add_copy_cols(path):
    df = pd.read_csv(path, keep_default_na=False)
    merged = df.merge(gt, on="question_id", how="left")
    merged["copy_ratio_support"] = merged.apply(
        lambda r: m.longest_common_substring_ratio(str(r["question"]), str(r["ref_support"])) if r["question"] else 0.0,
        axis=1)
    merged["copy_ratio_reference"] = merged.apply(
        lambda r: m.longest_common_substring_ratio(str(r["question"]), str(r["ref_question"])) if r["question"] else 0.0,
        axis=1)
    merged = merged.drop(columns=["ref_question", "ref_support"])
    # drop old copy_ratio column if present
    if "copy_ratio" in merged.columns:
        merged = merged.drop(columns=["copy_ratio"])
    merged.to_csv(path, index=False, quoting=1)
    print(f"Retrofitted {len(merged)} rows in {path}")

for f in ["generated_questions.csv", "clean_generated_questions.csv"]:
    p = Path(RESULTS_DIR) / f
    if p.exists():
        add_copy_cols(p)

# regenerate summary + plots with new metrics
s = Statistics(SCIQ_PATH, RESULTS_DIR, m)
s.generate_summary()
s.generate_plots()
print("Done.")