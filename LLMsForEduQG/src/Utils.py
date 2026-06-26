import pandas as pd
from src.Statistics import Statistics
from src.Metrics import Metrics

def sample_questions():
    try: 
        data = pd.read_csv("/Users/miriam/projects/LLMsForEduQG/datasets/SciQ_valid.csv").sample(frac=0.1, random_state=42)
        data.to_csv("/Users/miriam/projects/LLMsForEduQG/datasets/SciQ_100_valid.csv")
    except Exception as e:
        print(f"Something went wrong: {e}")
    finally:
        print("Saved the sampled data at: /Users/miriam/projects/LLMsForEduQG/datasets/SciQ_100_valid.csv")
    

def combine_results_ex_order():
    try:
        df_default = pd.read_csv("/Users/miriam/projects/LLMsForEduQG/LLMsForEduQG/results_prompt_modes/GPT5Mini-FewShot/clean_generated_questions.csv")
        df_random = pd.read_csv("/Users/miriam/projects/LLMsForEduQG/LLMsForEduQG/results_prompt_modes/GPT5Mini-FewShot_Random/clean_generated_questions.csv")
        df_reverse = pd.read_csv("/Users/miriam/projects/LLMsForEduQG/LLMsForEduQG/results_prompt_modes/GPT5Mini-FewShot_Reverse/clean_generated_questions.csv")

        combined = pd.concat([df_default, df_random, df_reverse], ignore_index=True)
        combined.to_csv("/Users/miriam/projects/LLMsForEduQG/LLMsForEduQG/results_prompt_modes/generated_questions.csv", index=False, quoting=1)
    except Exception as e:
        print(f"Ran into this following problem: {e}")


def get_combined_stats_ex_order():
    metrics = Metrics()
    results_dir = "/Users/miriam/projects/LLMsForEduQG/LLMsForEduQG/results_prompt_modes/"
    input_filename = "/Users/miriam/projects/LLMsForEduQG/LLMsForEduQG/results_prompt_modes/combined_prompt_modes_results.csv"
    stats = Statistics(input_filename, results_dir, metrics)

    stats.clean_generated_questions()
    stats.generate_summary()
    stats.compute_statistics()
    stats.generate_plots()

    print("Done! Check rresults_prompt_modes")

if __name__ == "__main__":
    combine_results_ex_order()
    get_combined_stats_ex_order()

