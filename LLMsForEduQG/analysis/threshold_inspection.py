import pandas as pd

def main():
    df = pd.read_csv("../datasets/retrieval_similarity_analysis.csv")

    for threshold in [0.8, 0.7, 0.6, 0.5, 0.4]:
        sample = df[df["support_similarity"].between(threshold - 0.05, threshold)].head(2)
        print(f"\n── ~{threshold} ──")
        for _, row in sample.iterrows():
            print(f"  GT:        {row['gt_support']}")
            print(f"  Retrieved: {row['retrieved_support']}")
            print(f"  Score:     {row['support_similarity']:.3f}")



if __name__ == "__main__":
    main()