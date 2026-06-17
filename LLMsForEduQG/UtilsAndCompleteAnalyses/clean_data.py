import pandas as pd
from transformers import AutoTokenizer


class DatasetCleaner:
    MAX_TOKENS = 512
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.df = pd.read_csv(data_path)
        self.df_clean = None

    def count_tokens(self) -> None:
        self.df["token_count"] = self.df["support"].fillna("").apply(
            lambda x: len(self.tokenizer.encode(x, add_special_tokens=False))
        )

    def clean(self) -> None:
        if "token_count" not in self.df.columns:
            self.count_tokens()
        self.df_clean = self.df[
            (self.df["token_count"] > 0) & (self.df["token_count"] <= self.MAX_TOKENS)
        ]

    def report(self) -> None:
        if self.df_clean is None:
            print("Run clean() first.")
            return
        before = len(self.df)
        after = len(self.df_clean)
        print(f"Dropped {before - after} rows ({(before - after) / before * 100:.2f}%)")
        print(f"Remaining rows: {after}")
        print("\nStats after cleaning:")
        print(self.df_clean["token_count"].describe())
        print(f"Min tokens: {self.df_clean['token_count'].min()}")

    def save(self) -> str:
        if self.df_clean is None:
            print("Run clean() first.")
            return ""
        out_path = self.data_path.replace(".csv", "_clean.csv")
        self.df_clean.drop(columns=["token_count"]).to_csv(out_path, index=False)
        print(f"Saved to: {out_path}")
        return out_path


if __name__ == "__main__":
    DATA_PATH = "/home/ldap/coccia@private.list.lu/oat_2024/RAGForEduQG/datasets/SciQ_train.csv"

    cleaner = DatasetCleaner(DATA_PATH)
    cleaner.clean()
    cleaner.report()
    cleaner.save()