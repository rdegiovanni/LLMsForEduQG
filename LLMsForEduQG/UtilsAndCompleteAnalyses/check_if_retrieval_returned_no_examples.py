"""
The retrieved examples per query (support to support comparison) are in /home/ldap/coccia@private.list.lu/oat_2024/RAGForEduQG/datasets/retrieval_similarity_analysis.csv
Are there any queries for which there was no retrieved example?
"""
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv(r"/home/ldap/coccia@private.list.lu/oat_2024/RAGForEduQG/datasets/retrieval_similarity_analysis.csv")
    data["gt_question_id"].nunique()
    df_test = pd.read_csv(r"/home/ldap/coccia@private.list.lu/oat_2024/RAGForEduQG/datasets/SciQ_test.csv")
    print(data["gt_question_id"].nunique(), "/", len(df_test["support"]))