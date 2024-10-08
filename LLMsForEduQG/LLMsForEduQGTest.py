import unittest
from LLMsForEduQG import LLMsForEduQG
from Prompt import Prompt, PromptID

class MyTestCase(unittest.TestCase):
    def test_only_Flan(self):
        MAX = 1
        random_choice = False
        SciQdataset = r"D:\PycharmProjects\LLMsForEduQG\datasets\SciQ_test.csv"
        results_dir = r"D:\PycharmProjects\LLMsForEduQG\results\\" + str(MAX)
        LLMrunner = LLMsForEduQG(SciQdataset, results_dir, MAX, random_choice)
        prompt_IDs = [PromptID.Simple_plus_Answer]
        models = LLMrunner.llm_service.get_model_ids_startswith("Flan")
        LLMrunner.run_per_qid(prompt_IDs, models)

    def test_only_Falcon(self):
        MAX = 1
        random_choice = False
        SciQdataset = r"D:\PycharmProjects\LLMsForEduQG\datasets\SciQ_test.csv"
        results_dir = r"D:\PycharmProjects\LLMsForEduQG\results\\" + str(MAX)
        LLMrunner = LLMsForEduQG(SciQdataset, results_dir, MAX, random_choice)
        prompt_IDs = [PromptID.Simple_plus_Answer]
        models = LLMrunner.llm_service.get_model_ids_startswith("Falcon")
        LLMrunner.run_per_qid(prompt_IDs, models)

    def test_Llama_32(self):
        MAX = 1
        random_choice = False
        SciQdataset = r"D:\PycharmProjects\LLMsForEduQG\datasets\SciQ_test.csv"
        results_dir = r"D:\PycharmProjects\LLMsForEduQG\results\\" + str(MAX)
        LLMrunner = LLMsForEduQG(SciQdataset, results_dir, MAX, random_choice)
        prompt_IDs = [PromptID.Simple_plus_Answer]
        models = ["Llama321Instruct","Llama321","Llama323Instruct","Llama323"]
        LLMrunner.run_per_qid(prompt_IDs, models)
    # def test_all(self):
    #     MAX = 0
    #     random_choice = False
    #     SciQdataset = r"D:\PycharmProjects\LLMsForEduQG\datasets\SciQ_test.csv"
    #     results_dir = r"D:\PycharmProjects\LLMsForEduQG\results\\" + str(MAX)
    #     # outname_without_answers = r"D:\PycharmProjects\EduQGdataAnalyzer\results\SciQ_test-generated-without-answers-"+str(MAX)+".csv"
    #
    #     LLMrunner = LLMsForEduQG(SciQdataset, results_dir, MAX, random_choice)
    #     prompt_IDs = PromptID.all()
    #     # prompt_IDs = [PromptID.Simple_plus_Answer]
    #     # models = ["meta-llama/Meta-Llama-3.1-70B-Instruct"]
    #     # models = ["FlanT5Large"]
    #     models = LLMrunner.llm_service.get_all_models()
    #     # models = LLMrunner.llm_service.get_model_ids_startswith("Flan")
    #     LLMrunner.run_per_qid(prompt_IDs, models)

    def test_only_GPT(self):
        MAX = 1
        random_choice = False
        SciQdataset = r"D:\PycharmProjects\LLMsForEduQG\datasets\SciQ_test.csv"
        results_dir = r"D:\PycharmProjects\LLMsForEduQG\results\\" + str(MAX)
        LLMrunner = LLMsForEduQG(SciQdataset, results_dir, MAX, random_choice)
        prompt_IDs = [PromptID.Simple_plus_Answer]
        models = LLMrunner.llm_service.get_model_ids_startswith("GPT")
        LLMrunner.run_per_qid(prompt_IDs, models)

if __name__ == '__main__':
    unittest.main()
