import csv
import json
import random

from Prompt import PromptID
from LLM_Service import LLM_Service
from Prompt import Prompt

import numpy as np
import pandas as pd

from Metrics import Metrics
from Statistics import Statistics

class LLMsForEduQG:
    metrics: Metrics
    statistics: Statistics
    llm_service: LLM_Service
    ground_truth_questions = pd.DataFrame(columns = ['question_id','set','question','correct_answer','distractor1','distractor2','distractor3','support'])
    selected_questions = []

    prompts = pd.DataFrame(columns=['question_id', 'prompt'])
    generated_questions = None

    def __init__(self,input_filename,results_dir,MAX=0,random_choice=False):

        self.metrics = Metrics()
        self.statistics = Statistics(input_filename,results_dir, self.metrics)
        self.llm_service = LLM_Service()

        self.generated_questions = pd.DataFrame(
            columns=['question_id', 'prompt_id', 'model_id', 'question', 'correct_answer', 'distractor1', 'distractor2',
                     'distractor3',
                     'support'] + self.metrics.get_available_metrics())

        self.load_testing_data(input_filename, MAX, random_choice)


    # It takes the question ID (qid) from the dataset, and the prompt id (pid)
    def generate_prompts(self,qid,pid):

        qid_indexes = self.ground_truth_questions["question_id"] == qid
        question = {'correct_answer': self.ground_truth_questions["correct_answer"][qid_indexes].values[0],
                  'distractor1' : self.ground_truth_questions["distractor1"][qid_indexes].values[0],
                  'distractor2' : self.ground_truth_questions["distractor2"][qid_indexes].values[0],
                  'distractor3' : self.ground_truth_questions["distractor3"][qid_indexes].values[0],
                  'question' : self.ground_truth_questions["question"][qid_indexes].values[0],
                  'support' : self.ground_truth_questions["support"][qid_indexes].values[0]}

        self.prompts = self.prompts._append({"question_id": qid, "prompt": Prompt(pid,question)
                                                 }, ignore_index=True)

    def execute(self,qid,pid,mid):
        qid_indexes = self.ground_truth_questions["question_id"] == qid
        expected_question = self.ground_truth_questions["question"][qid_indexes].values[0]

        for prompt in self.prompts[self.prompts["question_id"] == qid]["prompt"]:
            if prompt.id is not pid:
                continue
            print("Running QID: {}, PID: {}, MID: {}.".format(qid,prompt.id,mid))
            response = self.llm_service.execute_prompt(mid,prompt.prompt)
            if response is None:
                answer_result = {"question_id" : qid,
                                 "prompt_id" : prompt.id.name,
                                 "model_id": mid,
                                 "question" : "", "correct_answer" : "",
                                "distractor1" : "", "distractor2" : "", "distractor3":"", "support" : ""}
                for m in self.metrics.get_available_metrics():
                    answer_result[m] = "0.0"
                self.generated_questions = self.generated_questions._append(answer_result, ignore_index = True)
            elif response == "error=429":
                answer_result = {"question_id": qid,
                                 "prompt_id": prompt.id.name,
                                 "model_id": mid,
                                 "question": "error=429", "correct_answer": "",
                                 "distractor1": "", "distractor2": "", "distractor3": "", "support": ""}
                for m in self.metrics.get_available_metrics():
                    answer_result[m] = "0.0"
                self.generated_questions = self.generated_questions._append(answer_result, ignore_index=True)
            else:
                answer_scores = self.metrics.compute_scores(response.question,expected_question)
                answer_result = {"question_id" : qid,
                                 "prompt_id": prompt.id.name,
                                 "model_id": mid,
                                 "question" : response.question,
                                "correct_answer" : response.correct_answer,
                                "distractor1" : response.distractor1,
                                "distractor2" : response.distractor2,
                                "distractor3":response.distractor3,
                                "support" : response.support
                               }
                for m in self.metrics.get_available_metrics():
                    answer_result[m] = answer_scores[m][0]
                self.generated_questions = self.generated_questions._append(answer_result, ignore_index = True)
            break
        print(">>>")
        print(">>>")



    def load_testing_data(self, inputname, MAX=0, random_choice=False):
        self.ground_truth_questions = pd.read_csv(inputname, keep_default_na=False)
        if MAX <= 0:
            self.selected_questions.extend(sorted(self.ground_truth_questions["question_id"]))
        else:
            if random_choice:
                self.selected_questions.extend(random.sample(sorted(self.ground_truth_questions["question_id"]), MAX))
            else:
                self.selected_questions.extend(self.ground_truth_questions["question_id"][:MAX])

    def is_valid_context(self,qid):
        qid_indexes = self.ground_truth_questions["question_id"] == qid
        support_text : str
        support_text = self.ground_truth_questions["support"][qid_indexes].values[0]
        if support_text is None or support_text == "" or len(support_text) == 0 or support_text.lower() == "nan":
            return False
        return True

    def report(self,qid,pid,mid,first_time = False):
        data_file = None
        csv_writer = None
        # headers to the CSV file
        out_filename = self.statistics.RESULTS_FILENAME
        if first_time:
            data_file = open(out_filename, 'w', newline='', encoding='utf-8')
            csv_writer = csv.writer(data_file, quoting=csv.QUOTE_NONNUMERIC)
            csv_writer.writerow(
            ['question_id', 'prompt_id', 'model_id', 'question', 'correct_answer', 'distractor1', 'distractor2', 'distractor3', 'support']+self.metrics.get_available_metrics())
        else:
            data_file = open(out_filename, 'a', newline='', encoding='utf-8')
            csv_writer = csv.writer(data_file, quoting=csv.QUOTE_NONNUMERIC)
        qid_row = self.generated_questions.loc[(self.generated_questions["question_id"] == qid) &
                                               (self.generated_questions["prompt_id"] == pid.name) &
                                               (self.generated_questions["model_id"] == mid),:]
        csv_writer.writerow(qid_row.values[0])
        data_file.flush()
        data_file.close()


    def run_per_qid(self,prompt_ids=PromptID.all(),model_ids=[],new_file=True):
        for qid in self.selected_questions:
            if not self.is_valid_context(qid):
                continue
            for pid in prompt_ids:
                self.generate_prompts(qid,pid)
                for mid in model_ids:
                    self.execute(qid,pid,mid)
                    self.report(qid,pid,mid,new_file)
                    new_file = False
        self.statistics.clean_generated_questions()
        self.statistics.generate_summary()
        self.statistics.compute_statistics()
        self.statistics.generate_plots()

        print()
        print()
        print(">>>>")
        print("Cold Models: {}".format(self.llm_service.cold_models))
        print()
        print("Unsupported Models: {}".format(self.llm_service.error_models))

