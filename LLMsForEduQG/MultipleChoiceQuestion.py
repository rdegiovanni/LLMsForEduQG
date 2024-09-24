from pydantic import BaseModel, Field
from typing import Optional


class MultipleChoiceQuestion(BaseModel):
    question: str = Field(description="question to set up a multi-choice question")
    correct_answer: Optional[str] = Field(default="",description="correct answer of the multi-choice question")
    distractor1: Optional[str] = Field(default="",description="first distractor of the multi-choice question")
    distractor2: Optional[str] = Field(default="",description="second distractor of the multi-choice question")
    distractor3: Optional[str] = Field(default="",description="third distractor of the multi-choice question")
    support: Optional[str] = Field(default="",description="support text corresponding to the multi-choice question")

    # @validator("question")
    # def question_ends_with_question_mark(cls, field):
    #     if field[-1] != "?":
    #         raise ValueError("Badly formed question!")
    #     return field

    def toJSON(self):
        return {"correct_answer": self.correct_answer,
                "distractor1": self.distractor1,
                "distractor2": self.distractor2,
                "distractor3": self.distractor3,
                "question": self.question,
                "support": self.support}

    def toJSONstr(self):
        return (
            "{"
            "\"correct_answer\": \"%s\","
            "\"distractor1\": \"%s\","
            "\"distractor2\": \"%s\","
            "\"distractor3\": \"%s\","
            "\"question\": \"%s\","
            "\"support\": \"%s\"}"
        ) % (self.correct_answer, self.distractor1, self.distractor2, self.distractor3, self.question, self.support)


# class MultipleChoiceQuestionInstance(MultipleChoiceQuestion):
#     def __init__(self, correct_answer, distractor1, distractor2, distractor3, question, support):
#         self.correct_answer = correct_answer
#         self.distractor1 = distractor1
#         self.distractor2 = distractor2
#         self.distractor3 = distractor3
#         self.question = question
#         self.support = support



        # json.dumps(
        # self,
        # default=lambda o: o.__dict__,
        # sort_keys=True,
        # indent=4))

# def is_json(myjson):
#     try:
#         json.loads(myjson)
#     except ValueError as e:
#         return False
#     return True

# def __init__(self, question="", distractors=None, correct_answer="", support=""):
#     self.question = question
#     self.correct_answer = correct_answer
#     if distractors is not None and len(distractors) > 0:
#         self.distractors = distractors
#     self.support = support
#
# # ---------------------------------------------------------------------------------
# # Create from SciQ dataset entry
# # ---------------------------------------------------------------------------------
# def __init__(self, data_entry = None):
#     if data_entry is not None:
#         if (self.is_json(data_entry)
#                 and "question" in data_entry
#         and "distractor1" in data_entry
#         and "distractor2" in data_entry
#         and "distractor3" in data_entry
#         and "correct_answer" in data_entry
#         and "support" in data_entry):
#             distractors = []
#             distractors.append(data_entry["distractor1"])
#             distractors.append(data_entry["distractor2"])
#             distractors.append(data_entry["distractor3"])
#             self.__init__(data_entry["question"],distractors,data_entry["correct_answer"],data_entry["support"])
