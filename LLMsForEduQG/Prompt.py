from enum import Enum

class PromptID(Enum):
    Simple = 0
    Simple_plus_Answer = 1

    @classmethod
    def all(self):
        return list(map(lambda c: c, self))

    @classmethod
    def print_supported_prompts(self):
        print("List of supported Prompts:")
        for p in self.all():
            print("{} : {}".format(p.name, p))

class Prompt:

    id: PromptID
    prompt: str
    gt_question: {}

    def __init__(self, id=PromptID.Simple, question=None):
        self.id = id
        self.gt_question = question
        self.instantiate_prompt_template()

    # generate prompt templates
    def instantiate_prompt_template(self):
        distractor1 = self.gt_question["distractor1"]
        distractor2 = self.gt_question["distractor2"]
        distractor3 = self.gt_question["distractor3"]
        question = self.gt_question["question"]
        support_text = self.gt_question["support"]
        expected_answer = self.gt_question["correct_answer"]

        # simple prompt taken from the paper
        # Small Generative Language Models for Educational Question Generation
        # NeurIPS 2023 Workshop on Generative AI for Education (GAIED).
        if self.id == PromptID.Simple:
            self.prompt = ("Given support text \"%s\", create 1 expert level question with multiple choice answer from the text. "
                  "Please also include the correct answer and 3 distractors. ") % (support_text)
        elif self.id == PromptID.Simple_plus_Answer:
                self.prompt = ("Given support text \"%s\", create 1 expert level question with multiple choice answer from the text, "
              "for which the correct answer is \"%s\". Please, also create 3 distractors.") % (support_text,expected_answer)




