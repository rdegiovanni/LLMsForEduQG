""" "
This file handles the creation of user prompts in different formats: Simple, Simple_plus_Answer, and FewShot, which includes examples retrieved through RAG.
"""

import random
from enum import Enum
from typing import Any, Dict, List


class PromptID(Enum):
    Simple = 0
    Simple_plus_Answer = 1

    # example ordering prompts
    FewShot = 2
    FewShot_Reverse = (
        3  # full prompt (support text, answer, and examples in reversed order)
    )
    FewShot_Random = 4

    # ablation study prompts
    FewShot_NoContext = 5  # from the example-order experiments it turns out that putting the most relevant example first is better. Now we use this order for the ablation study checking which parts of the prompt are required.
    FewShot_NoContext_WithAnswer = 6
    FewShot_NoAnswer = 7

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
    gt_question: Dict[str, Any]
    examples: List[Dict]

    def __init__(self, id=PromptID.Simple, question=None, examples=None):
        self.id = id
        self.gt_question = question
        self.examples = examples if examples else []
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
            self.prompt = (
                'Given support text "%s", create 1 expert level question with multiple choice answer from the text. '
                "Please, also create the correct answer and 3 distractors. "
            ) % (support_text)
        elif self.id == PromptID.Simple_plus_Answer:
            self.prompt = (
                'Given support text "%s", create 1 expert level question '
                "with multiple choice answer from the text, "
                'for which the correct answer is "%s". '
                "Please, also create 3 distractors."
            ) % (support_text, expected_answer)
        elif self.id in [
            PromptID.FewShot,
            PromptID.FewShot_Reverse,
            PromptID.FewShot_Random,
        ]:
            self.ordered_examples = self.examples[:]

            if self.id == PromptID.FewShot_Reverse:
                self.ordered_examples.reverse()  # Least Similar first
            elif self.id == PromptID.FewShot_Random:
                random.shuffle(self.ordered_examples)

            examples_str = ""
            for i, ex in enumerate(self.ordered_examples):
                # add mechanims to change order of the examples here
                # like, there should be 3 configurations: most similar first, least similar first, and random
                examples_str += (
                    f"Example {i + 1}:\n"
                    f'Support Text: "{ex["support"]}"\n'
                    f"Question: {ex['question']}\n"
                    f"Correct Answer: {ex['correct_answer']}\n"
                    f"Distractors: {', '.join(ex['distractors'])}\n\n"
                )
            self.prompt = (
                f'Given support text "{support_text}", create 1 expert level question with multiple choice answer from the text. '
                "Please, also create the correct answer and 3 distractors. "
                "Use the following examples as a guide. \n\n"
                f"{examples_str}"
            )
        elif self.id == PromptID.FewShot_NoContext:
            self.ordered_examples = self.examples[:]

            examples_str = ""
            for i, ex in enumerate(self.ordered_examples):
                examples_str += (
                    f"Example {i + 1}:\n"
                    f"Question: {ex['question']}\n"
                    f"Correct Answer: {ex['correct_answer']}\n"
                    f"Distractors: {', '.join(ex['distractors'])}\n\n"
                )
            self.prompt = (
                "Create 1 expert level question with multiple choice answer. "
                "Please, also create the correct answer and 3 distractors. "
                "Use the following examples as a guide.\n\n"
                f"{examples_str}"
            )

        elif self.id == PromptID.FewShot_NoContext_WithAnswer:
            self.ordered_examples = self.examples[:]

            examples_str = ""
            for i, ex in enumerate(self.ordered_examples):
                examples_str += (
                    f"Example {i + 1}:\n"
                    f"Question: {ex['question']}\n"
                    f"Correct Answer: {ex['correct_answer']}\n"
                    f"Distractors: {', '.join(ex['distractors'])}\n\n"
                )
            self.prompt = (
                "Create 1 expert level question with multiple choice answer "
                f'for which the correct answer is "{expected_answer}" .'
                "Please, also create 3 distractors. "
                "Use the following examples as a guide.\n\n"
                f"{examples_str}"
            )
        elif self.id == PromptID.FewShot_NoAnswer:
            self.ordered_examples = self.examples[:]

            examples_str = ""
            for i, ex in enumerate(self.ordered_examples):
                examples_str += (
                    f"Example {i + 1}:\n"
                    f'Support Text: "{ex["support"]}"\n'
                    f"Question: {ex['question']}\n"
                    f"Distractors: {', '.join(ex['distractors'])}\n\n"
                )
            self.prompt = (
                f'Given support text "{support_text}", create 1 expert level question '
                "with multiple choice answer from the text. "
                "Please, also create the correct answer and 3 distractors. "
                "Use the following examples as a guide.\n\n"
                f"{examples_str}"
            )
