"""
This file handles the creation of user prompts in different formats:
ZeroShot, ZeroShot_WithAnswer, FewShot, and FewShot_WithAnswer.
FewShot variants use RAG-retrieved examples in most-similar-first order.
"""

import random
from enum import Enum
from typing import Any, Dict, List


class PromptID(Enum):
    ZeroShot = 0
    ZeroShot_WithAnswer = 1
    FewShot = 2
    FewShot_WithAnswer = 3

    @classmethod
    def all(cls):
        return list(map(lambda c: c, cls))

    @classmethod
    def print_supported_prompts(cls):
        print("List of supported Prompts:")
        for p in cls.all():
            print("{} : {}".format(p.name, p))


class Prompt:
    id: PromptID
    prompt: str
    gt_question: Dict[str, Any]
    examples: List[Dict]

    def __init__(self, id=PromptID.ZeroShot, question=None, examples=None):
        self.id = id
        self.gt_question = question
        self.examples = examples if examples else []
        self.retrieved_examples = self.examples
        self.instantiate_prompt_template()

    def instantiate_prompt_template(self):
        support_text = self.gt_question["support"]
        expected_answer = self.gt_question["correct_answer"]

        if self.id == PromptID.ZeroShot:
            self.prompt = (
                f'Given support text "{support_text}", create 1 expert level question '
                f'with multiple choice answer from the text. '
                f'Please, also create the correct answer and 3 distractors. '
            )

        elif self.id == PromptID.ZeroShot_WithAnswer:
            self.prompt = (
                f'Given support text "{support_text}", create 1 expert level question '
                f'with multiple choice answer from the text, '
                f'for which the correct answer is "{expected_answer}". '
                f'Please, also create 3 distractors.'
            )

        elif self.id == PromptID.FewShot:
            examples_str = ""
            for i, ex in enumerate(self.examples):   # most-similar-first (default RAG order)
                examples_str += (
                    f"Example {i + 1}:\n"
                    f'Support Text: "{ex["support"]}"\n'
                    f"Question: {ex['question']}\n"
                    f"Distractors: {', '.join(ex['distractors'])}\n\n"
                )
            self.prompt = (
                f'Given support text "{support_text}", create 1 expert level question '
                f'with multiple choice answer from the text. '
                f'Please, also create the correct answer and 3 distractors. '
                f'Use the following examples as a guide.\n\n'
                f'{examples_str}'
            )

        elif self.id == PromptID.FewShot_WithAnswer:
            examples_str = ""
            for i, ex in enumerate(self.examples):   # most-similar-first (default RAG order)
                examples_str += (
                    f"Example {i + 1}:\n"
                    f'Support Text: "{ex["support"]}"\n'
                    f"Question: {ex['question']}\n"
                    f"Correct Answer: {ex['correct_answer']}\n"
                    f"Distractors: {', '.join(ex['distractors'])}\n\n"
                )
            self.prompt = (
                f'Given support text "{support_text}", create 1 expert level question '
                f'with multiple choice answer from the text, '
                f'for which the correct answer is "{expected_answer}". '
                f'Please, also create 3 distractors. '
                f'Use the following examples as a guide.\n\n'
                f'{examples_str}'
            )