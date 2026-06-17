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
    FewShot = 2              # most-similar-first
    FewShot_WithAnswer = 3   # most-similar-first, correct answer in prompt
    FewShot_Reverse = 4      # least-similar-first
    FewShot_MostSimMiddle = 5  # most-similar in centre position
    FewShot_Reverse_WithAnswer = 6  # least-similar-first, correct answer in prompt
    Multiturn = 7
    Multiturn_WithAnswer = 8

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
        self.messages = []
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
            for i, ex in enumerate(self.examples):  # most-similar-first (default RAG order)
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
            for i, ex in enumerate(self.examples):  # most-similar-first (default RAG order)
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

        elif self.id == PromptID.FewShot_Reverse:
            ordered = self.examples[:]
            ordered.reverse()  # least-similar-first

            examples_str = ""
            for i, ex in enumerate(ordered):
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

        elif self.id == PromptID.FewShot_MostSimMiddle:
            ordered = self.examples[:]
            if len(ordered) == 3:
                ordered = [ordered[1], ordered[0], ordered[2]]  # [2nd, 1st, 3rd]

            examples_str = ""
            for i, ex in enumerate(ordered):
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
        elif self.id == PromptID.FewShot_Reverse_WithAnswer:
            ordered = self.examples[:]
            ordered.reverse()  # least-similar-first

            examples_str = ""
            for i, ex in enumerate(ordered):
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

        elif self.id == PromptID.Multiturn:
            messages = []

            for ex in self.examples: # providing examples in a simulated multiturn conversation
                messages.append({
                    "role": "user",
                    "content": (
                        f'Given support text "{ex["support"]}", '
                        f'create 1 expert level question '
                        f'with multiple choice answer from the text. '
                        f'Please, also create the correct answer. '
                    )
                })

                messages.append({
                    "role": "assistant",
                    "content": f"Question: {ex['question']}\n"
                })
        
            messages.append({
                "role": "user",
                "content": (
                    f'Given support text "{support_text}", '
                    f'create 1 expert level question '
                    f'with multiple choice answer from the text. '
                    f'Please, also create the correct answer. '
                )
            })

            self.prompt = ""
            self.messages = messages

        elif self.id == PromptID.Multiturn_WithAnswer:
            messages = []

            for ex in self.examples: # providing examples in a simulated multiturn conversation
                messages.append({
                    "role": "user",
                    "content": (
                        f'Given support text "{ex["support"]}", '
                        f'create 1 expert level question '
                        f'with multiple choice answer from the text '
                        f'for which the correct answer is "{ex["correct_answer"]}".'
                        f'Please, also create the correct answer. '
                    )
                })

                messages.append({
                    "role": "assistant",
                    "content": f"Question: {ex['question']}\n"
                })
        
            messages.append({
                "role": "user",
                "content": (
                    f'Given support text "{support_text}", '
                    f'create 1 expert level question '
                    f'with multiple choice answer from the text '
                    f'for which the correct answer is "{expected_answer}".'
                    f'Please, also create the correct answer. '
                )
            })

            self.prompt = ""
            self.messages = messages

