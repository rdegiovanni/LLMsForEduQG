import numpy as np
import pandas as pd
import re
import string
from collections import Counter
import torch
import language_tool_python
from transformers import AutoModelForMaskedLM, AutoTokenizer
from nltk import word_tokenize, ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import f1_score

import nltk
nltk.download('punkt_tab')

class Metrics():
    def normalize_answer(self,s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


    def f1_score(self,prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def exact_match_score(self,prediction, ground_truth):
        return (self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def lexical_diversity(self,prediction, rank=3):
        if prediction in None or prediction == "":
            return 0.0
        n_grams = list(ngrams(prediction.split(), rank))
        distinct_ngrams = set(ngrams(prediction.split(), rank))
        if len(prediction.split()) == 0:
            return 0.0
        return len(distinct_ngrams) / len(prediction.split())

    def compute_bleu(self,prediction,gt):
        # gts_tokens = list(map(lambda x: word_tokenize(x), gts))
        gt_tokens = [word_tokenize(gt)]

        prediction_tokens = word_tokenize(prediction)
        chencherry = SmoothingFunction()
        bleu_score1 = sentence_bleu(gt_tokens, prediction_tokens, weights=(1, 0, 0, 0),
                                    smoothing_function=chencherry.method2)
        bleu_score2 = sentence_bleu(gt_tokens, prediction_tokens, weights=(1. / 2., 1. / 2.),
                                    smoothing_function=chencherry.method2)
        bleu_score3 = sentence_bleu(gt_tokens, prediction_tokens, weights=(1. / 3., 1. / 3., 1. / 3.),
                                    smoothing_function=chencherry.method2)
        bleu_score4 = sentence_bleu(gt_tokens, prediction_tokens, weights=(1. / 4., 1. / 4., 1. / 4., 1. / 4.),
                                    smoothing_function=chencherry.method2)
        return [bleu_score1,bleu_score2,bleu_score3,bleu_score4]

    def compute_perplexity(self,sentence):
        model_bert = 'bert-base-uncased'
        model = AutoModelForMaskedLM.from_pretrained(model_bert)
        tokenizer = AutoTokenizer.from_pretrained(model_bert)
        tensor_input = tokenizer.encode(sentence, return_tensors='pt')
        repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
        labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
        with torch.inference_mode():
            loss = model(masked_input, labels=labels).loss
        return np.exp(loss.item())

    def compute_grammer(self,prediction):
        tool = language_tool_python.LanguageTool('en-US')
        matches = tool.check(prediction)
        return len(matches)

    def get_available_metrics(self):
        return ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'f1', 'ppl_scores', 'divs', 'grammer'] #, 'stats', 'words', 'count']


    def compute_scores(self,prediction,ground_truth):
        results = {
            'f1': [], 'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': [], 'ppl_scores': [], 'divs': [],
            'grammer': [] #, 'stats': [], 'words': [], 'count': []
        }
        # Calc f1
        score_f1 = self.f1_score(prediction,ground_truth)
        results['f1'].append(score_f1)

        # Calc Bleu
        bleu_scores = self.compute_bleu(prediction,ground_truth)
        results['bleu_1'].append(bleu_scores[0])
        results['bleu_2'].append(bleu_scores[1])
        results['bleu_3'].append(bleu_scores[2])
        results['bleu_4'].append(bleu_scores[3])

        # Calc Perplexity
        ppl_score = self.compute_perplexity(sentence=prediction)
        results['ppl_scores'].append(ppl_score)

        # Calc Diversity
        diversity = self.lexical_diversity(prediction)
        results['divs'].append(diversity)

        # Calc Grammer
        # grammer_score = self.compute_grammer(prediction)
        results['grammer'].append("0.0")

        return results

