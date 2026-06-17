# Project Phases
## Phase 1
### Description
Create 3 different prompt modalities wrt the order of the examples presented to the model (gpt5).
The order of the RAG examples are: most similar first (default), least similar first (reverse), random order
### Results
Presenting the RAG examples in a **reversed** oder (least similar first), generally resulted in higher scores for the generated questions.
HOWEVER, after I have slightly changed the prompt (adding punctuation, consistent wording with the best prompt we had, i.e. Simple_with_Answer). Moreover statistics.csv shows that example order doesn't significantly impact question quality (F1, BLEU)
FewShot Default (most similar first) produces significantly more fluent output (lower PPL) (in this case this is significant)

# Next Steps:
This means the ablation prompt types should use FewShot instead of FewShot_Reverse.
FewShot_Reverse_NoContext → should become FewShot_NoContext
FewShot_Reverse_NoAnswer → should become FewShot_NoAnswer

## Phase 2
### Description
Ablation Study to select the best prompt. I have run the ablation study checking which parts of the prompts yield the best result, given the goal of generating novel questions based on the same material. 
### Results
* Use Context: Removing the support text from the prompt caused the worst performance across all configs (F1 of 0.07 and 0.14). Without it, the model generates questions unrelated to the source material.
* Do not include answer: Providing the answer inflates similarity scores but limits novelty. Simple+Answer scored highest on F1/BLEU because the model reverse-engineers the original question. Not useful when the goal is generating new questions.
* Use examples: Few-shot examples improve structure without over-constraining. FewShot outperformed the zero-shot baseline (F1 0.254 vs 0.207, BLEU-1 0.243 vs 0.198) — the RAG-retrieved examples teach the model MCQ format and style.
* Answer in few-shot doesn't change things: Removing the answer from few-shot barely changed results. FewShot vs FewShot_NoAnswer were nearly identical, but FewShot had better fluency (PPL 11.05 vs 12.91).
Decision: Go forward with FewShot (context + examples, no answer provided) for cross-model evaluation.
## Phase 3
### Description
### Results
## Phase 4

---

# RAG
## Overlap Analysis
We used 2 different vector stores: one embedding the support text, the other embedding the question-answer pairs. We then used them for retrieval and we calculated the overlap between the examples retrieved from both datastores.
Results: 
```
mean_jaccard,overlap_percentage,no_examples_support,no_examples_qa
0.25,0.61,0.22,0.17
```
So:
- on average, 25% of the examples are shared between the 2 methods
- 61% of the queries have AT LEAST 1 shared example
- 22% of the queries found no similar support text within the threshold (0.65 -> chosen by applying the elbow method)
- 17% of the queries found no similar question-answer pair within the threshold (0.55)
