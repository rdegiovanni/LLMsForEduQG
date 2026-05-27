# Project Phases

## Phase 1 — Example Order Study
### Description
Investigate how the order of RAG-retrieved examples affects question quality.
- Three orderings tested: most similar first (default), least similar first (reverse), random
- Model: GPT-5 mini
- Dataset: SciQ validation set, 100 datapoints

### Results
- Reversed order (least similar first) initially showed higher scores
- After prompt normalization (consistent wording, punctuation aligned with `Simple_plus_Answer`), the difference disappeared
- `statistics.csv` confirms example order does **not** significantly impact F1 or BLEU
- **Exception**: `FewShot` (most similar first) produces significantly more fluent output (lower PPL)

### Decision
- Go forward with `FewShot` (most similar first) as the default few-shot ordering
- Rename variants for clarity:
  - `FewShot_Reverse_NoContext` → `FewShot_NoContext`
  - `FewShot_Reverse_NoAnswer` → `FewShot_NoAnswer`

---

## Phase 2 — Prompt Ablation Study
### Description
Ablation study to identify which prompt components drive question quality.
- Goal: generate **novel** questions grounded in source material
- Model: GPT-5 mini
- Dataset: SciQ validation set, 100 examples

### Results
- **Context matters**: removing the support text caused the worst performance (F1: 0.07–0.14); without it, the model generates off-topic questions
- **Answer hurts novelty**: including the correct answer inflates F1/BLEU because the model reverse-engineers the original question rather than generating a new one — not useful for the actual goal
- **Few-shot helps**: RAG examples improve structure and MCQ format (F1: 0.254 vs 0.207 zero-shot; BLEU-1: 0.243 vs 0.198)
- **Answer in few-shot examples is negligible**: `FewShot` vs `FewShot_NoAnswer` nearly identical on F1/BLEU; `FewShot` had slightly better fluency (PPL 11.05 vs 12.91)

### Decision
Go forward with **`FewShot`** (support text + RAG examples, no answer provided) as the golden config for cross-model evaluation.

---

## Phase 3 — Ablation with Similarity Threshold (RQ2 Rerun)
### Description
Rerun the ablation study with two key changes requested after supervisor review:
1. **Similarity threshold on RAG retrieval** (~70%, i.e. cosine distance ≤ 0.30): only include few-shot examples that are sufficiently similar to the query — some questions may receive fewer than 3 examples
2. **Reduced to 4 focused prompt modalities** (dropping order and context variants):

| Modality | Support Text | Answer in Prompt | Few-Shot Examples |
|---|---|---|---|
| `Simple` | ✅ | ❌ | ❌ |
| `Simple_plus_Answer` | ✅ | ✅ | ❌ |
| `FewShot_Random` | ✅ | ❌ | ✅ (random order) |
| `FewShot_WithAnswer` | ✅ | ✅ | ✅ (random order) |

- Model: GPT-5.4 mini only
- Dataset: SciQ validation set, 100 examples

### Results
_TBD_

### Decision
_TBD — select best modality from `summary.csv` to carry forward as the new golden config_

---

## Phase 4 — Full Cross-Model Run
### Description
Run the 4 prompt modalities from Phase 3 on the full ScienceQA test set across 3 models.
- Models: GPT-5.4 mini, Llama 3.3 70B, DeepSeek V3 (Qwen excluded)
- Dataset: ScienceQA test set (MC split)
- Similarity threshold applied to RAG retrieval (same as Phase 3)
- Run only if time permits after Phase 3 analysis is complete

### Results
_TBD_