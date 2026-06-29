# RAGForEduQG

**RAG-based dynamic few-shot for LLMs in educational multiple-choice question generation.**

This is an expansion of the replication package for the paper:

> **Towards Reliable LLM-based Exam Generation. Lessons Learned and Open Challenges in an Industrial Project**
> *Industry Showcase Track — ASE 2025 (40th IEEE/ACM International Conference on Automated Software Engineering), Seoul, South Korea*

This pipeline generates multiple-choice questions (MCQs) from science texts using large language models (LLMs). Unlike the paper above, this project moves beyond zero-shot prompting and explores different few-shot prompting modalities.

It uses **Retrieval-Augmented Generation (RAG)**: before asking the LLM to generate a question, the system first retrieves similar example questions from a database and injects them into the prompt as few-shot examples.

> **Dataset:** [SciQ](https://huggingface.co/datasets/allenai/sciq) — science multiple-choice questions

---

## Research Questions

| ID | Question |
|----|----------|
| **RQ1** | Do RAG-retrieved few-shot examples improve generated-question similarity to ground truth, without hurting linguistic quality? |
| **RQ2a** | Does *including* few-shot examples matter? (example presence) |
| **RQ2b** | Does the *order* of few-shot examples matter? (example ordering) |
| **RQ3** | Which LLM performs best under the optimal prompt configuration? |

---

## Project Structure

```
LLMsForEduQG/
│
├── Main.py                        # ← entry point: parse args, launch pipeline
├── LLMsForEduQG.py                # ← main orchestrator: loop over questions/prompts/models
├── config.py                      # ← ALL settings and constants (edit this, not other files)
│
├── src/
│   ├── LLM_Service.py             # API calls to OpenAI / Nebius / HuggingFace
│   ├── Prompt.py                  # prompt templates and PromptID enum
│   ├── RAG_Service.py             # ChromaDB retrieval: embed, store, query
│   ├── Metrics.py                 # BLEU, F1, BERTScore, PPL, diversity
│   ├── Statistics.py              # Wilcoxon, Kendall, A12 effect size, boxplots
│   └── MultipleChoiceQuestion.py  # Pydantic model: structured LLM output
│
├── analysis/                      # standalone analysis scripts (run after pipeline)
├── datasets/                      # SciQ CSV splits (test, train, valid)
├── results/                       # pipeline outputs (auto-created per run)
├── chroma_db/                     # ChromaDB vector store (persisted, auto-created)
├── logs/                          # nohup and run logs
├── notebooks/                     # Jupyter notebooks for exploration
└── tests/                         # unit tests
```

---

## How the Workflow Works

This is what happen when you run the pipeline.

### Step 0 — Startup

- `Main.py` parses CLI arguments (input file, models, prompts, max questions)
- It creates an instance of `LLMsForEduQG`
- `LLMsForEduQG.__init__()` loads the ground-truth questions from the input CSV into a DataFrame
- It also instantiates `Metrics`, `Statistics`, and `LLM_Service`

### Step 1 — ChromaDB warm-up (RAG only)

- `RAG_Service` connects to ChromaDB via `PersistentClient`
- On **first run**: all SciQ_train questions are embedded using `all-mpnet-base-v2` and stored (~8 min)
- On **subsequent runs**: the stored embeddings are loaded from memory (seconds)
- Collection names encode the corpus + embedding model to prevent cross-experiment collisions

### Step 2 — Outer loop: for each model

```
for each model_id:
    for each question_id:
        for each prompt_id:
```

The model loop is the outermost loop

### Step 3 — Skip already-done rows (resume logic)

- Before doing anything, the pipeline checks `generated_questions.csv`
- If a `(question_id, prompt_id, model_id)` triple already exists → skip it
- This means **interrupted runs can be safely resumed** with the same command

### Step 4 — RAG retrieval (for few-shot prompts)

- The support text of the current question is embedded
- ChromaDB returns the `TOP_K` most similar questions from SciQ_train
- These retrieved examples are injected into the prompt template

### Step 5 — Prompt construction

- `Prompt(pid, question_data)` is called with the prompt ID and the current question's data
- The template fills in: support text, correct answer (if `WithAnswer` config), and retrieved examples (if RAG config)
- The result is a plain string ready to send to the LLM

### Step 6 — LLM call

- `LLM_Service.execute_prompt(model_id, prompt_string)` routes the call:
  - `GPT*` → OpenAI API
  - `NEB_*` → Nebius API (OpenAI-compatible)
- The LLM response is parsed into a `MultipleChoiceQuestion` object (Pydantic)
- If parsing fails → `None` is returned and the row is recorded with empty fields and `0.0` scores

### Step 7 — Metric computation

- `Metrics.compute_scores(generated_question, ground_truth_question)` is called
- Returns a dict of scores: BLEU 1-4, F1, BERTScore, PPL, lexical diversity

### Step 8 — Write to CSV (immediately)

- The result row is appended to `generated_questions.csv` right away
- The file is flushed after every row — no data is lost if the run crashes

### Step 9 — Post-processing (after all rows are done)

Once all `(qid, pid, mid)` triples are processed, the pipeline runs:

1. `clean_generated_questions()` → removes rows where `question` or `correct_answer` is empty → writes `clean_generated_questions.csv`
2. `generate_summary()` → computes mean metrics per (prompt, model) pair → writes `summary.csv`
3. `compute_statistics()` → runs Wilcoxon, Kendall, A12 for all prompt comparisons → writes `statistics.csv`
4. `generate_plots()` → saves one boxplot PDF per metric

### Data flow diagram

```
datasets/SciQ_test.csv
        │
        ▼
  LLMsForEduQG.py ──── loads questions
        │
        ├─── RAG_Service.py ─── ChromaDB (SciQ_train)
        │         │                    │
        │     embed query         retrieve TOP_K
        │         └──────────────────▶ few-shot examples
        │
        ├─── Prompt.py ─── fills template (support + answer + examples)
        │
        ├─── LLM_Service.py ─── sends prompt to API ──▶ LLM
        │                                               │
        │                                  MultipleChoiceQuestion (Pydantic)
        │
        ├─── Metrics.py ─── scores generated vs ground truth
        │
        └─── Statistics.py ─── writes CSVs + boxplot PDFs
                    │
                    ▼
           results/
           ├── generated_questions.csv
           ├── clean_generated_questions.csv
           ├── summary.csv
           ├── statistics.csv
           └── *.pdf (one per metric)
```

---

## File Reference

### `Main.py`

**Entry point.** Parses CLI arguments and launches the pipeline.

**Responsibilities:**
- Parse and validate all arguments (`argparse`)
- Instantiate `LLMsForEduQG`
- Filter the model list to only supported models
- Map prompt name strings to `PromptID` enum values
- Call `LLMrunner.run_per_qid()`

**When to touch this file:**
- Adding a new CLI argument
- Changing how models or prompts are selected from the command line

**Does NOT contain:**
- Metric computation
- File I/O

---

### `LLMsForEduQG.py`

**Main orchestrator.** The core loop of the pipeline.

**Class:** `LLMsForEduQG`

| Method | What it does |
|--------|-------------|
| `__init__(input_filename, results_dir, MAX, random_choice)` | Loads dataset, creates output dir, instantiates Metrics, Statistics, LLM_Service |
| `load_testing_data(inputname, MAX, random_choice)` | Reads ground-truth CSV into `self.ground_truth_questions` DataFrame |
| `is_valid_context(qid)` | Returns `False` if the `support` text is empty — those questions are skipped |
| `generate_prompts(qid, pid)` | Builds a `Prompt` object for a given question + prompt ID |
| `execute(qid, pid, mid)` | Sends prompt to LLM, scores the response, stores result in `self.generated_questions` |
| `report(qid, pid, mid, first_time)` | Appends the result row to the CSV output file |
| `run_per_qid(prompt_ids, model_ids)` | Outer loop: iterates over all (model, question, prompt) combinations |

**Key design detail:**
- `execute()` handles three cases: valid response, `None` (LLM failed), `"error=429"` (rate limit)
- `report()` writes **one row at a time**, immediately after each call — no batching, no data loss on crash
- `run_per_qid()` calls post-processing steps at the end: clean → summary → statistics → plots

**When to touch this file:**
- Changing the loop order (model / question / prompt)
- Changing how failed responses are recorded
- Adding new post-processing steps

---

### `config.py`

**All settins are set here.**

All constants extracted from other files live here. If you need to change a path, model name, threshold, or toggle — edit this file only.

```python
# Paths
DATASET_PATH        = "datasets/SciQ_test.csv"
RAG_CORPUS_PATH     = "datasets/SciQ_train.csv"
RESULTS_DIR         = "results/"
CHROMA_PERSIST_DIR  = "chroma_db/"

# Active models
ACTIVE_MODELS = [
    "NEB_Llama3370Instruct",
    "NEB_Qwen3235BInstruct",
    "GPT54Mini",
]

# RAG
EMBEDDING_MODEL     = "sentence-transformers/all-mpnet-base-v2"
RAG_TOP_K           = 3

# Metrics
BERTSCORE_MODEL     = "bert-base-uncased"
GRAMMAR_ENABLED     = False      # needs Java 17; cluster has Java 11
```

**Rule:** Add settings, paths, etc. here instead of hardcoding them in the other files.

---

### `src/LLM_Service.py`

**Handles all LLM API calls.** Routes requests to the correct API based on model ID.

**Class:** `LLM_Service`

| Method | What it does |
|--------|-------------|
| `execute_prompt(model_id, prompt)` | Router: picks the right API method based on model prefix |
| `gpt_execute_prompt(model_id, prompt)` | Calls OpenAI chat completions API |
| `gpt_old_execute_prompt(model_id, prompt)` | Calls legacy OpenAI completions API (GPT-3.5-Turbo-Instruct only) |
| `hf_execute_prompt(model_id, prompt)` | Calls HuggingFace Inference API (most HF models) - LEGACY |
| `hf_execute_prompt_Llama32(model_id, prompt)` | Calls HuggingFace via `InferenceClient` (Llama 3.2 specifically)  - LEGACY |
| `get_all_models()` | Returns list of all supported model IDs |
| `get_model_url(model_id)` | Maps model ID → API model string |
| `get_model_ids_startswith(prefix)` | Filters models by prefix (e.g., `"GPT"`, `"NEB_"`) |

**Routing logic:**
```
model_id starts with "GPT35TurboInstruct" → gpt_old_execute_prompt - LEGACY
model_id starts with "GPT"               → gpt_execute_prompt
model_id starts with "Llama32"           → hf_execute_prompt_Llama32 - LEGACY
everything else                          → hf_execute_prompt - LEGACY
```

**Output:** Always returns a `MultipleChoiceQuestion` object, or `None` on failure, or `"error=429"` on rate limit.

**API keys used:**
- `OPENAI_API_KEY` → OpenAI
- `NEBIUS_API_KEY` → Nebius (OpenAI-compatible, base URL in config)
- `API_KEY_HUGGINGFACE` → HuggingFace - LEGACY

**When to touch this file:**
- Adding a new model provider (e.g., Anthropic, Cohere)
- Changing timeout or retry logic
- Adding a new routing rule

---

### `src/Prompt.py`

**Defines all prompt templates.** Each prompt is a named configuration in the `PromptID` enum.

**Class:** `PromptID` (Enum)

| Value | Description |
|-------|-------------|
| `Simple` | Zero-shot. Support text only. No answer, no examples. |
| `Simple_plus_Answer` | Zero-shot. Support text + correct answer. |
| *(more — see file)* | RAG few-shot variants, with/without answer conditioning |

**Class:** `Prompt`

| Method | What it does |
|--------|-------------|
| `__init__(id, question)` | Takes a `PromptID` and a question dict, builds the prompt string |
| `instantiate_prompt_template()` | Fills the template based on `self.id` |

**What a `question` dict contains:**
```python
{
    "support":        "...",   # science passage
    "correct_answer": "...",   # ground truth answer
    "question":       "...",   # ground truth question (NOT given to LLM)
    "distractor1":    "...",
    "distractor2":    "...",
    "distractor3":    "..."
}
```

> The ground-truth `question` text is **never included** in the prompt. Only the `support` text and `correct_answer` (in WithAnswer configs) are given to the LLM.

**Key finding:** Prompts that include the correct answer (`WithAnswer`) are the dominant driver of BLEU/F1 improvement. Example ordering shows no statistically significant effect (A12 ≈ 0.50).

**When to touch this file:**
- Adding a new prompt configuration → add a new `PromptID` value and a new `elif` block
- Changing how few-shot examples are formatted in the prompt

---

### `src/RAG_Service.py`

**Retrieval-Augmented Generation engine.** Embeds questions, stores them in ChromaDB, retrieves similar ones.

**Class:** `RAG_Service`

| Method | What it does |
|--------|-------------|
| `__init__(corpus_path, embed_model)` | Connects to ChromaDB, loads embedding model |
| `build_corpus()` | Embeds all questions in the corpus and stores them (only if collection is empty) |
| `retrieve(query_text, top_k)` | Embeds the query, returns `top_k` most similar questions |

**ChromaDB setup:**
- Uses `PersistentClient` — data is saved to disk at `CHROMA_PERSIST_DIR`
- Collection names encode: corpus path + embedding model → prevents silent collisions across experiments
- First run: ~8 minutes to embed all of SciQ_train
- Subsequent runs: loads in seconds

**Embedding model:** `sentence-transformers/all-mpnet-base-v2`

> **Self-retrieval leakage warning:** If the RAG corpus and the eval corpus overlap (e.g., both use SciQ_test), the target question will appear as its own retrieved example ~65% of the time. This is why the RAG corpus must always be `SciQ_train`.

**When to touch this file:**
- Changing the embedding model
- Adding a similarity threshold for retrieval (currently removed: post-hoc filtering preferred)
- Switching to a different vector database

---

### `src/MultipleChoiceQuestion.py`

**Data model for LLM output.** A Pydantic `BaseModel` that defines the expected JSON structure.

**Class:** `MultipleChoiceQuestion`

| Field | Type | Description |
|-------|------|-------------|
| `question` | `str` | The generated question text |
| `correct_answer` | `Optional[str]` | The correct answer |
| `distractor1–3` | `Optional[str]` | The three wrong answers |
| `support` | `Optional[str]` | Support text — **excluded from LLM schema** |

**Why `support` is excluded:**
- `SkipJsonSchema` annotation prevents LangChain from asking the LLM to generate the `support` field
- Instead, `execute()` in `LLMsForEduQG.py` sets `support` to the ground-truth value explicitly

**Parsing flow:**
```
LLM raw text → PydanticOutputParser → MultipleChoiceQuestion object
```
If parsing fails (invalid JSON, missing fields) → `ValidationError` is caught → method returns `None`

**When to touch this file:**
- Adding or removing a field from the LLM output (e.g., adding `explanation`)
- Changing which fields the LLM is asked to generate

---

### `src/Metrics.py`

**Computes all evaluation metrics** for a generated question vs. the ground-truth question.

**Class:** `Metrics`

| Method | What it measures | Notes |
|--------|-----------------|-------|
| `compute_bleu(prediction, gt)` | BLEU-1 to BLEU-4 | N-gram overlap; uses SmoothingFunction method2 |
| `f1_score(prediction, gt)` | Token F1 | Precision + recall on normalized token bags |
| `compute_perplexity(sentence)` | PPL | Masked LM perplexity via BERT; lower = more fluent |
| `lexical_diversity(prediction)` | Distinct 3-grams / total words | Higher = more varied vocabulary |
| `compute_grammer(prediction)` | Grammar error count | **Disabled** — requires Java 17 |
| `normalize_answer(s)` | Preprocessing | Lowercases, strips punctuation and articles |
| `compute_scores(prediction, gt)` | Calls all metrics, returns dict | Main entry point called from `LLMsForEduQG.execute()` |
| `get_available_metrics()` | Returns list of metric names | Used to build CSV headers dynamically |

**BERTScore:**
- Uses a `BERTScorer` object instantiated once at startup (not per call) — avoids reloading the model for every question
- Model: `bert-base-uncased`

**Evaluation philosophy:**
- BLEU/F1/PPL enable **cross-model comparison** with prior work (T5, BART baselines)
- BERTScore + LLM-as-judge used as a **complementary lens** within the LLM subset
- These two perspectives are a deliberate two-perspective framing (per supervisor direction)

> BLEU/F1 measure **annotator style conformity**, not generation quality. 

**When to touch this file:**
- Adding a new metric → add a method + add its key to `get_available_metrics()`
- Re-enabling grammar → set `GRAMMAR_ENABLED = True` in `config.py` (requires Java 17)

---

### `src/Statistics.py`

**Runs all statistical tests and generates visualizations** after the pipeline completes.

**Class:** `Statistics`

| Method | What it does |
|--------|-------------|
| `clean_generated_questions()` | Removes rows with empty `question` or `correct_answer` → saves `clean_generated_questions.csv` |
| `generate_summary()` | Computes mean per (prompt, model) pair → saves `summary.csv` |
| `compute_statistics()` | Runs Wilcoxon, Kendall, A12 for all (model, metric, prompt_A vs prompt_B) → saves `statistics.csv` |
| `generate_plots()` | One seaborn boxplot per metric, all models × prompts → saves `*.pdf` |
| `VD_A(treatment, control)` | Computes Vargha-Delaney A12 effect size |

**Statistical tests explained:**

| Test | What it answers | Significance |
|------|----------------|-------------|
| Wilcoxon signed-rank | Is there a significant difference between two prompt configs? | p < 0.05 |
| Kendall τ | Are the rankings between two configs consistent across questions? | p < 0.05 |
| A12 (Vargha-Delaney) | How large is the effect? | **Primary metric for interpretation** |

**A12 thresholds (Hess & Kromrey, 2004):**

| A12 value | Effect size |
|-----------|-------------|
| 0.50 | No effect |
| > 0.56 | Small |
| > 0.64 | Medium |
| > 0.71 | Large |

> Wilcoxon p-values of `0.0000` are **not errors** — they are truncated at 4 decimal places due to large sample size (N=884). Always use A12 as the primary interpretive metric.

> `np.allclose` guard can silently prevent Wilcoxon computation when treatment and control values are nearly identical. Fixed with `zero_method='zsplit'` + `try/except`.

**When to touch this file:**
- Adding a new statistical test
- Changing how plots look
- Adding new output files (e.g., per-question score CSVs)

---

## Configuration (`config.py`)

**All constants live here.** Do not hardcode values in other files.

```python
# Paths
DATASET_PATH        = "datasets/SciQ_test.csv"
RAG_CORPUS_PATH     = "datasets/SciQ_train.csv"
RESULTS_DIR         = "results/"
CHROMA_PERSIST_DIR  = "chroma_db/"

# Active models (Nebius + OpenAI)
ACTIVE_MODELS = [
    "NEB_Llama3370Instruct",
    "NEB_Qwen3235BInstruct",
    "GPT54Mini",
]

# RAG settings
EMBEDDING_MODEL     = "sentence-transformers/all-mpnet-base-v2"
RAG_TOP_K           = 3

# Metrics
BERTSCORE_MODEL     = "bert-base-uncased"
GRAMMAR_ENABLED     = False      # needs Java 17; cluster has Java 11
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/rdegiovanni/LLMsForEduQG.git
cd LLMsForEduQG
git checkout development
```

### 2. Install dependencies

```bash
# On the HPC cluster (uv, no sudo)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# On local machine
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows PowerShell
pip install -r requirements.txt
```

### 3. Set API keys

```bash
# Permanent — add to ~/.bashrc on cluster
export OPENAI_API_KEY="sk-..."
export NEBIUS_API_KEY="..."
```

Or create a `.env` file at the project root (never commit this):

```
OPENAI_API_KEY=sk-...
NEBIUS_API_KEY=...
```

---

## Running the Pipeline

### Basic run

You can run the pipeline via Python directly or via the provided shell script wrapper:

```bash
# Using the shell script (recommended)
./LLMsForEduQG.sh \
  -i datasets/SciQ_test.csv \
  -d results/my_run \
  -m NEB_Llama3370Instruct,GPT54Mini \
  -p Simple,Simple_plus_Answer
```

```bash
# Or directly with Python
python Main.py \
  -i datasets/SciQ_test.csv \
  -d results/my_run \
  -m NEB_Llama3370Instruct,GPT54Mini \
  -p Simple,Simple_plus_Answer
```

### Full production run (HPC, background)

```bash
nohup python Main.py \
  -i datasets/SciQ_test.csv \
  -d results/full_run \
  -m NEB_Llama3370Instruct,NEB_Qwen3235BInstruct,GPT54Mini \
  -N 0 \
  > logs/full_run.log 2>&1 &
```

Monitor progress:

```bash
tail -f logs/full_run.log
wc -l results/full_run/generated_questions.csv
ps -p <PID>
```

### CLI arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-i` | Input CSV (ground truth questions) | **required** |
| `-d` | Output directory | `results/<uuid>` |
| `-N` | Max questions (0 = all) | `0` |
| `-r` | Random question order | `False` |
| `-m` | Comma-separated model IDs | all models |
| `-p` | Comma-separated prompt IDs | all prompts |
| `-sw` | Select all models starting with prefix | — |
| `-ll` | Print list of all supported LLMs | — |
| `-pl` | Print list of all prompt IDs | — |

### List available prompts and models

```bash
python Main.py -i datasets/SciQ_test.csv -ll -pl
```

---

## Output Files

All outputs go into the directory specified by `-d`.

| File | Contents | When it's written |
|------|----------|-------------------|
| `generated_questions.csv` | Raw LLM output + scores for every `(qid, pid, mid)` triple | Row by row, during the run |
| `clean_generated_questions.csv` | Same, with empty/broken rows removed | After the full run |
| `summary.csv` | Mean metrics per (prompt, model) | After the full run |
| `statistics.csv` | Wilcoxon, Kendall, A12 for all comparisons | After the full run |
| `*.pdf` | One boxplot per metric | After the full run |

### Column reference — `generated_questions.csv`

| Column | Description |
|--------|-------------|
| `question_id` | ID from the input dataset |
| `prompt_id` | Name of the prompt config used |
| `model_id` | Name of the model used |
| `question` | Generated question text (empty if model failed) |
| `correct_answer` | Generated correct answer |
| `distractor1–3` | Generated distractors |
| `support` | Ground-truth support text (set by pipeline, not generated) |
| `bleu_1–4` | BLEU scores |
| `f1` | Token F1 score |
| `bertscore` | BERTScore F1 |
| `ppl_scores` | Perplexity |
| `divs` | Lexical diversity |


### Resume logic

The pipeline **skips already-completed `(qid, pid, mid)` triples** automatically.
If the run is interrupted, re-run the same command — it will continue from where it stopped.

---

## Metrics

| Metric | What it measures | Better when |
|--------|-----------------|-------------|
| `bleu_1–4` | N-gram overlap vs. ground truth | Higher |
| `f1` | Token-level precision + recall vs. ground truth | Higher |
| `bertscore` | Semantic similarity (BERT embeddings) | Higher |
| `ppl_scores` | Fluency — how natural the sentence is | **Lower** |
| `divs` | Vocabulary variety (distinct 3-grams) | Higher |
| `grammer` | Grammar error count | Lower (disabled) |
| `success_ratio` | Fraction of questions successfully generated | Higher |

> **BLEU/F1 measure annotator style conformity, not generation quality.**

---

## RAG Service

The RAG system retrieves similar questions from SciQ_train and injects them as few-shot examples.

- **Embedding model:** `sentence-transformers/all-mpnet-base-v2`
- **Vector DB:** ChromaDB `PersistentClient` — saved to `chroma_db/`
- **First run:** ~8 min to embed all SciQ_train questions
- **Subsequent runs:** loads in seconds
- **Corpus separation:** RAG = SciQ_train. Eval = SciQ_test. **Never overlap these.**


## Active Models

| Model ID | Provider | Underlying model |
|----------|----------|-----------------|
| `NEB_Llama3370Instruct` | Nebius | Meta Llama 3.3 70B Instruct |
| `NEB_Qwen3235BInstruct` | Nebius | Qwen3 235B Instruct |
| `GPT54Mini` | OpenAI | GPT-4.5 Mini |

Nebius API base URL: `https://api.tokenfactory.nebius.com/v1/`



## Dataset Format

Input CSV must have exactly these columns:

```
question_id, set, question, correct_answer, distractor1, distractor2, distractor3, support
```

| Column | Description |
|--------|-------------|
| `question_id` | Unique string identifier |
| `set` | Dataset split (`train` / `valid` / `test`) |
| `question` | Ground-truth question text |
| `correct_answer` | Ground-truth correct answer |
| `distractor1–3` | Ground-truth wrong answers |
| `support` | Science passage the question was derived from |

> Questions with empty `support` are **skipped automatically** by `is_valid_context()`.

---

## For Collaborators

1. **Never hardcode** paths, model names, or thresholds — put them in `config.py`
2. **Never commit** `.env`, API keys, `__pycache__`, `.DS_Store`, or `results/`
3. **Branch:** always work on `development`, not `main`
4. **Statistics** are run as standalone scripts in `analysis/`, not inline in the pipeline
5. **Before a new run:** check `logs/` for active `nohup` processes to avoid collision
6. **Results** are reviewed by Renzo Degiovanni before going into the paper
7. **New metric?** Add the method to `Metrics.py` + add its name to `get_available_metrics()` — the rest of the pipeline picks it up automatically
8. **New prompt?** Add a `PromptID` value + an `elif` block in `instantiate_prompt_template()` — nothing else needs changing

---

## Running Tests

```bash
python -m pytest tests/
```

Run a specific test:

```bash
python -m pytest tests/LLMsForEduQGTest.py::MyTestCase::test_only_GPT
```

