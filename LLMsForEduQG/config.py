# ── APIs ──────────────────────────────────────────────────
NEBIUS_BASE_URL     = "https://api.tokenfactory.nebius.com/v1/"
HF_API_BASE_URL     = "https://api-inference.huggingface.co/models"
LLM_TIMEOUT         = 120

# ── RAG ───────────────────────────────────────────────────
RAG_COLLECTION_NAME         = "sciq_examples"
RAG_DEFAULT_SIMILARITY      = 1.0   # no filtering by default
RAG_PIPELINE_SIMILARITY     = 0.3   # used in LLMsForEduQG pipeline
CHROMA_PERSIST_DIR = "chroma_store"
# ── Metrics ───────────────────────────────────────────────
BERT_MODEL = "bert-base-uncased"

# ── Statistics: result filenames ──────────────────────────
GENERATED_QUESTIONS_FILE    = "generated_questions.csv"
CLEAN_GENERATED_FILE        = "clean_generated_questions.csv"
CHOICES_QUALITY_FILE        = "choices_quality_analysis.csv"
CHOICES_SUMMARY_FILE        = "choices_summary.csv"
STATISTICS_FILE             = "statistics.csv"
SUMMARY_FILE                = "summary.csv"

# ── Effect size thresholds (Hess & Kromrey 2004) ──────────
VDA_LEVELS      = [0.147, 0.33, 0.474]
VDA_MAGNITUDE   = ["negligible", "small", "medium", "large"]