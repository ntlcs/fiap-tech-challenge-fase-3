from pathlib import Path


# ============================================================
# Diretórios do projeto
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = DATA_DIR / "database"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"


# ============================================================
# Banco de dados
# ============================================================

DB_PATH = DATABASE_DIR / "hospital.db"


# ============================================================
# Protocolos e RAG
# ============================================================

PROTOCOL_PATH = (
    SYNTHETIC_DIR
    / "protocolos_sinteticos.csv"
)

VECTOR_DB_DIR = (
    DATABASE_DIR
    / "faiss_protocolos"
)

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# LLM
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

MODEL_DIR = (
    MODELS_DIR
    / "qwen2.5_0.5b_lora"
)


# ============================================================
# Auditoria
# ============================================================

LOGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_PATH = (
    LOGS_DIR
    / "clinical_assistant.jsonl"
)


# ============================================================
# Parâmetros da aplicação
# ============================================================

RAG_TOP_K = 2

MAX_NEW_TOKENS = 120

HUMAN_VALIDATION_REQUIRED = True