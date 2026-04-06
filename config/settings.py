import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHROMA_DIR = BASE_DIR / "chroma_db"

# ── Domain folders ─────────────────────────────────────
DOMAINS = ["finance", "legal", "hr"]

# ── Chunking ───────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── Embedding model ────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Chroma collection name ─────────────────────────────
COLLECTION_NAME = "enterprise_docs"

# ── LLM ────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama3-8b-8192"
TEMPERATURE = 0

# ── Retrieval ──────────────────────────────────────────
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 5
SIMILARITY_THRESHOLD = 0.3