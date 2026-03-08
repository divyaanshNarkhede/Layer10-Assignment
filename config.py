"""
Configuration constants for the Layer10 memory pipeline.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CORPUS_RAW_PATH = os.path.join(DATA_DIR, "enron_raw_emails.json")
EXTRACTION_PATH = os.path.join(OUTPUT_DIR, "extractions.json")
GRAPH_PATH = os.path.join(OUTPUT_DIR, "memory_graph.json")
CONTEXT_PACKS_PATH = os.path.join(OUTPUT_DIR, "context_packs.json")

# ─── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_MODEL = "llama3.2:3b"       # change to your pulled model
OLLAMA_HOST = "http://localhost:11434"

# ─── Embedding ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.85        # cosine similarity for entity merging

# ─── Extraction ──────────────────────────────────────────────────────────────
MAX_RETRIES = 3                    # retry on invalid JSON
CONFIDENCE_THRESHOLD = 0.5         # minimum confidence to keep a claim
BATCH_SIZE = 5                     # emails per extraction batch

# ─── Corpus (thread-wise) ────────────────────────────────────────────────────
TARGET_EMAIL_COUNT = 75            # legacy – individual email cap
MIN_MESSAGES_PER_THREAD = 5        # minimum emails to form a thread
MAX_MESSAGES_PER_THREAD = 15       # cap emails per thread
TARGET_THREAD_COUNT = 15           # how many threads to extract

# Ensure dirs exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
