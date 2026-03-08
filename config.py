"""
All the knobs and dials for the pipeline in one place.

If you want to use a different model, change a threshold, or point
to a different data directory — this is where you do it.
"""
import os

# ── Where data lives and where outputs go ────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CORPUS_RAW_PATH = os.path.join(DATA_DIR, "enron_raw_emails.json")
EXTRACTION_PATH = os.path.join(OUTPUT_DIR, "extractions.json")
GRAPH_PATH = os.path.join(OUTPUT_DIR, "memory_graph.json")
CONTEXT_PACKS_PATH = os.path.join(OUTPUT_DIR, "context_packs.json")

# ── Local LLM settings — assumes Ollama is running on localhost ─────────────
OLLAMA_MODEL = "llama3.2:3b"       # change to your pulled model
OLLAMA_HOST = "http://localhost:11434"

# ── Embedding model used for semantic entity matching during dedup ───────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.85        # cosine similarity for entity merging

# ── Extraction tuning — retry logic and confidence filtering ────────────────
MAX_RETRIES = 3                    # retry on invalid JSON
CONFIDENCE_THRESHOLD = 0.5         # minimum confidence to keep a claim
BATCH_SIZE = 5                     # emails per extraction batch

# ── How we slice the corpus into threads for the LLM ────────────────────────
TARGET_EMAIL_COUNT = 75            # legacy – individual email cap
MIN_MESSAGES_PER_THREAD = 5        # minimum emails to form a thread
MAX_MESSAGES_PER_THREAD = 15       # cap emails per thread
TARGET_THREAD_COUNT = 15           # how many threads to extract

# Make sure the data and output folders exist on first import — nothing to configure here
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
