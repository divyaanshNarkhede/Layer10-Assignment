"""
run_pipeline.py – Execute the full Layer10 memory pipeline end-to-end.

Steps:
  1. Download / prepare Enron corpus
  2. Structured extraction (Ollama + rule-based fallback)
  3. Deduplication & canonicalization
  4. Build memory graph
  5. Retrieval demo (generate context packs)

Usage:
    python run_pipeline.py           # full pipeline
    python run_pipeline.py --skip-extract  # skip LLM extraction (use cached)
"""
import argparse, os, sys, time, json

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    CORPUS_RAW_PATH, EXTRACTION_PATH, GRAPH_PATH,
    CONTEXT_PACKS_PATH, OUTPUT_DIR
)


def main():
    parser = argparse.ArgumentParser(description="Layer10 Memory Pipeline")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip corpus download (use existing)")
    parser.add_argument("--skip-extract", action="store_true",
                       help="Skip extraction (use cached results)")
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM extraction (rule-based only, fast)")
    args = parser.parse_args()

    if args.no_llm:
        os.environ["DISABLE_OLLAMA"] = "1"

    start = time.time()

    print("\n" + "═" * 60)
    print("  Layer10 Grounded Long-Term Memory Pipeline")
    print("  Corpus: Enron Email Dataset")
    print("═" * 60)

    from importlib import import_module

    # ─── Step 1: Corpus ──────────────────────────────────────────────────
    if not args.skip_download or not os.path.exists(CORPUS_RAW_PATH):
        print("\n▶ STEP 1: Preparing corpus...")
        dl = import_module("01_download_corpus")
        dl.main()
    else:
        print("\n▶ STEP 1: Using cached corpus")

    # ─── Step 2: Extraction ──────────────────────────────────────────────
    if not args.skip_extract or not os.path.exists(EXTRACTION_PATH):
        print("\n▶ STEP 2: Running structured extraction...")
        extract = import_module("02_extract")
        extract.run_extraction()
    else:
        print("\n▶ STEP 2: Using cached extractions")

    # ─── Step 3: Dedup ───────────────────────────────────────────────────
    print("\n▶ STEP 3: Deduplication & canonicalization...")
    dedup = import_module("03_dedup")
    dedup.run_dedup()

    # ─── Step 4: Graph ───────────────────────────────────────────────────
    print("\n▶ STEP 4: Building memory graph...")
    graph = import_module("04_graph")
    graph.run_graph_build()

    # ─── Step 5: Retrieval ───────────────────────────────────────────────
    print("\n▶ STEP 5: Running retrieval demo...")
    retrieve = import_module("05_retrieve")
    retrieve.run_retrieval()

    elapsed = time.time() - start
    print(f"\n{'═' * 60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Outputs in: {OUTPUT_DIR}")
    print(f"{'═' * 60}")
    print(f"\nNext steps:")
    print(f"  1. Review outputs:  ls {OUTPUT_DIR}/")
    print(f"  2. Launch UI:       streamlit run 06_app.py")
    print(f"  3. Read write-up:   cat WRITEUP.md")


if __name__ == "__main__":
    main()
