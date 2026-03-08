# Layer10 - Memory System

A pipeline and interactive visualization tool for extracting, deduplicating, and querying a grounded long-term memory graph from the Enron email corpus. It uses local LLMs (via Ollama) and a rule-based extraction fallback to build a robust, structured knowledge graph of entities and claims, which are strictly grounded to the original source text.

## Features

- **End-to-End Pipeline**: Automatically downloads the CMU Enron corpus (or uses a built-in synthetic corpus), groups emails into conversation threads, and extracts intelligence.
- **Dual Extraction Strategy**: Leverages Ollama (default: `llama3.2:3b`) for semantic depth, paired with a deterministic rule-based extractor for reliable baseline coverage.
- **Exact Evidence Grounding**: Every extracted claim maintains bidirectional pointers to the exact source document, complete with rigorous character-level offsets (`char_offset_start`, `char_offset_end`).
- **Graph Deduplication (3 Layers)**:
  1. *Artifact Dedup*: Collapses near-identical quoted/forwarded content to prevent duplicate extraction.
  2. *Entity Canonicalization*: Resolves identities using `all-MiniLM-L6-v2` embeddings in a Union-Find data structure.
  3. *Claim Dedup / Conflict Resolution*: Merges redundant facts and chronologically organizes contradictory claims (e.g., changing job roles over time).
- **Interactive Streamlit UI**: A clean, responsive front-end for semantic search, grounded Q&A, and interactive network visualization (powered by `vis-network.js`).

## Prerequisites

- **Python 3.10+** (Python 3.12 recommended)
- **Ollama**: Installed and running locally. 
  - Install from [ollama.com](https://ollama.com/)
  - Pull the model: `ollama pull llama3.2:3b`

## Installation

1. **Clone the repository** (or navigate to the workspace directory).
2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Build the Memory Graph

Run the master pipeline script to seamlessly download the dataset, parse threads, extract knowledge, deduplicate entities, and generate the final knowledge graph. 

Because we process the dataset in batches, large extraction runs are saved via checkpoints, so you won't lose your data if interrupted.

```bash
python run_pipeline.py
```

**Optional Flags:**
- `--skip-download`: Skip downloading the dataset (use existing).
- `--skip-extract`: Skip the LLM extraction phase (use existing `extractions.json`).
- `--no-llm`: Disable LLM usage entirely (runs rule-based extraction only, useful for fast testing).

### 2. Launch the Explorer UI

Once the graph is built (`output/memory_graph.json` exists), start the interactive Streamlit explorer:

```bash
streamlit run 06_app.py
```

### 3. Navigation
- **Query**: Run semantic questions (e.g., "Who warned about accounting fraud?"). You can toggle "LLM Answer" on to synthesize an evidence-backed text response via Ollama, or look at the top-ranked ground claims.
- **Graph**: Explore the nodes and edges visually using a force-directed layout.
- **Evidence**: Browse all extracted entities and claims, mapping them exactly to their origin source text.
- **Merges**: View the audit log of entity deduplication. You can check the `reversible` status of graph updates.
- **Statistics**: View overall statistics, density, degree centrality, and coverage metrics.

## Project Structure

- `schema.py`: Core Pydantic data models (`Entity`, `Claim`, `MemoryStore`, `MergeRecord`).
- `config.py`: High-level configuration, hardware limits, model naming, and file paths.
- `01_download_corpus.py`: Fetches Enron dataset and isolates robust email threads.
- `02_extract.py`: Iterates over threads, calls Ollama + regex extractors to build knowledge, tracks character offsets, and runs extraction check-pointing.
- `03_dedup.py`: Runs the 3-layer `Deduplicator` pipeline, creating recoverable snapshots of graph merges.
- `04_graph.py`: Converts the refined store into an exportable NetworkX graph structure.
- `05_retrieve.py`: Backend demonstrator for localized semantic graph retrieval.
- `06_app.py`: The Streamlit frontend application.
- `run_pipeline.py`: Main orchestration script.

## Advanced Features 

* **MemoryStore Container**: All entities and claims are wrapped in a first-class `MemoryStore` API for safe querying and modification constraints.
* **Undo Merge**: Included in `03_dedup.py`, the `undo_merge` function will take any logged `MergeRecord` and cleanly revert the knowledge graph state.
* **Re-running Idempotency**: If you configure the schema to add a new Entity type, re-running the extraction tool will idempotently append new insights on top of pre-computed checkpoints.
