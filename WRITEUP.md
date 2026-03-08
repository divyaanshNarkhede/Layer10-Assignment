# Layer10 Take-Home: Grounded Long-Term Memory via Structured Extraction, Deduplication, and a Context Graph

## 1. Corpus Selection

**Dataset:** Enron Email Dataset (synthetic replica of historical Enron emails)

**Source:** The pipeline supports three data sources:
1. **Kaggle CSV** – download from [kaggle.com/datasets/wcukierski/enron-email-dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset), place `emails.csv` in `data/`
2. **CMU maildir** – extract the CMU Enron tarball into `data/maildir/`
3. **Built-in synthetic sample** – 75 historically-accurate emails covering key Enron events (used by default when no local data is present)

**Reproduction:**
```bash
python 01_download_corpus.py
```

The synthetic corpus mirrors real Enron patterns: email threads, forwarding/quoting, identity resolution challenges, decisions that get reversed, conflicting sources, and changing organisational state over the Jan 2001 – Dec 2001 timeline.

---

## 2. Ontology / Schema Design

### Entity Types
| Type | Examples | Rationale |
|------|----------|-----------|
| `Person` | Kenneth Lay, Jeff Skilling | People sending/receiving email |
| `Organisation` | Enron, Arthur Andersen, Dynegy | Companies, regulators, firms |
| `System` | EnronOnline, RiskTrak | Software platforms |
| `Project` | Death Star, Fat Boy | Named initiatives/strategies |
| `FinancialInstrument` | Raptor, LJM, Chewco | SPEs, funds, partnerships |
| `Event` | Bankruptcy filing, FERC investigation | Discrete happenings |
| `Role` | CEO, CFO, Treasurer | Job titles |
| `Location` | Houston, Portland, California | Places |

### Relation Types
Organised into four semantic groups:
- **Organisational:** `works_at`, `has_role`, `reports_to`, `manages`, `member_of`
- **Actions:** `created`, `sent_email`, `decided`, `proposed`, `approved`, `rejected`, `warned_about`, `investigated`
- **States/Facts:** `has_status`, `has_value`, `depends_on`, `conflicts_with`, `supersedes`, `related_to`
- **Financial:** `invested_in`, `profited_from`, `owes`, `acquired`

### Why these choices?
The Enron corpus revolves around people making decisions within an organisation, financial vehicles that change state, and warnings that get ignored. The schema captures this with first-class support for temporal state changes (via `ClaimStatus`: current / historical / retracted / disputed) and explicit evidence grounding on every claim.

### Extensibility
Adding a new entity type or relation type requires only extending the corresponding `Enum` in `schema.py`. All downstream code (extraction, dedup, graph, retrieval) is parametric over the enums—no hard-coded type checks.

---

## 3. Structured Extraction

### Extraction Contract
Every extracted claim **must** carry:
- `source_id` – which email
- `excerpt` – verbatim quote (≤ 300 chars)
- `timestamp` – when the source was authored
- `author` – who wrote it
- `confidence` – 1.0 (explicit), 0.7 (implied), 0.5 (loosely implied)

### Dual Extraction Strategy
1. **LLM extraction** (Ollama + llama3.2) – a system prompt forces JSON output matching the Pydantic schema. A validation loop retries up to 3 times on parse failure.
2. **Rule-based extraction** – deterministic regex/heuristic fallback that always runs, catching emails, dollar amounts, role titles, known project names, decision verbs, and locations.

Results from both passes are merged: LLM entities/claims take priority; rule-based results fill gaps.

### Validation & Repair
- **`parse_llm_json()`** – attempts `json.loads()` first, then regex-extracts JSON from markdown code fences or raw text.
- **Pydantic validators** – `field_validator` on `Claim.confidence` clamps values to [0, 1]; unknown entity/relation types are normalised deterministically.
- **Retry loop** – up to `MAX_RETRIES` (3) calls to the LLM if JSON is invalid.
- **Quality gate** – claims below `CONFIDENCE_THRESHOLD` (0.5) are discarded during dedup.

### Versioning
Each `ExtractionResult` stores:
- `extraction_version` – a string tag (`"v1"`)
- `model` – which LLM was used (or `"rule_based"`)
- `prompt_hash` – MD5 of the system prompt, so changes are detectable

**Back-fill strategy:** If the ontology changes, re-run `02_extract.py` on the same corpus. The rule-based extractor is idempotent; LLM results will differ but the merge-on-canonical-key logic in dedup means old and new extractions can coexist.

---

## 4. Deduplication & Canonicalization

Three layers, all logged in reversible `MergeRecord` objects:

### Layer 1 – Artifact Dedup
Near-identical emails (forwarded/quoted content) are detected by content-hashing the first 150 chars of the earliest evidence excerpt. Duplicate extractions are merged into the original, preserving all evidence pointers.

### Layer 2 – Entity Canonicalization (Union-Find)
1. **Exact match** – entities with the same `(lowered_name, type)` are unioned immediately.
2. **Semantic match** – within each entity type, `all-MiniLM-L6-v2` embeddings are compared pairwise. Pairs with cosine similarity ≥ 0.85 are merged. The winner inherits all aliases and the earliest/latest seen timestamps.

The Union-Find data structure with path compression ensures O(α(n)) amortised merges and makes the canonical representative deterministic.

### Layer 3 – Claim Dedup + Conflict Resolution
1. **Canonical key** – `(lowered_subject_name | relation | lowered_object_name)`.
2. **Merge** – duplicate claims are collapsed into the most recent one; evidence lists are concatenated (de-fingerprinted to avoid exact-duplicate evidence).
3. **Conflict detection** – for `has_status` and `has_role` claims on the same subject, if the object differs across time, the older claim is marked `HISTORICAL` with `valid_until` set and `superseded_by` pointing to the newer claim.

### Reversibility
Every merge operation creates a `MergeRecord` with:
- `winner_id`, `loser_id`
- `reason` (e.g. `"exact_match"`, `"semantic_similarity=0.912"`, `"temporal_supersede"`)
- `reversible = True`
- `timestamp`

The merge log is persisted in the graph snapshot, so any merge can be audited or undone.

---

## 5. Memory Graph Design

### Storage
- **Engine:** NetworkX `MultiDiGraph` (allows multiple directed edges between the same pair of nodes)
- **Nodes:** Canonical entities with attributes (`name`, `type`, `aliases`, `first_seen`, `last_seen`)
- **Edges:** Claims, keyed by `claim.id`, carrying `relation`, `confidence`, `status`, `valid_from`, `valid_until`, `superseded_by`, and the full `evidence` list

### Time
- **Event time** = email `timestamp` (when the source was authored)
- **Validity time** = `valid_from` / `valid_until` on claims (when the fact was true)
- **"Current"** = `ClaimStatus.CURRENT` (no `valid_until` set)

### Updates & Idempotency
Re-running the pipeline on the same corpus is safe:
1. `01_download_corpus.py` is deterministic (fixed random seed, sorted output)
2. `02_extract.py` overwrites `extractions.json`
3. `03_dedup.py` rebuilds from scratch—entity IDs are UUID-based so the final graph is a fresh snapshot

In production, incremental ingestion would:
- Hash each source artifact on ingest; skip if hash matches an existing record
- Run extraction only on new/changed artifacts
- Run dedup only on the delta (new entities/claims vs existing graph)
- Log a `reprocessing` event in the merge log

### Edits / Deletes / Redactions
The `ClaimStatus.RETRACTED` status handles source deletion. In production:
1. When a source is deleted, mark all claims grounded solely in that source as `RETRACTED`
2. If a claim has multiple evidence sources and only one is deleted, remove that evidence pointer but keep the claim alive
3. Maintain a tombstone record so the deletion is auditable

### Permissions (Conceptual)
At retrieval time, filter claims by the ACL on their underlying sources:
```
visible_claims = [c for c in claims
                  if any(user.can_access(ev.source_id) for ev in c.evidence)]
```
This ensures memory is never retrieved from a source the user cannot read.

### Observability
- `graph_stats.json` tracks node/edge counts, type distributions, evidence coverage, and connected components
- Extraction logs show per-email entity/claim counts and LLM failure rates
- Merge log tracks every dedup operation for quality monitoring

---

## 6. Retrieval & Grounding

### Pipeline
1. **Embed** the user's question with `all-MiniLM-L6-v2`
2. **Score** all entities by cosine similarity (names + aliases)
3. **Expand** the top-k entity matches by 1-hop graph traversal (successors + predecessors)
4. **Rank** retrieved claims by: `relevance * 0.6 + confidence * 0.2 + status_boost * 0.2`
   - Current claims get a 1.0 status boost; historical claims get 0.7
5. **Pack** the top claims + evidence excerpts into a structured context pack
6. **Generate** a grounded answer (via Ollama or a template-based fallback)

### Grounding guarantee
Every item in the context pack includes `source_id`, `excerpt`, `timestamp`, and `author`. The generation prompt explicitly instructs: _"Cite the source_id for every fact you state."_

### Handling conflicts
When the context pack contains both a current and historical claim about the same subject, both are presented with their validity periods. The prompt says: _"If there are conflicting claims, present both with timestamps."_

### Example questions (pre-computed)
1. Who warned about the Raptor structures and what happened?
2. What was the role of Andrew Fastow at Enron?
3. How did EnronOnline perform and what happened to it?
4. What were the California energy trading strategies?
5. Why did the Dynegy merger fail?

---

## 7. Visualization Layer

**Tech:** Streamlit + PyVis

### Features
| Tab | Description |
|-----|-------------|
| 📊 Graph View | Interactive force-directed graph (PyVis). Nodes coloured by entity type, edges coloured by claim status (green = current, red = historical). Hover for details. Sidebar filters for entity type, confidence threshold, and claim status. |
| 📄 Evidence Panel | Searchable list of claims. Expand any claim to see its supporting evidence (exact quotes, source IDs, authors, timestamps). |
| 🔗 Merges & Duplicates | Full merge log with entity-alias inspector. Shows every dedup operation and its reason. |
| 💬 Query Interface | Pre-computed context packs + live Q&A. Type a question, get a grounded answer with citations. |
| 📈 Statistics | Entity/relation/status distributions, top-connected entities, evidence coverage. |

### Launch
```bash
streamlit run 06_app.py
```

---

## 8. Adapting to Layer10's Target Environment

### Ontology changes for email, Slack, Jira/Linear

| Current (Enron) | Layer10 (Enterprise) |
|-----------------|---------------------|
| `Person` (from email) | `Person` resolved across email, Slack handles, Jira assignees |
| `Organisation` | `Team`, `Department`, `Customer` |
| `FinancialInstrument` | `Ticket`, `Epic`, `Sprint` |
| `Project` | `Project`, `Component`, `Repository` |
| `Event` | `Decision`, `StatusChange`, `Deployment`, `Incident` |
| `Role` | `Role` (same, but resolved from Jira permissions and Slack workspace roles) |

### Extraction contract changes
- **Email:** Same as current. Add header parsing for thread-ID based dedup.
- **Slack:** Messages are shorter; thread replies carry parent context. Extract from (message + thread) units rather than single messages.
- **Jira/Linear:** Structured fields (status, assignee, priority) are extracted directly without LLM. Only comments/descriptions need LLM extraction. Ticket state transitions become first-class `ClaimStatus` changes.

### Dedup strategy changes
- **Identity resolution** becomes harder: `jane.doe@company.com`, `@janedoe` on Slack, and `Jane Doe` in Jira are the same person. Use a multi-signal resolver (email domain + display name + embedding similarity).
- **Cross-source claim dedup:** A decision discussed in Slack, confirmed in email, and tracked in Jira should produce one canonical claim with three evidence pointers.
- **Structured + unstructured fusion:** Link chat/email discussions to tickets via explicit references (Jira key mentions in Slack/email) and temporal proximity.

### Grounding & safety
- **Provenance:** Every claim must trace back to a specific message ID, ticket ID, or document version.
- **Deletions/redactions:** When a Slack message is deleted or an email is redacted, all claims grounded solely in that source are marked `RETRACTED`. Claims with multiple evidence sources survive.
- **Permissions:** Memory retrieval is constrained by the user's access to underlying sources. A claim grounded in a private Slack channel is only visible to channel members.

### Long-term memory behaviour
- **Durable memory** = claims with ≥ 2 evidence sources and confidence ≥ 0.7. These are promoted to the permanent graph.
- **Ephemeral context** = single-source, low-confidence claims. These are available for 30 days, then archived (not deleted—moved to cold storage).
- **Drift prevention:** Weekly re-extraction on a sample of recent sources, compared against existing claims. Divergence above a threshold triggers a full re-extraction run.

### Operational reality
- **Scaling:** Move from in-memory NetworkX to PostgreSQL with adjacency list tables (or Neo4j for heavier traversal). Entity embeddings in pgvector for semantic search.
- **Cost:** Rule-based extraction handles structured sources (Jira) at zero LLM cost. LLM extraction reserved for unstructured text (email, Slack, doc comments).
- **Incremental updates:** Webhook-driven ingestion from Slack/Jira/email. Only new/changed artifacts are processed. Graph is updated incrementally, not rebuilt.
- **Evaluation/regression testing:** Maintain a golden set of ~100 manually verified claims. After any pipeline change, check that the golden set is still correctly extracted and deduplicated.

---

## 9. Reproducibility

```bash
# 1. Clone and setup
cd layer10_new
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) Install Ollama for LLM extraction
# curl -fsSL https://ollama.com/install.sh | sh
# ollama pull llama3.2

# 3. Run the full pipeline
python run_pipeline.py

# 4. Launch visualization
streamlit run 06_app.py
```

The pipeline runs fully without Ollama (using rule-based extraction only). Ollama enhances extraction quality but is not required.

---

## 10. Trade-offs & Decisions

| Decision | Rationale | Alternative |
|----------|-----------|-------------|
| NetworkX in-memory graph | Simple, no external dependencies, fast for <10K nodes | Neo4j for production scale |
| Pydantic for schema | Type safety, validation, serialisation for free | Dataclasses (less validation) |
| Union-Find for entity merge | O(α(n)) merges, deterministic canonical representatives | Clustering (slower, non-deterministic) |
| Dual extraction (LLM + rules) | Rules provide baseline; LLM adds semantic depth | LLM-only (fragile when LLM unavailable) |
| Synthetic corpus as default | Reproducible without external downloads; covers all key patterns | Real Enron data only (harder to reproduce) |
| Cosine similarity threshold 0.85 | Balances precision (avoid false merges) vs recall | Lower threshold = more merges, more collisions |
| Confidence threshold 0.5 | Keeps loosely implied facts; filters noise | Higher = cleaner graph, fewer claims |

---

## 11. Advanced Features

### 11.1 Character-Offset Evidence Pointers
Every `Evidence` object now carries optional `char_offset_start` and `char_offset_end` fields. In rule-based extraction, these are computed from regex match positions; in LLM extraction, the quote is searched against the original email body. This enables precise source-text highlighting in the UI and allows downstream consumers to locate the exact span in the source document.

### 11.2 MemoryStore – First-Class Container
The `MemoryStore` class (in `schema.py`) wraps the flat `MemoryGraphSnapshot` with dict-indexed entities and claims, plus helper methods:
- `add_entity()`, `get_entity()`, `remove_entity()`, `find_entities_by_name()`
- `add_claim()`, `get_claim()`, `remove_claim()`, `get_claims_for_entity()`, `get_claims_between()`
- `to_snapshot()` / `from_snapshot()` for interop with the existing JSON format
- `serialize()` / `deserialize()` for disk persistence
- `remove_orphan_entities()` for cleanup

### 11.3 Reversible Merges (undo_merge)
Every `MergeRecord` now stores `original_snapshots` — the full pre-merge state of both winner and loser. The `undo_merge(store, merge_id)` function in `03_dedup.py` restores the original entities/claims and removes the merge record from the log. This is critical for iterating on dedup thresholds without re-running the full pipeline.

### 11.4 Extraction Checkpointing
The extraction step (`02_extract.py`) now saves a checkpoint file (`output/_extraction_checkpoint.json`) after each thread. If the process crashes or is interrupted (e.g., GPU OOM), re-running `run_extraction()` automatically resumes from the last completed thread. The checkpoint is deleted on successful completion.

### 11.5 OOP Deduplicator
The `Deduplicator` class in `03_dedup.py` wraps all dedup logic into a cohesive object:
- `run_full_pipeline()` — executes all four layers: artifact dedup → entity canonicalisation → claim dedup → orphan cleanup
- `to_memory_store()` — converts results into a `MemoryStore` container
- `save()` — persists results to disk
The original `run_dedup()` function still works as a backward-compatible wrapper.

### 11.6 LLM-Powered Answer Synthesis in Chat UI
The Query tab in the Streamlit app (`06_app.py`) now calls Ollama to generate a synthesized, grounded answer from the search results. The LLM receives structured context (entities + claims + evidence excerpts) and produces a natural-language answer that cites specific evidence. A checkbox lets users toggle between LLM synthesis and the template-based fallback.
