"""
Given a question, this script finds the most relevant parts of the graph
and packages them up so the LLM can give a grounded answer.

How it works:
  1. Embed the question using the same model we used for dedup
  2. Score every entity by how closely it matches the question
  3. Walk one hop out from the top matches to catch related context
  4. Rank the resulting claims by relevance + confidence + freshness
  5. Pull out the supporting quotes and format everything as citations

If Ollama is running, it will also generate a final answer using
the context pack as its source of truth.
"""
import json, os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    GRAPH_PATH, OUTPUT_DIR, CONTEXT_PACKS_PATH,
    OLLAMA_MODEL, EMBEDDING_MODEL
)
from schema import Claim, ClaimStatus, Evidence, Entity

try:
    import ollama as ollama_client
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

import networkx as nx
import numpy as np

# ── Only initialize the embedding model when we first need it ───────────────
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ── Load the graph from disk into memory ───────────────────────────────────
def load_graph_simple(path: str = GRAPH_PATH):
    """Reads the graph JSON from disk and rebuilds entities, claims, and the NetworkX graph."""
    with open(path) as f:
        data = json.load(f)

    entities = [Entity(**e) for e in data.get("entities", [])]
    claims = [Claim(**c) for c in data.get("claims", [])]

    # Build a simple graph
    G = nx.MultiDiGraph()
    for ent in entities:
        G.add_node(ent.id, name=ent.name, type=ent.type.value,
                   aliases=ent.aliases)
    for claim in claims:
        if claim.subject_id not in G:
            G.add_node(claim.subject_id, name=claim.subject_name, type="Unknown")
        if claim.object_id not in G:
            G.add_node(claim.object_id, name=claim.object_name, type="Unknown")
        G.add_edge(claim.subject_id, claim.object_id, key=claim.id, **{
            "claim_id": claim.id,
            "relation": claim.relation.value,
            "confidence": claim.confidence,
            "status": claim.status.value,
            "valid_from": claim.valid_from,
            "valid_until": claim.valid_until,
            "evidence": [e.model_dump() for e in claim.evidence],
            "evidence_count": len(claim.evidence),
            "subject_name": claim.subject_name,
            "object_name": claim.object_name,
        })

    return G, entities, claims


# ── The main retrieval function — turns a question into a context pack ────────
def retrieve_context(
    question: str,
    G: nx.MultiDiGraph,
    entities: list[Entity],
    claims: list[Claim],
    top_k: int = 10,
    expand_hops: int = 1,
) -> dict:
    """
    Retrieve a grounded context pack for a question.

    Returns:
    {
        "question": str,
        "matched_entities": [...],
        "relevant_claims": [...],
        "evidence_snippets": [...],
        "context_summary": str,
    }
    """
    model = get_embed_model()

    # Step 1: Turn the question into a vector
    q_emb = model.encode([question])[0]

    # Step 2: Score every entity by how closely its name/aliases match the question
    entity_names = [e.name for e in entities]
    entity_alias_texts = [
        f"{e.name} {' '.join(e.aliases)}" for e in entities
    ]
    entity_embs = model.encode(entity_alias_texts)

    entity_scores = []
    for i, ent in enumerate(entities):
        sim = cosine_sim(q_emb, entity_embs[i])
        entity_scores.append((ent, sim))

    entity_scores.sort(key=lambda x: -x[1])
    top_entities = entity_scores[:top_k]

    # Step 3: Collect every claim touching our top entities, then walk one hop further
    matched_entity_ids = {ent.id for ent, _ in top_entities}
    relevant_claim_ids = set()
    relevant_claims = []

    for ent_id in list(matched_entity_ids):
        # Direct edges
        for u, v, key, data in G.edges(ent_id, data=True, keys=True):
            if key not in relevant_claim_ids:
                relevant_claim_ids.add(key)
                relevant_claims.append(data)
        for u, v, key, data in G.in_edges(ent_id, data=True, keys=True):
            if key not in relevant_claim_ids:
                relevant_claim_ids.add(key)
                relevant_claims.append(data)

        # 1-hop expansion
        if expand_hops >= 1:
            for neighbor in list(G.successors(ent_id)) + list(G.predecessors(ent_id)):
                matched_entity_ids.add(neighbor)
                for u, v, key, data in G.edges(neighbor, data=True, keys=True):
                    if key not in relevant_claim_ids:
                        relevant_claim_ids.add(key)
                        relevant_claims.append(data)

    # Step 4: Score each candidate claim against the question
    claim_texts = []
    for c in relevant_claims:
        text = f"{c.get('subject_name', '')} {c.get('relation', '')} {c.get('object_name', '')}"
        claim_texts.append(text)

    if claim_texts:
        claim_embs = model.encode(claim_texts)
        scored_claims = []
        for i, c in enumerate(relevant_claims):
            sim = cosine_sim(q_emb, claim_embs[i])
            confidence = c.get("confidence", 0.5)
            # Boost current claims, penalise historical
            status_boost = 1.0 if c.get("status") == "current" else 0.7
            score = sim * 0.6 + confidence * 0.2 + status_boost * 0.2
            scored_claims.append((c, score))

        scored_claims.sort(key=lambda x: -x[1])
        ranked_claims = scored_claims[:top_k * 2]
    else:
        ranked_claims = []

    # Step 5: Pull out the supporting quotes, deduplicated by fingerprint
    evidence_snippets = []
    seen_evidence = set()
    for claim_data, score in ranked_claims:
        for ev in claim_data.get("evidence", []):
            fp = f"{ev.get('source_id', '')}|{ev.get('excerpt', '')[:50]}"
            if fp not in seen_evidence:
                evidence_snippets.append({
                    "source_id": ev.get("source_id", ""),
                    "excerpt": ev.get("excerpt", ""),
                    "timestamp": ev.get("timestamp", ""),
                    "author": ev.get("author", ""),
                    "subject": ev.get("subject", ""),
                    "supporting_claim": f"{claim_data.get('subject_name', '')} "
                                       f"{claim_data.get('relation', '')} "
                                       f"{claim_data.get('object_name', '')}",
                    "relevance_score": round(score, 3),
                })
                seen_evidence.add(fp)

    # Step 6: Build a human-readable summary of the top claims for the context pack
    context_lines = []
    for i, (claim_data, score) in enumerate(ranked_claims[:10]):
        status_marker = "✓" if claim_data.get("status") == "current" else "⟲"
        line = (
            f"[{status_marker}] {claim_data.get('subject_name', '?')} "
            f"—{claim_data.get('relation', '?')}→ "
            f"{claim_data.get('object_name', '?')} "
            f"(confidence: {claim_data.get('confidence', 0):.2f}, "
            f"evidence: {claim_data.get('evidence_count', 0)} sources)"
        )
        context_lines.append(line)

    return {
        "question": question,
        "matched_entities": [
            {"name": ent.name, "type": ent.type.value, "score": round(sim, 3)}
            for ent, sim in top_entities
        ],
        "relevant_claims": [
            {
                "subject": c.get("subject_name", ""),
                "relation": c.get("relation", ""),
                "object": c.get("object_name", ""),
                "confidence": c.get("confidence", 0),
                "status": c.get("status", ""),
                "evidence_count": c.get("evidence_count", 0),
                "valid_from": c.get("valid_from", ""),
                "valid_until": c.get("valid_until", ""),
                "relevance_score": round(score, 3),
            }
            for c, score in ranked_claims[:10]
        ],
        "evidence_snippets": evidence_snippets[:15],
        "context_summary": "\n".join(context_lines),
    }


def generate_grounded_answer(question: str, context_pack: dict) -> str:
    """Generate a grounded answer using Ollama (optional)."""
    if not HAS_OLLAMA:
        return _format_answer_from_context(question, context_pack)

    context_text = context_pack.get("context_summary", "")
    evidence_text = "\n".join(
        f"  [{ev['source_id']}] ({ev.get('author', '?')}, {ev.get('timestamp', '?')}): "
        f"\"{ev['excerpt'][:200]}\""
        for ev in context_pack.get("evidence_snippets", [])[:8]
    )

    prompt = f"""Answer the following question using ONLY the provided context and evidence.
Cite the source_id for every fact you state. If the context doesn't contain enough
information, say so explicitly.

Question: {question}

Context (claims from memory graph):
{context_text}

Evidence (exact quotes from source emails):
{evidence_text}

Instructions:
1. Only state facts supported by the evidence above.
2. Cite sources like [source_id] after each fact.
3. If there are conflicting claims, present both with timestamps.
4. Distinguish between current and historical facts.
"""
    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 1024},
        )
        return response["message"]["content"]
    except Exception as e:
        return _format_answer_from_context(question, context_pack)


def _format_answer_from_context(question: str, context_pack: dict) -> str:
    """Format a grounded answer without LLM."""
    lines = [f"**Question:** {question}\n"]
    lines.append("**Relevant Facts (from memory graph):**\n")

    for i, claim in enumerate(context_pack.get("relevant_claims", [])[:8]):
        status = "✓ Current" if claim["status"] == "current" else "⟲ Historical"
        lines.append(
            f"{i+1}. [{status}] **{claim['subject']}** —{claim['relation']}→ "
            f"**{claim['object']}** "
            f"(confidence: {claim['confidence']:.0%}, "
            f"{claim['evidence_count']} evidence sources)"
        )
        if claim.get("valid_from"):
            lines.append(f"   Valid from: {claim['valid_from']}")
        if claim.get("valid_until"):
            lines.append(f"   Valid until: {claim['valid_until']}")

    lines.append("\n**Supporting Evidence:**\n")
    for ev in context_pack.get("evidence_snippets", [])[:6]:
        lines.append(
            f"- **[{ev['source_id']}]** ({ev.get('author', '?')}, "
            f"{ev.get('timestamp', '?')}):\n"
            f"  > \"{ev['excerpt'][:250]}\"\n"
            f"  _Supports: {ev.get('supporting_claim', '')}_"
        )

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Demo questions
# ═════════════════════════════════════════════════════════════════════════════
DEMO_QUESTIONS = [
    "Who warned about the Raptor structures and what happened?",
    "What was the role of Andrew Fastow at Enron?",
    "How did EnronOnline perform and what happened to it?",
    "What were the California energy trading strategies?",
    "Why did the Dynegy merger fail?",
]


def run_retrieval(graph_path: str = GRAPH_PATH):
    """Run retrieval demo with sample questions."""
    print("=" * 60)
    print("Layer10 Memory Pipeline - Retrieval & Grounding")
    print("=" * 60)

    G, entities, claims = load_graph_simple(graph_path)
    print(f"\nLoaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    all_packs = []

    for q in DEMO_QUESTIONS:
        print(f"\n{'─'*50}")
        print(f"Q: {q}")
        print(f"{'─'*50}")

        pack = retrieve_context(q, G, entities, claims)
        answer = generate_grounded_answer(q, pack)

        print(answer[:1500])
        pack["generated_answer"] = answer
        all_packs.append(pack)

    # Save context packs
    with open(CONTEXT_PACKS_PATH, "w") as f:
        json.dump(all_packs, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print(f"Context packs saved to: {CONTEXT_PACKS_PATH}")

    return all_packs


if __name__ == "__main__":
    run_retrieval()
