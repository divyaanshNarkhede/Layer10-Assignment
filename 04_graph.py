"""
Step 4 – Memory Graph Design.

Builds a NetworkX MultiDiGraph:
  • Nodes  = canonical Entities (with attributes)
  • Edges  = Claims (with Evidence as edge data)

Also provides serialisation to/from JSON for persistence.
"""
import json, os, sys
import networkx as nx
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from config import OUTPUT_DIR, GRAPH_PATH
from schema import (
    Entity, EntityType, Claim, ClaimStatus, Evidence,
    MergeRecord, MemoryGraphSnapshot
)


def build_graph(
    entities: list[Entity],
    claims: list[Claim],
    merge_log: list[MergeRecord],
) -> nx.MultiDiGraph:
    """Build the memory graph from canonical entities and claims."""
    G = nx.MultiDiGraph()

    # Add entity nodes
    for ent in entities:
        G.add_node(ent.id, **{
            "name": ent.name,
            "type": ent.type.value,
            "aliases": ent.aliases,
            "first_seen": ent.first_seen,
            "last_seen": ent.last_seen,
            "metadata": ent.metadata,
        })

    # Add claim edges
    for claim in claims:
        # Ensure both nodes exist
        if claim.subject_id not in G:
            G.add_node(claim.subject_id, name=claim.subject_name, type="Unknown")
        if claim.object_id not in G:
            G.add_node(claim.object_id, name=claim.object_name, type="Unknown")

        G.add_edge(
            claim.subject_id,
            claim.object_id,
            key=claim.id,
            **{
                "claim_id": claim.id,
                "relation": claim.relation.value,
                "confidence": claim.confidence,
                "status": claim.status.value,
                "valid_from": claim.valid_from,
                "valid_until": claim.valid_until,
                "superseded_by": claim.superseded_by,
                "evidence": [e.model_dump() for e in claim.evidence],
                "evidence_count": len(claim.evidence),
                "subject_name": claim.subject_name,
                "object_name": claim.object_name,
                "extraction_version": claim.extraction_version,
            }
        )

    return G


def graph_stats(G: nx.MultiDiGraph) -> dict:
    """Compute summary statistics for the memory graph."""
    # Node type distribution
    type_dist = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "Unknown")
        type_dist[t] = type_dist.get(t, 0) + 1

    # Relation type distribution
    rel_dist = {}
    for _, _, data in G.edges(data=True):
        r = data.get("relation", "unknown")
        rel_dist[r] = rel_dist.get(r, 0) + 1

    # Status distribution
    status_dist = {}
    for _, _, data in G.edges(data=True):
        s = data.get("status", "unknown")
        status_dist[s] = status_dist.get(s, 0) + 1

    # Top entities by degree
    top_entities = sorted(
        [(n, G.degree(n), G.nodes[n].get("name", n)) for n in G.nodes()],
        key=lambda x: x[1], reverse=True
    )[:15]

    # Evidence coverage
    total_evidence = sum(
        data.get("evidence_count", 0) for _, _, data in G.edges(data=True)
    )
    claims_with_evidence = sum(
        1 for _, _, data in G.edges(data=True) if data.get("evidence_count", 0) > 0
    )
    total_claims = G.number_of_edges()

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "node_types": type_dist,
        "relation_types": rel_dist,
        "status_distribution": status_dist,
        "top_entities": [(name, deg) for _, deg, name in top_entities],
        "total_evidence_pointers": total_evidence,
        "claims_with_evidence": claims_with_evidence,
        "evidence_coverage": f"{claims_with_evidence/max(total_claims,1)*100:.1f}%",
        "connected_components": nx.number_weakly_connected_components(G),
    }


def serialize_graph(
    G: nx.MultiDiGraph,
    entities: list[Entity],
    claims: list[Claim],
    merge_log: list[MergeRecord],
) -> str:
    """Serialize the full graph snapshot to JSON."""
    snapshot = MemoryGraphSnapshot(
        entities=entities,
        claims=claims,
        merge_log=merge_log,
        corpus_info={
            "name": "Enron Email Dataset (synthetic sample)",
            "source": "Generated from historical Enron email patterns",
            "size": len(set(c.evidence[0].source_id for c in claims if c.evidence)),
            "date_range": "2001-01 to 2001-12",
        }
    )
    return snapshot.model_dump_json(indent=2)


def save_graph(
    G: nx.MultiDiGraph,
    entities: list[Entity],
    claims: list[Claim],
    merge_log: list[MergeRecord],
    path: str = GRAPH_PATH,
):
    """Save the memory graph to disk."""
    # Save Pydantic snapshot
    json_str = serialize_graph(G, entities, claims, merge_log)
    with open(path, "w") as f:
        f.write(json_str)

    # Also save NetworkX adjacency for pyvis
    nx_path = path.replace(".json", "_nx.json")
    nx_data = nx.node_link_data(G)
    with open(nx_path, "w") as f:
        json.dump(nx_data, f, indent=2, default=str)

    print(f"  Graph saved to: {path}")
    print(f"  NetworkX data:  {nx_path}")


def load_graph(path: str = GRAPH_PATH) -> tuple:
    """Load graph from JSON snapshot."""
    with open(path) as f:
        data = json.load(f)

    snapshot = MemoryGraphSnapshot(**data)
    G = build_graph(snapshot.entities, snapshot.claims, snapshot.merge_log)
    return G, snapshot.entities, snapshot.claims, snapshot.merge_log


def run_graph_build(dedup_path: str = None):
    """Build the memory graph from deduped data."""
    print("=" * 60)
    print("Layer10 Memory Pipeline - Memory Graph Construction")
    print("=" * 60)

    if dedup_path is None:
        dedup_path = os.path.join(OUTPUT_DIR, "deduped.json")

    with open(dedup_path) as f:
        data = json.load(f)

    entities = [Entity(**e) for e in data["entities"]]
    claims = [Claim(**c) for c in data["claims"]]
    merge_log = [MergeRecord(**m) for m in data.get("merge_log", [])]

    print(f"\nBuilding graph from {len(entities)} entities, {len(claims)} claims...")

    G = build_graph(entities, claims, merge_log)
    stats = graph_stats(G)

    print(f"\nGraph Statistics:")
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Connected components: {stats['connected_components']}")
    print(f"  Evidence coverage: {stats['evidence_coverage']}")
    print(f"\n  Node types:")
    for t, count in sorted(stats['node_types'].items(), key=lambda x: -x[1]):
        print(f"    {t}: {count}")
    print(f"\n  Relation types:")
    for r, count in sorted(stats['relation_types'].items(), key=lambda x: -x[1]):
        print(f"    {r}: {count}")
    print(f"\n  Top entities by connections:")
    for name, deg in stats['top_entities'][:10]:
        print(f"    {name}: {deg} connections")

    save_graph(G, entities, claims, merge_log)

    # Save stats
    stats_path = os.path.join(OUTPUT_DIR, "graph_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return G, entities, claims, merge_log


if __name__ == "__main__":
    run_graph_build()
