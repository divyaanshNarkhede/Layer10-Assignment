"""
Step 3 – Deduplication and Canonicalization.

Three layers of dedup:
  1. Artifact dedup   – near-identical emails (quoting/forwarding)
  2. Entity dedup     – exact match + semantic similarity (Union-Find)
  3. Claim dedup      – merge claims with same (subject, relation, object)

All merges are logged in MergeRecord with pre-merge snapshots for undo.
OOP Deduplicator class wraps all logic.
"""
import json, os, sys, copy
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    EXTRACTION_PATH, OUTPUT_DIR, SIMILARITY_THRESHOLD,
    CONFIDENCE_THRESHOLD, EMBEDDING_MODEL
)
from schema import (
    Entity, EntityType, Claim, ClaimStatus, Evidence,
    ExtractionResult, MergeRecord, MemoryStore
)

# ─── Lazy-load sentence-transformers ─────────────────────────────────────────
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def cosine_sim(a, b) -> float:
    """Cosine similarity between two vectors."""
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ═════════════════════════════════════════════════════════════════════════════
# Union-Find for entity merging
# ═════════════════════════════════════════════════════════════════════════════
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra  # a becomes the canonical representative
            return True
        return False


# ═════════════════════════════════════════════════════════════════════════════
# Layer 1: Artifact Dedup (email-level)
# ═════════════════════════════════════════════════════════════════════════════
def dedup_artifacts(extractions: list[ExtractionResult]) -> list[ExtractionResult]:
    """Remove extractions from near-duplicate emails (forwarded/quoted)."""
    seen_bodies = {}
    deduped = []
    duplicates = 0

    for ext in extractions:
        # Simple content hash – first 200 chars of first evidence excerpt
        # In production, use MinHash / SimHash for fuzzy matching
        key = ext.source_id
        body_key = ""
        for claim in ext.claims:
            for ev in claim.evidence:
                body_key += ev.excerpt[:100]
                break
            if body_key:
                break

        if not body_key:
            deduped.append(ext)
            continue

        # Normalize: lowercase, strip whitespace
        norm_key = body_key.lower().strip()[:150]
        if norm_key in seen_bodies:
            duplicates += 1
            # Still keep the evidence – merge into the existing extraction
            existing = seen_bodies[norm_key]
            for claim in ext.claims:
                existing.claims.append(claim)
            for ent in ext.entities:
                existing.entities.append(ent)
        else:
            seen_bodies[norm_key] = ext
            deduped.append(ext)

    print(f"  Artifact dedup: {duplicates} near-duplicate emails merged")
    return deduped


# ═════════════════════════════════════════════════════════════════════════════
# Layer 2: Entity Canonicalization
# ═════════════════════════════════════════════════════════════════════════════
def canonicalize_entities(
    all_entities: list[Entity],
) -> tuple[list[Entity], dict[str, str], list[MergeRecord]]:
    """
    Merge entities via:
      1. Exact match (lowered name + type)
      2. Semantic similarity (embedding cosine > threshold)

    Returns: (canonical_entities, id_remap, merge_log)
    """
    merge_log = []
    uf = UnionFind()

    # Group by canonical_key for exact match
    key_to_entities: dict[str, list[Entity]] = defaultdict(list)
    for ent in all_entities:
        key_to_entities[ent.canonical_key()].append(ent)

    # Phase 1: Exact match merges
    canonical_map: dict[str, Entity] = {}
    for key, group in key_to_entities.items():
        winner = group[0]
        for other in group[1:]:
            # Snapshot before merge for undo support
            winner_snap = winner.model_dump()
            loser_snap = other.model_dump()

            uf.union(winner.id, other.id)
            # Merge aliases
            for alias in other.aliases:
                if alias not in winner.aliases:
                    winner.aliases.append(alias)
            if other.name not in winner.aliases and other.name != winner.name:
                winner.aliases.append(other.name)
            # Update time range
            if other.first_seen and (not winner.first_seen or other.first_seen < winner.first_seen):
                winner.first_seen = other.first_seen
            if other.last_seen and (not winner.last_seen or other.last_seen > winner.last_seen):
                winner.last_seen = other.last_seen

            merge_log.append(MergeRecord(
                merge_type="entity",
                winner_id=winner.id,
                loser_id=other.id,
                reason="exact_match",
                original_snapshots={
                    "winner": winner_snap,
                    "loser": loser_snap,
                },
            ))
        canonical_map[key] = winner

    # Phase 2: Semantic similarity (within same EntityType)
    canonical_list = list(canonical_map.values())
    type_groups: dict[EntityType, list[Entity]] = defaultdict(list)
    for ent in canonical_list:
        type_groups[ent.type].append(ent)

    model = get_embedding_model()
    for etype, group in type_groups.items():
        if len(group) < 2:
            continue
        names = [e.name for e in group]
        embeddings = model.encode(names)

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                sim = cosine_sim(embeddings[i], embeddings[j])
                if sim >= SIMILARITY_THRESHOLD:
                    winner, loser = group[i], group[j]
                    if uf.union(winner.id, loser.id):
                        # Snapshot before merge
                        winner_snap = winner.model_dump()
                        loser_snap = loser.model_dump()

                        if loser.name not in winner.aliases:
                            winner.aliases.append(loser.name)
                        merge_log.append(MergeRecord(
                            merge_type="entity",
                            winner_id=winner.id,
                            loser_id=loser.id,
                            reason=f"semantic_similarity={sim:.3f}",
                            original_snapshots={
                                "winner": winner_snap,
                                "loser": loser_snap,
                            },
                        ))

    # Build final canonical entity list and ID remap
    id_to_entity = {e.id: e for e in canonical_list}
    # Also index all entities so we can resolve any Union-Find root
    all_entity_by_id = {e.id: e for e in all_entities}
    final_entities = {}
    id_remap = {}

    for ent in all_entities:
        root = uf.find(ent.id)
        id_remap[ent.id] = root
        if root not in final_entities:
            # Prefer the canonical-list version; fall back to any entity
            final_entities[root] = id_to_entity.get(root,
                                                     all_entity_by_id.get(root, ent))

    print(f"  Entity canon: {len(all_entities)} → {len(final_entities)} "
          f"({len(all_entities) - len(final_entities)} merged)")

    return list(final_entities.values()), id_remap, merge_log


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3: Claim Dedup + Conflict Resolution
# ═════════════════════════════════════════════════════════════════════════════
def dedup_claims(
    all_claims: list[Claim],
    id_remap: dict[str, str],
) -> tuple[list[Claim], list[MergeRecord]]:
    """
    1. Remap subject/object IDs to canonical IDs
    2. Merge claims with same (subject, relation, object)
    3. Handle temporal conflicts (supersede old claims)
    """
    merge_log = []

    # Remap IDs
    for claim in all_claims:
        claim.subject_id = id_remap.get(claim.subject_id, claim.subject_id)
        claim.object_id = id_remap.get(claim.object_id, claim.object_id)

    # Group by canonical key
    key_to_claims: dict[str, list[Claim]] = defaultdict(list)
    for claim in all_claims:
        key = claim.canonical_key()
        key_to_claims[key].append(claim)

    final_claims = []
    for key, group in key_to_claims.items():
        if len(group) == 1:
            final_claims.append(group[0])
            continue

        # Sort by timestamp (earliest first)
        group.sort(key=lambda c: c.valid_from or "")

        # Winner = most recent, with all evidence merged
        winner = group[-1]  # latest timestamp
        for older in group[:-1]:
            # Snapshot before merge
            winner_snap = winner.model_dump()
            older_snap = older.model_dump()

            # Merge evidence
            existing_fps = {e.fingerprint() for e in winner.evidence}
            for ev in older.evidence:
                if ev.fingerprint() not in existing_fps:
                    winner.evidence.append(ev)
                    existing_fps.add(ev.fingerprint())
            # Average confidence
            winner.confidence = max(winner.confidence, older.confidence)

            merge_log.append(MergeRecord(
                merge_type="claim",
                winner_id=winner.id,
                loser_id=older.id,
                reason=f"same_canonical_key ({len(group)} occurrences)",
                original_snapshots={
                    "winner": winner_snap,
                    "loser": older_snap,
                },
            ))

        final_claims.append(winner)

    # Conflict detection: look for contradicting status claims
    status_claims = [c for c in final_claims if c.relation.value in ("has_status", "has_role")]
    subj_status: dict[str, list[Claim]] = defaultdict(list)
    for c in status_claims:
        subj_status[c.subject_id].append(c)

    for subj_id, claims_group in subj_status.items():
        if len(claims_group) < 2:
            continue
        claims_group.sort(key=lambda c: c.valid_from or "")
        for i in range(len(claims_group) - 1):
            older = claims_group[i]
            newer = claims_group[i + 1]
            if older.object_name != newer.object_name:
                # Snapshot before temporal supersede
                older_snap = older.model_dump()
                newer_snap = newer.model_dump()

                # Conflict: old claim is now historical
                older.status = ClaimStatus.HISTORICAL
                older.valid_until = newer.valid_from
                older.superseded_by = newer.id
                merge_log.append(MergeRecord(
                    merge_type="claim",
                    winner_id=newer.id,
                    loser_id=older.id,
                    reason=f"temporal_supersede: '{older.object_name}' → '{newer.object_name}'",
                    original_snapshots={
                        "winner": newer_snap,
                        "loser": older_snap,
                    },
                ))

    # Quality gate: filter low-confidence claims
    before_filter = len(final_claims)
    final_claims = [c for c in final_claims if c.confidence >= CONFIDENCE_THRESHOLD]
    filtered = before_filter - len(final_claims)

    print(f"  Claim dedup: {len(all_claims)} → {len(final_claims)} "
          f"({len(all_claims) - len(final_claims)} merged/filtered, "
          f"{filtered} below confidence threshold)")

    return final_claims, merge_log


# ═════════════════════════════════════════════════════════════════════════════
# OOP Deduplicator class
# ═════════════════════════════════════════════════════════════════════════════
class Deduplicator:
    """
    OOP wrapper around the dedup pipeline.
    Provides run_full_pipeline(), undo_merge(), orphan cleanup.
    """

    def __init__(self, extractions: list[ExtractionResult]):
        self.extractions = extractions
        self.entities: list[Entity] = []
        self.claims: list[Claim] = []
        self.merge_log: list[MergeRecord] = []
        self.id_remap: dict[str, str] = {}

    def run_full_pipeline(self) -> "Deduplicator":
        """Execute all three dedup layers in sequence."""
        # Layer 1: Artifact dedup
        self.extractions = dedup_artifacts(self.extractions)

        # Flatten
        all_entities = []
        all_claims = []
        for ext in self.extractions:
            all_entities.extend(ext.entities)
            all_claims.extend(ext.claims)

        print(f"\n  Raw totals: {len(all_entities)} entities, {len(all_claims)} claims")

        # Layer 2: Entity canonicalization
        self.entities, self.id_remap, entity_merges = canonicalize_entities(all_entities)

        # Layer 3: Claim dedup + conflict resolution
        self.claims, claim_merges = dedup_claims(all_claims, self.id_remap)

        self.merge_log = entity_merges + claim_merges

        # Layer 4: Orphan cleanup
        self._cleanup_orphans()

        print(f"\n  Final: {len(self.entities)} entities, {len(self.claims)} claims")
        print(f"  Merge log: {len(self.merge_log)} operations")

        return self

    def _cleanup_orphans(self) -> int:
        """Remove entities not referenced by any claim."""
        referenced = set()
        for c in self.claims:
            referenced.add(c.subject_id)
            referenced.add(c.object_id)
        before = len(self.entities)
        self.entities = [e for e in self.entities if e.id in referenced]
        removed = before - len(self.entities)
        if removed:
            print(f"  Orphan cleanup: removed {removed} unreferenced entities")
        return removed

    def to_memory_store(self) -> MemoryStore:
        """Convert results to a MemoryStore container."""
        store = MemoryStore()
        for e in self.entities:
            store.add_entity(e)
        for c in self.claims:
            store.add_claim(c)
        store.merge_log = self.merge_log
        return store

    def save(self, output_dir: str = OUTPUT_DIR) -> str:
        """Save dedup results to JSON."""
        output = {
            "entities": [e.model_dump() for e in self.entities],
            "claims": [c.model_dump() for c in self.claims],
            "merge_log": [m.model_dump() for m in self.merge_log],
            "id_remap": self.id_remap,
        }
        dedup_path = os.path.join(output_dir, "deduped.json")
        with open(dedup_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Saved to: {dedup_path}")
        return dedup_path


# ═════════════════════════════════════════════════════════════════════════════
# Undo merge
# ═════════════════════════════════════════════════════════════════════════════
def undo_merge(
    store: MemoryStore,
    merge_id: str,
) -> bool:
    """
    Reverse a single merge using its stored snapshots.
    Returns True if successful, False if not found or not reversible.
    """
    # Find the merge record
    target: MergeRecord | None = None
    target_idx: int = -1
    for i, m in enumerate(store.merge_log):
        if m.id == merge_id:
            target = m
            target_idx = i
            break

    if target is None:
        print(f"  undo_merge: merge {merge_id} not found")
        return False

    if not target.reversible:
        print(f"  undo_merge: merge {merge_id} is not reversible")
        return False

    snaps = target.original_snapshots
    if not snaps or "winner" not in snaps or "loser" not in snaps:
        print(f"  undo_merge: merge {merge_id} has no snapshots – cannot undo")
        return False

    if target.merge_type == "entity":
        # Restore the winner to its pre-merge state
        winner_data = snaps["winner"]
        loser_data = snaps["loser"]
        winner_entity = Entity(**winner_data)
        loser_entity = Entity(**loser_data)

        # Replace the current winner with the pre-merge version
        store.entities[winner_entity.id] = winner_entity
        # Re-add the loser entity
        store.entities[loser_entity.id] = loser_entity

        # Remap claims: any claim pointing to winner that originally
        # pointed to loser should be restored
        for c in store.claims.values():
            if c.subject_id == winner_entity.id:
                # Check if the subject_name matches the loser
                if c.subject_name.lower().strip() == loser_entity.name.lower().strip():
                    c.subject_id = loser_entity.id
            if c.object_id == winner_entity.id:
                if c.object_name.lower().strip() == loser_entity.name.lower().strip():
                    c.object_id = loser_entity.id

    elif target.merge_type == "claim":
        winner_data = snaps["winner"]
        loser_data = snaps["loser"]
        winner_claim = Claim(**winner_data)
        loser_claim = Claim(**loser_data)

        # Restore the winner to its pre-merge state
        store.claims[winner_claim.id] = winner_claim
        # Re-add the loser claim
        store.claims[loser_claim.id] = loser_claim

    # Remove the merge record from the log
    store.merge_log.pop(target_idx)
    print(f"  undo_merge: successfully reversed {target.merge_type} merge {merge_id}")
    return True


# ═════════════════════════════════════════════════════════════════════════════
# Main orchestration (backward-compatible wrapper)
# ═════════════════════════════════════════════════════════════════════════════
def run_dedup(extraction_path: str = EXTRACTION_PATH):
    """Full deduplication pipeline."""
    print("=" * 60)
    print("Layer10 Memory Pipeline - Deduplication & Canonicalization")
    print("=" * 60)

    # Load extractions
    with open(extraction_path) as f:
        raw = json.load(f)

    extractions = [ExtractionResult(**r) for r in raw]
    print(f"\nLoaded {len(extractions)} extraction results")

    # Run OOP pipeline
    deduper = Deduplicator(extractions)
    deduper.run_full_pipeline()
    deduper.save()

    return deduper.entities, deduper.claims, deduper.merge_log, deduper.id_remap


if __name__ == "__main__":
    run_dedup()
