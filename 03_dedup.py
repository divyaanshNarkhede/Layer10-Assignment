"""
The deduplication step — this is where we clean up all the redundancy.

After extraction, we end up with hundreds of near-identical entities
("K. Lay", "Kenneth Lay", "Ken Lay") and duplicate claims from the
same fact appearing across multiple threads. This step collapses all
of that into one clean canonical graph.

Three layers, applied in order:
  1. Artifact dedup  – merge extractions from near-identical forwarded emails
  2. Entity dedup    – exact string match, then semantic similarity via embeddings
  3. Claim dedup     – merge identical (subject, relation, object) triples,
                       and mark older role/status claims as historical

Every merge is logged with a before/after snapshot so it can be undone.
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

# ── Only load the embedding model when we actually need it — it's slow to import
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


# ── Union-Find data structure for tracking which entities have been merged ────
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


# ── Layer 1: Drop extractions from emails we've already seen ────────────────
def dedup_artifacts(extractions: list[ExtractionResult]) -> list[ExtractionResult]:
    """Merges extractions that came from near-identical emails (e.g. the same email forwarded twice)."""
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


# ── Layer 2: Merge entities that refer to the same real-world thing ─────────
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

    # First group everything by its canonical key (lowercased name + type)
    key_to_entities: dict[str, list[Entity]] = defaultdict(list)
    for ent in all_entities:
        key_to_entities[ent.canonical_key()].append(ent)

    # Phase 1: Merge anything with an identical name and type
    canonical_map: dict[str, Entity] = {}
    for key, group in key_to_entities.items():
        winner = group[0]
        for other in group[1:]:
            # Save the state of both sides before we modify anything — needed for undo
            winner_snap = winner.model_dump()
            loser_snap = other.model_dump()

            uf.union(winner.id, other.id)
            # Merge aliases
            for alias in other.aliases:
                if alias not in winner.aliases:
                    winner.aliases.append(alias)
            if other.name not in winner.aliases and other.name != winner.name:
                winner.aliases.append(other.name)
            # Expand the winner's time window to cover the loser's dates too
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

    # Phase 2: For anything that didn't match exactly, try embedding similarity
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
                        # Capture both sides before touching anything so we can undo this later
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

    # Build the final list and a mapping from old IDs to canonical IDs
    id_to_entity = {e.id: e for e in canonical_list}
    # We need to be able to resolve any entity ID back to its canonical root
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


# ── Layer 3: Merge duplicate claims and handle conflicting facts over time ────
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

    # First, rewrite all subject/object IDs to point to canonical entities
    for claim in all_claims:
        claim.subject_id = id_remap.get(claim.subject_id, claim.subject_id)
        claim.object_id = id_remap.get(claim.object_id, claim.object_id)

    # Group claims that say the same thing (same subject, relation, and object)
    key_to_claims: dict[str, list[Claim]] = defaultdict(list)
    for claim in all_claims:
        key = claim.canonical_key()
        key_to_claims[key].append(claim)

    final_claims = []
    for key, group in key_to_claims.items():
        if len(group) == 1:
            final_claims.append(group[0])
            continue

        # Put them in chronological order so the newest one becomes the winner
        group.sort(key=lambda c: c.valid_from or "")

        # Keep the most recent claim and fold all the evidence from older ones into it
        winner = group[-1]  # latest timestamp
        for older in group[:-1]:
            # Snapshot both sides before we change anything
            winner_snap = winner.model_dump()
            older_snap = older.model_dump()

            # Fold in any evidence the older claim had that the winner doesn't
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

    # Now handle temporal conflicts — same person, different role at different times
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

    # Drop anything below the confidence threshold — it's more noise than signal
    before_filter = len(final_claims)
    final_claims = [c for c in final_claims if c.confidence >= CONFIDENCE_THRESHOLD]
    filtered = before_filter - len(final_claims)

    print(f"  Claim dedup: {len(all_claims)} → {len(final_claims)} "
          f"({len(all_claims) - len(final_claims)} merged/filtered, "
          f"{filtered} below confidence threshold)")

    return final_claims, merge_log


# ── A clean class wrapper around the whole dedup pipeline ───────────────────
class Deduplicator:
    """
    Wraps all three dedup layers into a single object you can run
    and inspect. Also handles orphan cleanup and saving results.
    """

    def __init__(self, extractions: list[ExtractionResult]):
        self.extractions = extractions
        self.entities: list[Entity] = []
        self.claims: list[Claim] = []
        self.merge_log: list[MergeRecord] = []
        self.id_remap: dict[str, str] = {}

    def run_full_pipeline(self) -> "Deduplicator":
        """Runs all three dedup layers in order and returns self for chaining."""
        # Start with artifact dedup — collapse near-identical forwarded emails
        self.extractions = dedup_artifacts(self.extractions)

        # Flatten everything into two big lists before the entity/claim passes
        all_entities = []
        all_claims = []
        for ext in self.extractions:
            all_entities.extend(ext.entities)
            all_claims.extend(ext.claims)

        print(f"\n  Raw totals: {len(all_entities)} entities, {len(all_claims)} claims")

        # Merge entities that refer to the same person/org/project
        self.entities, self.id_remap, entity_merges = canonicalize_entities(all_entities)

        # Merge duplicate claims and mark superseded ones as historical
        self.claims, claim_merges = dedup_claims(all_claims, self.id_remap)

        self.merge_log = entity_merges + claim_merges

        # Remove any entities that ended up with no claims pointing to them
        self._cleanup_orphans()

        print(f"\n  Final: {len(self.entities)} entities, {len(self.claims)} claims")
        print(f"  Merge log: {len(self.merge_log)} operations")

        return self

    def _cleanup_orphans(self) -> int:
        """Deletes any entities that no claim references — they'd just be floating noise."""
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
        """Packages up the dedup results into a MemoryStore ready for graph building."""
        store = MemoryStore()
        for e in self.entities:
            store.add_entity(e)
        for c in self.claims:
            store.add_claim(c)
        store.merge_log = self.merge_log
        return store

    def save(self, output_dir: str = OUTPUT_DIR) -> str:
        """Writes the deduped entities, claims, and merge log to output/deduped.json."""
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


# ── Undo a specific merge by restoring both sides from their snapshots ───────
def undo_merge(
    store: MemoryStore,
    merge_id: str,
) -> bool:
    """
    Reverses a single merge operation using the before/after snapshots stored
    in the MergeRecord. Returns True on success, False if it can't be undone.
    """
    # Search the merge log for the record with this ID
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

        # Restore the winner claim to its pre-merge state
        store.claims[winner_claim.id] = winner_claim
        # Bring the loser claim back as well
        store.claims[loser_claim.id] = loser_claim

    # Remove the merge record itself so the audit log stays accurate
    store.merge_log.pop(target_idx)
    print(f"  undo_merge: successfully reversed {target.merge_type} merge {merge_id}")
    return True


# ── Top-level runner — called by run_pipeline.py or directly via __main__ ────
def run_dedup(extraction_path: str = EXTRACTION_PATH):
    """Loads extractions from disk and runs the full dedup pipeline."""
    print("=" * 60)
    print("Layer10 Memory Pipeline - Deduplication & Canonicalization")
    print("=" * 60)

    # Pull the raw extractions that 02_extract.py produced
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
