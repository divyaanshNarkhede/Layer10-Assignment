"""
All the data models for the Layer10 memory system live here.

These Pydantic classes are the single source of truth for what an entity,
a claim, and a piece of evidence actually look like. Every other module
imports from this file — the extractor, the deduplicator, the graph builder,
and the UI all work with these same typed objects.

What's in here:
  - Entity       — a person, org, project, etc.
  - Claim        — a typed, evidenced relationship between two entities
  - Evidence     — a verbatim quote with char offsets back to the source email
  - ExtractionResult — the envelope wrapping one thread's extraction output
  - MergeRecord  — audit trail entry for every dedup operation
  - MemoryStore  — the main in-memory container for the whole graph
"""
from __future__ import annotations
import uuid, hashlib
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── The eight types of things we track in the graph ─────────────────────────
class EntityType(str, Enum):
    PERSON = "Person"
    ORGANISATION = "Organisation"
    SYSTEM = "System"               # e.g. EnronOnline, RiskTrak
    PROJECT = "Project"             # e.g. Raptor, LJM, Broadband
    FINANCIAL_INSTRUMENT = "FinancialInstrument"  # SPEs, funds
    EVENT = "Event"                 # meetings, filings, announcements
    ROLE = "Role"                   # CEO, Treasurer, etc.
    LOCATION = "Location"           # Houston, Portland, etc.


class Entity(BaseModel):
    """One node in the knowledge graph — a person, org, project, or anything else we track."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    type: EntityType
    aliases: list[str] = Field(default_factory=list)
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

    def canonical_key(self) -> str:
        """Returns a stable key used to detect duplicate entities during dedup."""
        return f"{self.type.value}::{self.name.lower().strip()}"


# ── Every type of relationship a claim can express ──────────────────────────
class RelationType(str, Enum):
    # Organisational
    WORKS_AT = "works_at"
    HAS_ROLE = "has_role"
    REPORTS_TO = "reports_to"
    MANAGES = "manages"
    MEMBER_OF = "member_of"

    # Actions
    CREATED = "created"
    SENT_EMAIL = "sent_email"
    DECIDED = "decided"
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    WARNED_ABOUT = "warned_about"
    INVESTIGATED = "investigated"

    # States / facts
    HAS_STATUS = "has_status"
    HAS_VALUE = "has_value"
    DEPENDS_ON = "depends_on"
    CONFLICTS_WITH = "conflicts_with"
    SUPERSEDES = "supersedes"
    RELATED_TO = "related_to"

    # Financial
    INVESTED_IN = "invested_in"
    PROFITED_FROM = "profited_from"
    OWES = "owes"
    ACQUIRED = "acquired"


# ── A grounded pointer back to the exact email that supports a claim ─────────
class Evidence(BaseModel):
    """Ties a claim back to the exact quote in the exact email that supports it."""
    source_id: str                         # email_id
    excerpt: str                           # exact quote (≤ 300 chars)
    timestamp: Optional[str] = None        # event time
    source_type: str = "email"
    author: Optional[str] = None
    subject: Optional[str] = None          # email subject
    char_offset_start: Optional[int] = None  # start char offset in source
    char_offset_end: Optional[int] = None    # end char offset in source

    def fingerprint(self) -> str:
        """A short hash we use to detect when two evidence objects are actually the same quote."""
        raw = f"{self.source_id}|{self.excerpt[:100]}"
        return hashlib.md5(raw.encode()).hexdigest()


# ── A single factual assertion between two entities, grounded in evidence ─────
class ClaimStatus(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"       # superseded by a newer claim
    RETRACTED = "retracted"         # source was deleted / redacted
    DISPUTED = "disputed"           # conflicting evidence exists


class Claim(BaseModel):
    """One edge in the knowledge graph — something we believe to be true, with a source to back it up."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    subject_id: str
    subject_name: str
    relation: RelationType
    object_id: str
    object_name: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    status: ClaimStatus = ClaimStatus.CURRENT
    evidence: list[Evidence] = Field(default_factory=list)
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None      # set when superseded
    superseded_by: Optional[str] = None    # claim id
    extraction_version: str = "v1"
    metadata: dict = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v):
        return max(0.0, min(1.0, v))

    def canonical_key(self) -> str:
        """Returns a signature for this claim used to find duplicates (ignores evidence and confidence)."""
        return (
            f"{self.subject_name.lower().strip()}|"
            f"{self.relation.value}|"
            f"{self.object_name.lower().strip()}"
        )


# ── The output envelope for one thread's extraction pass ────────────────────
class ExtractionResult(BaseModel):
    """Everything the extractor found in a single email thread — entities, claims, and metadata."""
    source_id: str
    entities: list[Entity] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    extraction_version: str = "v1"
    model: str = ""
    prompt_hash: str = ""
    extracted_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    raw_llm_output: Optional[str] = None   # for debugging


# ── Audit trail — one record for every merge the deduplicator performed ──────
class MergeRecord(BaseModel):
    """Records what got merged, why, and what both sides looked like before the merge."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    merge_type: str                # "entity" or "claim"
    winner_id: str
    loser_id: str
    reason: str                    # "exact_match", "semantic_similarity=0.93"
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    reversible: bool = True        # can be undone
    original_snapshots: dict = Field(
        default_factory=dict,
        description="Pre-merge snapshots of winner/loser for undo support",
    )


# ── A flat, JSON-serializable snapshot of the whole graph at a point in time ──
class MemoryGraphSnapshot(BaseModel):
    """A flat list representation of the graph — easy to write to JSON and read back."""
    entities: list[Entity] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    merge_log: list[MergeRecord] = Field(default_factory=list)
    schema_version: str = "1.0.0"
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    corpus_info: dict = Field(default_factory=dict)


# ── The main in-memory container — indexed access, helpers, and serialization ─
class MemoryStore(BaseModel):
    """
    The main working container for the graph during the pipeline.
    Gives you fast dict-based lookups, helper methods for finding
    entities and claims, and easy JSON serialization.
    """
    entities: dict[str, Entity] = Field(default_factory=dict)
    claims: dict[str, Claim] = Field(default_factory=dict)
    merge_log: list[MergeRecord] = Field(default_factory=list)
    schema_version: str = "1.0.0"
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    corpus_info: dict = Field(default_factory=dict)

    # ── Helpers for working with entities ────────────────────────────────
    def add_entity(self, entity: Entity) -> Entity:
        """Adds or overwrites an entity in the store by its ID."""
        self.entities[entity.id] = entity
        return entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def remove_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.pop(entity_id, None)

    def find_entities_by_name(self, name: str) -> list[Entity]:
        """Searches both names and aliases, case-insensitively."""
        needle = name.lower().strip()
        return [
            e for e in self.entities.values()
            if needle in e.name.lower()
            or any(needle in a.lower() for a in e.aliases)
        ]

    # ── Helpers for working with claims ─────────────────────────────────
    def add_claim(self, claim: Claim) -> Claim:
        self.claims[claim.id] = claim
        return claim

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        return self.claims.get(claim_id)

    def remove_claim(self, claim_id: str) -> Optional[Claim]:
        return self.claims.pop(claim_id, None)

    def get_claims_for_entity(self, entity_id: str) -> list[Claim]:
        """Returns every claim that involves this entity, whether as subject or object."""
        return [
            c for c in self.claims.values()
            if c.subject_id == entity_id or c.object_id == entity_id
        ]

    def get_claims_between(self, entity_a: str, entity_b: str) -> list[Claim]:
        """Finds all claims that link these two specific entities together."""
        return [
            c for c in self.claims.values()
            if {c.subject_id, c.object_id} == {entity_a, entity_b}
        ]

    # ── Append to the audit trail ────────────────────────────────────────
    def add_merge(self, record: MergeRecord) -> None:
        self.merge_log.append(record)

    # ── Read/write the store to and from JSON ───────────────────────────
    def to_snapshot(self) -> MemoryGraphSnapshot:
        """Flattens the store into a snapshot that's easy to write to disk."""
        return MemoryGraphSnapshot(
            entities=list(self.entities.values()),
            claims=list(self.claims.values()),
            merge_log=self.merge_log,
            schema_version=self.schema_version,
            created_at=self.created_at,
            corpus_info=self.corpus_info,
        )

    @classmethod
    def from_snapshot(cls, snap: MemoryGraphSnapshot) -> "MemoryStore":
        """Rebuilds a MemoryStore from a snapshot that was loaded from disk."""
        return cls(
            entities={e.id: e for e in snap.entities},
            claims={c.id: c for c in snap.claims},
            merge_log=snap.merge_log,
            schema_version=snap.schema_version,
            created_at=snap.created_at,
            corpus_info=snap.corpus_info,
        )

    def serialize(self) -> dict:
        """Returns the whole store as a plain dict ready for json.dump()."""
        return self.to_snapshot().model_dump()

    @classmethod
    def deserialize(cls, data: dict) -> "MemoryStore":
        """Loads a MemoryStore from a dict — typically the result of json.load()."""
        snap = MemoryGraphSnapshot(**data)
        return cls.from_snapshot(snap)

    # ── Remove entities that no claim points to anymore ─────────────────
    def remove_orphan_entities(self) -> int:
        """Deletes any entity that isn't referenced by at least one claim."""
        referenced = set()
        for c in self.claims.values():
            referenced.add(c.subject_id)
            referenced.add(c.object_id)
        orphans = [eid for eid in self.entities if eid not in referenced]
        for eid in orphans:
            del self.entities[eid]
        return len(orphans)
