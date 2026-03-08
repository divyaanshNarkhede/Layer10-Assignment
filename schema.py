"""
Ontology / Schema for the Layer10 Memory Graph.

Defines all Pydantic models for:
  - Entities  (Person, Organisation, System, Project, FinancialInstrument, Event)
  - Claims    (typed relations between entities with confidence)
  - Evidence  (grounded pointers back to source text)
  - Extraction envelope (versioned wrapper)
"""
from __future__ import annotations
import uuid, hashlib
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ─── Entity Types ────────────────────────────────────────────────────────────
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
    """A canonical entity in the memory graph."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    type: EntityType
    aliases: list[str] = Field(default_factory=list)
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

    def canonical_key(self) -> str:
        """Deterministic key for dedup: lowered name + type."""
        return f"{self.type.value}::{self.name.lower().strip()}"


# ─── Relation / Claim Types ─────────────────────────────────────────────────
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


# ─── Evidence ────────────────────────────────────────────────────────────────
class Evidence(BaseModel):
    """A grounded pointer back to a source artifact."""
    source_id: str                         # email_id
    excerpt: str                           # exact quote (≤ 300 chars)
    timestamp: Optional[str] = None        # event time
    source_type: str = "email"
    author: Optional[str] = None
    subject: Optional[str] = None          # email subject
    char_offset_start: Optional[int] = None  # start char offset in source
    char_offset_end: Optional[int] = None    # end char offset in source

    def fingerprint(self) -> str:
        """Content hash for dedup of identical evidence."""
        raw = f"{self.source_id}|{self.excerpt[:100]}"
        return hashlib.md5(raw.encode()).hexdigest()


# ─── Claim ───────────────────────────────────────────────────────────────────
class ClaimStatus(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"       # superseded by a newer claim
    RETRACTED = "retracted"         # source was deleted / redacted
    DISPUTED = "disputed"           # conflicting evidence exists


class Claim(BaseModel):
    """A single factual assertion linking two entities, grounded in evidence."""
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
        """Key for claim dedup (ignores evidence list)."""
        return (
            f"{self.subject_name.lower().strip()}|"
            f"{self.relation.value}|"
            f"{self.object_name.lower().strip()}"
        )


# ─── Extraction Envelope ────────────────────────────────────────────────────
class ExtractionResult(BaseModel):
    """Wrapper returned by the extraction step for one source artifact."""
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


# ─── Merge / Dedup Records ──────────────────────────────────────────────────
class MergeRecord(BaseModel):
    """Audit trail for every entity or claim merge."""
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


# ─── Full Graph Snapshot ─────────────────────────────────────────────────────
class MemoryGraphSnapshot(BaseModel):
    """Serialisable snapshot of the entire memory graph."""
    entities: list[Entity] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    merge_log: list[MergeRecord] = Field(default_factory=list)
    schema_version: str = "1.0.0"
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    corpus_info: dict = Field(default_factory=dict)


# ─── Memory Store (first-class container) ────────────────────────────────────
class MemoryStore(BaseModel):
    """
    First-class container for the memory graph.
    Provides indexed access, add/remove helpers, and serialisation.
    """
    entities: dict[str, Entity] = Field(default_factory=dict)
    claims: dict[str, Claim] = Field(default_factory=dict)
    merge_log: list[MergeRecord] = Field(default_factory=list)
    schema_version: str = "1.0.0"
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    corpus_info: dict = Field(default_factory=dict)

    # ── Entity helpers ────────────────────────────────────────────────────
    def add_entity(self, entity: Entity) -> Entity:
        """Add an entity (or update if same ID exists)."""
        self.entities[entity.id] = entity
        return entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def remove_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.pop(entity_id, None)

    def find_entities_by_name(self, name: str) -> list[Entity]:
        """Case-insensitive search by name or alias."""
        needle = name.lower().strip()
        return [
            e for e in self.entities.values()
            if needle in e.name.lower()
            or any(needle in a.lower() for a in e.aliases)
        ]

    # ── Claim helpers ─────────────────────────────────────────────────────
    def add_claim(self, claim: Claim) -> Claim:
        self.claims[claim.id] = claim
        return claim

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        return self.claims.get(claim_id)

    def remove_claim(self, claim_id: str) -> Optional[Claim]:
        return self.claims.pop(claim_id, None)

    def get_claims_for_entity(self, entity_id: str) -> list[Claim]:
        """Return all claims where entity is subject or object."""
        return [
            c for c in self.claims.values()
            if c.subject_id == entity_id or c.object_id == entity_id
        ]

    def get_claims_between(self, entity_a: str, entity_b: str) -> list[Claim]:
        """Return claims linking two specific entities."""
        return [
            c for c in self.claims.values()
            if {c.subject_id, c.object_id} == {entity_a, entity_b}
        ]

    # ── Merge log ─────────────────────────────────────────────────────────
    def add_merge(self, record: MergeRecord) -> None:
        self.merge_log.append(record)

    # ── Serialisation ─────────────────────────────────────────────────────
    def to_snapshot(self) -> MemoryGraphSnapshot:
        """Convert to a flat MemoryGraphSnapshot for JSON serialisation."""
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
        """Build a MemoryStore from a flat snapshot."""
        return cls(
            entities={e.id: e for e in snap.entities},
            claims={c.id: c for c in snap.claims},
            merge_log=snap.merge_log,
            schema_version=snap.schema_version,
            created_at=snap.created_at,
            corpus_info=snap.corpus_info,
        )

    def serialize(self) -> dict:
        """Return a JSON-serialisable dict."""
        return self.to_snapshot().model_dump()

    @classmethod
    def deserialize(cls, data: dict) -> "MemoryStore":
        """Load from a dict (e.g. from JSON file)."""
        snap = MemoryGraphSnapshot(**data)
        return cls.from_snapshot(snap)

    # ── Orphan cleanup ────────────────────────────────────────────────────
    def remove_orphan_entities(self) -> int:
        """Remove entities not referenced by any claim."""
        referenced = set()
        for c in self.claims.values():
            referenced.add(c.subject_id)
            referenced.add(c.object_id)
        orphans = [eid for eid in self.entities if eid not in referenced]
        for eid in orphans:
            del self.entities[eid]
        return len(orphans)
