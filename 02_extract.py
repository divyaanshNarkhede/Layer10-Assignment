"""
Step 2 – Structured Extraction Pipeline (thread-wise).

For each *conversation thread*, feeds the full thread to a local LLM
(via Ollama, GPU-accelerated) to extract typed entities and grounded
claims.  Includes:
  • System prompt with explicit JSON schema
  • Pydantic validation loop with retries
  • Deterministic rule-based fallback / supplement
  • Confidence scoring
  • Version tracking

The input format is a list of thread dicts:
    { thread_id, subject, messages: [{email_id, from, to, ...}], ... }
"""
import json, os, sys, hashlib, time, re
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    CORPUS_RAW_PATH, EXTRACTION_PATH, OLLAMA_MODEL,
    MAX_RETRIES, CONFIDENCE_THRESHOLD, OUTPUT_DIR
)
from schema import (
    Entity, EntityType, Claim, ClaimStatus, Evidence,
    RelationType, ExtractionResult
)

# ─── Try to import ollama; provide fallback ──────────────────────────────────
try:
    if os.environ.get("DISABLE_OLLAMA"):
        raise ImportError("Ollama disabled via DISABLE_OLLAMA env var")
    import ollama as ollama_client
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

EXTRACTION_VERSION = "v2-thread"

# ─── System Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an information extraction engine. Given an EMAIL CONVERSATION THREAD (multiple related emails), extract structured data as JSON.

Return ONLY a valid JSON object with this exact structure (no markdown, no commentary):
{
  "entities": [
    {
      "name": "exact entity name",
      "type": "Person|Organisation|System|Project|FinancialInstrument|Event|Role|Location",
      "aliases": ["alternate names if any"]
    }
  ],
  "claims": [
    {
      "subject": "entity name (must match an entity above)",
      "relation": "works_at|has_role|reports_to|manages|member_of|created|sent_email|decided|proposed|approved|rejected|warned_about|investigated|has_status|has_value|depends_on|conflicts_with|supersedes|related_to|invested_in|profited_from|owes|acquired",
      "object": "entity name or value (must match an entity above or be a simple value)",
      "confidence": 0.0 to 1.0,
      "quote": "exact short quote from one of the emails supporting this claim (max 200 chars)",
      "email_id": "the email_id containing the supporting quote"
    }
  ]
}

Rules:
1. Read the ENTIRE thread to understand context and evolution of the conversation.
2. Extract ALL people, organisations, systems, projects, financial instruments, events, roles, and locations mentioned across all emails.
3. Every claim MUST have an exact quote from one of the emails as evidence, along with its email_id.
4. Only use relation types from the list above.
5. Confidence: 1.0 = explicitly stated, 0.7 = strongly implied, 0.5 = loosely implied.
6. Track how facts evolve across the thread - if something changes, create separate claims with appropriate confidence.
7. Keep entity names concise and consistent across all emails.
8. Return ONLY the JSON object, nothing else."""


THREAD_PROMPT_TEMPLATE = """Extract entities and claims from this email conversation thread:

Thread Subject: {subject}
Thread ID: {thread_id}
Message Count: {message_count}
Participants: {participants}

--- EMAILS IN CHRONOLOGICAL ORDER ---
{emails_text}
--- END OF THREAD ---
"""

SINGLE_EMAIL_TEMPLATE = """[Email {idx}]
Email ID: {email_id}
From: {sender}
To: {to}
Date: {timestamp}
Subject: {subject}

{body}
"""


# ─── Prompt Hash ─────────────────────────────────────────────────────────────
PROMPT_HASH = hashlib.md5(SYSTEM_PROMPT.encode()).hexdigest()[:8]


def call_ollama(thread_data: dict) -> str | None:
    """Call Ollama with GPU acceleration and return raw text response."""
    if not HAS_OLLAMA:
        return None

    # Build the full thread text
    messages = thread_data.get("messages", [])
    emails_text = ""
    for i, msg in enumerate(messages):
        emails_text += SINGLE_EMAIL_TEMPLATE.format(
            idx=i + 1,
            email_id=msg.get("email_id", ""),
            sender=msg.get("from", ""),
            to=msg.get("to", ""),
            timestamp=msg.get("timestamp", ""),
            subject=msg.get("subject", ""),
            body=msg.get("body", "")[:1000],
        )

    user_msg = THREAD_PROMPT_TEMPLATE.format(
        subject=thread_data.get("subject", ""),
        thread_id=thread_data.get("thread_id", ""),
        message_count=len(messages),
        participants=", ".join(thread_data.get("participants", [])),
        emails_text=emails_text,
    )

    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            options={
                "temperature": 0.1,
                "num_predict": 2048,
                "num_gpu": 99,         # offload all layers to GPU
                "num_ctx": 4096,       # fits in 6GB VRAM with 3B model
            },
            keep_alive="10m",
        )
        return response["message"]["content"]
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"    Ollama error: {e}")
        return None


def parse_llm_json(raw: str) -> dict | None:
    """Best-effort extraction of JSON from LLM output."""
    if not raw:
        return None
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try to find JSON block
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'(\{.*\})',
    ]
    for pat in patterns:
        m = re.search(pat, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return None


def normalize_entity_type(raw_type: str) -> EntityType:
    """Map raw string to EntityType enum, with fallback."""
    mapping = {
        "person": EntityType.PERSON,
        "organisation": EntityType.ORGANISATION,
        "organization": EntityType.ORGANISATION,
        "org": EntityType.ORGANISATION,
        "system": EntityType.SYSTEM,
        "project": EntityType.PROJECT,
        "financialinstrument": EntityType.FINANCIAL_INSTRUMENT,
        "financial_instrument": EntityType.FINANCIAL_INSTRUMENT,
        "event": EntityType.EVENT,
        "role": EntityType.ROLE,
        "location": EntityType.LOCATION,
    }
    return mapping.get((raw_type or "").lower().strip(), EntityType.PROJECT)


def normalize_relation(raw_rel: str) -> RelationType:
    """Map raw string to RelationType enum."""
    raw_rel = raw_rel or "related_to"
    try:
        return RelationType(raw_rel.lower().strip())
    except ValueError:
        for rt in RelationType:
            if raw_rel.lower().replace(" ", "_") == rt.value:
                return rt
        return RelationType.RELATED_TO


def build_extraction(thread_data: dict, parsed: dict) -> ExtractionResult:
    """Convert parsed LLM JSON into validated Pydantic models."""
    thread_id = thread_data.get("thread_id", "")
    messages = thread_data.get("messages", [])
    # Build a map of email_ids for evidence lookup
    msg_map = {m["email_id"]: m for m in messages}
    first_ts = messages[0].get("timestamp", "") if messages else ""
    last_ts = messages[-1].get("timestamp", "") if messages else ""

    entities = []
    entity_name_to_id = {}

    for raw_ent in parsed.get("entities", []):
        name = (raw_ent.get("name") or "").strip()
        if not name:
            continue
        etype = normalize_entity_type(raw_ent.get("type", "Project"))
        aliases = raw_ent.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]

        ent = Entity(
            name=name,
            type=etype,
            aliases=aliases,
            first_seen=first_ts,
            last_seen=last_ts,
        )
        entities.append(ent)
        entity_name_to_id[name.lower()] = ent.id

    claims = []
    for raw_claim in parsed.get("claims", []):
        subj_name = (raw_claim.get("subject") or "").strip()
        obj_name = (raw_claim.get("object") or "").strip()
        if not subj_name or not obj_name:
            continue

        relation = normalize_relation(raw_claim.get("relation") or "related_to")
        try:
            confidence = float(raw_claim.get("confidence") or 0.7)
        except (ValueError, TypeError):
            confidence = 0.7
        quote = (raw_claim.get("quote") or "")[:300]
        ev_email_id = raw_claim.get("email_id", thread_id)

        # Resolve entity IDs (create if not found)
        subj_id = entity_name_to_id.get(subj_name.lower(), "")
        obj_id = entity_name_to_id.get(obj_name.lower(), "")

        if not subj_id:
            ent = Entity(name=subj_name, type=EntityType.PERSON,
                         first_seen=first_ts, last_seen=last_ts)
            entities.append(ent)
            entity_name_to_id[subj_name.lower()] = ent.id
            subj_id = ent.id

        if not obj_id:
            ent = Entity(name=obj_name, type=EntityType.PROJECT,
                         first_seen=first_ts, last_seen=last_ts)
            entities.append(ent)
            entity_name_to_id[obj_name.lower()] = ent.id
            obj_id = ent.id

        # Find evidence context from the referenced email
        ev_msg = msg_map.get(ev_email_id, messages[0] if messages else {})
        ev_body = ev_msg.get("body", "")

        # Compute char offsets for the quote within the email body
        offset_start, offset_end = None, None
        if quote and ev_body:
            idx = ev_body.find(quote)
            if idx == -1:
                # Try case-insensitive fallback
                idx = ev_body.lower().find(quote.lower()[:80])
            if idx >= 0:
                offset_start = idx
                offset_end = idx + len(quote)

        evidence = Evidence(
            source_id=ev_msg.get("email_id", ev_email_id),
            excerpt=quote if quote else f"[Thread: {thread_data.get('subject','')}]",
            timestamp=ev_msg.get("timestamp", first_ts),
            author=ev_msg.get("from", ""),
            subject=ev_msg.get("subject", ""),
            char_offset_start=offset_start,
            char_offset_end=offset_end,
        )

        claim = Claim(
            subject_id=subj_id,
            subject_name=subj_name,
            relation=relation,
            object_id=obj_id,
            object_name=obj_name,
            confidence=confidence,
            evidence=[evidence],
            valid_from=first_ts,
            extraction_version=EXTRACTION_VERSION,
        )
        claims.append(claim)

    return ExtractionResult(
        source_id=thread_id,
        entities=entities,
        claims=claims,
        extraction_version=EXTRACTION_VERSION,
        model=OLLAMA_MODEL,
        prompt_hash=PROMPT_HASH,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Rule-based extraction (per-message within a thread)
# ═════════════════════════════════════════════════════════════════════════════
def rule_based_extraction_thread(thread_data: dict) -> ExtractionResult:
    """
    Deterministic fallback extraction using regex and heuristics.
    Processes every message in the thread.
    """
    thread_id = thread_data.get("thread_id", "")
    messages = thread_data.get("messages", [])

    entities = []
    claims = []
    entity_map: dict[str, Entity] = {}

    def get_or_create_entity(name: str, etype: EntityType,
                             ts: str = "") -> Entity:
        key = f"{etype.value}::{name.lower().strip()}"
        if key in entity_map:
            ent = entity_map[key]
            if ts and (not ent.last_seen or ts > ent.last_seen):
                ent.last_seen = ts
            return ent
        ent = Entity(name=name, type=etype, first_seen=ts, last_seen=ts)
        entity_map[key] = ent
        entities.append(ent)
        return ent

    def make_evidence(source_id: str, quote: str, ts: str,
                      author: str, subject: str,
                      char_offset_start: int | None = None,
                      char_offset_end: int | None = None) -> Evidence:
        return Evidence(
            source_id=source_id, excerpt=quote[:300],
            timestamp=ts, author=author, subject=subject,
            char_offset_start=char_offset_start,
            char_offset_end=char_offset_end,
        )

    # Ensure Enron entity exists
    enron = get_or_create_entity("Enron", EntityType.ORGANISATION)

    for msg in messages:
        source_id = msg.get("email_id", "")
        timestamp = msg.get("timestamp", "")
        author = msg.get("from", "")
        to_field = msg.get("to", "")
        subject = msg.get("subject", "")
        body = msg.get("body", "")

        # --- Sender ---
        sender_name = (author.split("@")[0].replace(".", " ").title()
                       if "@" in author else author)
        sender_ent = get_or_create_entity(sender_name, EntityType.PERSON, timestamp)

        # --- Recipients ---
        for recip in [r.strip() for r in to_field.split(",") if r.strip()]:
            if "@" in recip:
                recip_name = recip.split("@")[0].replace(".", " ").title()
            else:
                recip_name = recip
            if "all-" in recip.lower() or "team" in recip.lower():
                recip_ent = get_or_create_entity(recip_name, EntityType.ORGANISATION, timestamp)
            else:
                recip_ent = get_or_create_entity(recip_name, EntityType.PERSON, timestamp)

            claims.append(Claim(
                subject_id=sender_ent.id, subject_name=sender_ent.name,
                relation=RelationType.SENT_EMAIL,
                object_id=recip_ent.id, object_name=recip_ent.name,
                confidence=1.0,
                evidence=[make_evidence(source_id,
                    f"Email from {author} to {recip} re: {subject}",
                    timestamp, author, subject)],
                valid_from=timestamp,
                extraction_version=EXTRACTION_VERSION,
            ))

        # --- works_at Enron ---
        if "enron.com" in author:
            claims.append(Claim(
                subject_id=sender_ent.id, subject_name=sender_ent.name,
                relation=RelationType.WORKS_AT,
                object_id=enron.id, object_name=enron.name,
                confidence=0.95,
                evidence=[make_evidence(source_id,
                    f"Sender domain: {author}", timestamp, author, subject)],
                valid_from=timestamp,
                extraction_version=EXTRACTION_VERSION,
            ))

        # --- Dollar amounts ---
        for m in re.finditer(r'\$[\d,.]+[MBK]?\b', body):
            amount = m.group()
            ctx_s = max(0, m.start() - 80)
            ctx_e = min(len(body), m.end() + 80)
            context = body[ctx_s:ctx_e].strip()
            val_ent = get_or_create_entity(amount, EntityType.FINANCIAL_INSTRUMENT, timestamp)
            claims.append(Claim(
                subject_id=sender_ent.id, subject_name=sender_ent.name,
                relation=RelationType.HAS_VALUE,
                object_id=val_ent.id, object_name=amount,
                confidence=0.85,
                evidence=[make_evidence(source_id, context, timestamp, author, subject,
                                        char_offset_start=ctx_s, char_offset_end=ctx_e)],
                valid_from=timestamp,
                extraction_version=EXTRACTION_VERSION,
            ))

        # --- Role titles ---
        for m in re.finditer(
            r'\b(CEO|CFO|COO|CRO|CAO|Treasurer|President|Chairman|'
            r'Managing Director|Chief\s+\w+\s+Officer|Head\s+of\s+\w+|'
            r'VP\s+\w+|SVP\s+\w+|EVP\s+\w+|General Counsel)\b',
            body, re.IGNORECASE
        ):
            role_name = m.group().strip()
            role_ent = get_or_create_entity(role_name, EntityType.ROLE, timestamp)
            ctx_start = max(0, m.start()-100)
            ctx_end = m.end()+50
            ctx = body[ctx_start:ctx_end]
            claims.append(Claim(
                subject_id=sender_ent.id, subject_name=sender_ent.name,
                relation=RelationType.HAS_ROLE,
                object_id=role_ent.id, object_name=role_name,
                confidence=0.7,
                evidence=[make_evidence(source_id, ctx[:300], timestamp, author, subject,
                                        char_offset_start=ctx_start, char_offset_end=ctx_end)],
                valid_from=timestamp,
                extraction_version=EXTRACTION_VERSION,
            ))

        # --- Known Enron projects / orgs ---
        known_entities = {
            "EnronOnline": EntityType.SYSTEM, "EOL": EntityType.SYSTEM,
            "Raptor": EntityType.FINANCIAL_INSTRUMENT,
            "LJM": EntityType.FINANCIAL_INSTRUMENT,
            "LJM2": EntityType.FINANCIAL_INSTRUMENT,
            "Chewco": EntityType.FINANCIAL_INSTRUMENT,
            "Whitewing": EntityType.FINANCIAL_INSTRUMENT,
            "JEDI": EntityType.FINANCIAL_INSTRUMENT,
            "Marlin": EntityType.FINANCIAL_INSTRUMENT,
            "Osprey": EntityType.FINANCIAL_INSTRUMENT,
            "Azurix": EntityType.PROJECT, "Dabhol": EntityType.PROJECT,
            "Death Star": EntityType.PROJECT, "Fat Boy": EntityType.PROJECT,
            "Get Shorty": EntityType.PROJECT,
            "Blockbuster": EntityType.ORGANISATION,
            "Arthur Andersen": EntityType.ORGANISATION,
            "Andersen": EntityType.ORGANISATION,
            "Vinson & Elkins": EntityType.ORGANISATION,
            "Dynegy": EntityType.ORGANISATION, "FERC": EntityType.ORGANISATION,
            "SEC": EntityType.ORGANISATION,
            "UBS Warburg": EntityType.ORGANISATION,
            "Moody's": EntityType.ORGANISATION, "S&P": EntityType.ORGANISATION,
            "ChevronTexaco": EntityType.ORGANISATION,
            "Weil Gotshal": EntityType.ORGANISATION,
        }
        for name, etype in known_entities.items():
            if name.lower() in body.lower():
                ent = get_or_create_entity(name, etype, timestamp)
                idx = body.lower().find(name.lower())
                ctx_start = max(0, idx-40)
                ctx_end = min(len(body), idx+len(name)+120)
                ctx = body[ctx_start:ctx_end].strip()
                claims.append(Claim(
                    subject_id=sender_ent.id, subject_name=sender_ent.name,
                    relation=RelationType.RELATED_TO,
                    object_id=ent.id, object_name=name,
                    confidence=0.75,
                    evidence=[make_evidence(source_id, ctx, timestamp, author, subject,
                                            char_offset_start=ctx_start, char_offset_end=ctx_end)],
                    valid_from=timestamp,
                    extraction_version=EXTRACTION_VERSION,
                ))

        # --- Decision / action keywords ---
        decision_kw = {
            "approved": RelationType.APPROVED, "rejected": RelationType.REJECTED,
            "decided": RelationType.DECIDED, "proposed": RelationType.PROPOSED,
            "recommend": RelationType.PROPOSED,
            "warned": RelationType.WARNED_ABOUT,
            "concerned": RelationType.WARNED_ABOUT,
            "investigating": RelationType.INVESTIGATED,
            "resigned": RelationType.HAS_STATUS,
            "terminated": RelationType.HAS_STATUS,
            "filed": RelationType.HAS_STATUS,
            "acquired": RelationType.ACQUIRED,
            "downgrade": RelationType.HAS_STATUS,
            "bankruptcy": RelationType.HAS_STATUS,
        }
        for keyword, rel_type in decision_kw.items():
            if keyword.lower() in body.lower():
                idx = body.lower().find(keyword.lower())
                ctx_start = max(0, idx-60)
                ctx_end = min(len(body), idx+120)
                ctx = body[ctx_start:ctx_end].strip()
                event_name = f"{keyword.title()} - {subject[:50]}"
                event_ent = get_or_create_entity(event_name, EntityType.EVENT, timestamp)
                claims.append(Claim(
                    subject_id=sender_ent.id, subject_name=sender_ent.name,
                    relation=rel_type,
                    object_id=event_ent.id, object_name=event_name,
                    confidence=0.7,
                    evidence=[make_evidence(source_id, ctx, timestamp, author, subject,
                                            char_offset_start=ctx_start, char_offset_end=ctx_end)],
                    valid_from=timestamp,
                    extraction_version=EXTRACTION_VERSION,
                ))

        # --- Locations ---
        for loc in ["Houston", "Portland", "London", "California",
                     "New York", "India", "Brazil", "Argentina",
                     "Maharashtra", "Mumbai"]:
            if loc.lower() in body.lower():
                loc_ent = get_or_create_entity(loc, EntityType.LOCATION, timestamp)
                idx = body.lower().find(loc.lower())
                ctx_start = max(0, idx-40)
                ctx_end = min(len(body), idx+80)
                ctx = body[ctx_start:ctx_end].strip()
                claims.append(Claim(
                    subject_id=sender_ent.id, subject_name=sender_ent.name,
                    relation=RelationType.RELATED_TO,
                    object_id=loc_ent.id, object_name=loc,
                    confidence=0.6,
                    evidence=[make_evidence(source_id, ctx, timestamp, author, subject,
                                            char_offset_start=ctx_start, char_offset_end=ctx_end)],
                    valid_from=timestamp,
                    extraction_version=EXTRACTION_VERSION,
                ))

    return ExtractionResult(
        source_id=thread_id,
        entities=entities,
        claims=claims,
        extraction_version=EXTRACTION_VERSION,
        model="rule_based",
        prompt_hash="deterministic",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Per-thread extraction orchestrator
# ═════════════════════════════════════════════════════════════════════════════
def extract_thread(thread_data: dict) -> ExtractionResult:
    """
    Extract entities & claims from one conversation thread.
    Tries Ollama first (full thread context), supplements with rule-based.
    """
    # Always run rule-based for baseline
    rule_result = rule_based_extraction_thread(thread_data)

    # Try LLM extraction
    llm_result = None
    if HAS_OLLAMA:
        for attempt in range(MAX_RETRIES):
            raw = call_ollama(thread_data)
            if raw:
                parsed = parse_llm_json(raw)
                if parsed:
                    try:
                        llm_result = build_extraction(thread_data, parsed)
                        llm_result.raw_llm_output = raw
                        break
                    except Exception as e:
                        print(f"    Validation error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)

    # Merge: prefer LLM entities/claims, supplement with rule-based
    if llm_result:
        # Build remap: rule-based entity canonical_key → LLM entity ID
        llm_key_to_id = {}
        for ent in llm_result.entities:
            llm_key_to_id[ent.canonical_key()] = ent.id

        seen_keys = set(llm_key_to_id.keys())
        id_remap: dict[str, str] = {}

        for ent in rule_result.entities:
            ck = ent.canonical_key()
            if ck in seen_keys:
                # Rule entity duplicates an LLM entity – record remap
                id_remap[ent.id] = llm_key_to_id[ck]
            else:
                llm_result.entities.append(ent)
                llm_key_to_id[ck] = ent.id
                seen_keys.add(ck)

        seen_claims = {c.canonical_key() for c in llm_result.claims}
        for claim in rule_result.claims:
            # Remap entity IDs in rule-based claims
            claim.subject_id = id_remap.get(claim.subject_id, claim.subject_id)
            claim.object_id = id_remap.get(claim.object_id, claim.object_id)
            if claim.canonical_key() not in seen_claims:
                llm_result.claims.append(claim)
                seen_claims.add(claim.canonical_key())
        return llm_result

    return rule_result


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════════
def _checkpoint_path() -> str:
    """Path to the extraction checkpoint file."""
    return os.path.join(OUTPUT_DIR, "_extraction_checkpoint.json")


def _save_checkpoint(completed_ids: list[str], results: list[dict]) -> None:
    """Save extraction progress after each thread."""
    cp = {"completed": completed_ids, "results": results}
    path = _checkpoint_path()
    with open(path, "w") as f:
        json.dump(cp, f, default=str)


def _load_checkpoint() -> tuple[set[str], list[dict]] | None:
    """Load checkpoint if it exists. Returns (completed_ids, results) or None."""
    path = _checkpoint_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            cp = json.load(f)
        return set(cp.get("completed", [])), cp.get("results", [])
    except (json.JSONDecodeError, KeyError):
        return None


def _remove_checkpoint() -> None:
    """Delete checkpoint after successful completion."""
    path = _checkpoint_path()
    if os.path.exists(path):
        os.remove(path)


def run_extraction(corpus_path: str = CORPUS_RAW_PATH) -> list[ExtractionResult]:
    """Run extraction over the full thread-based corpus (with checkpointing)."""
    print("=" * 60)
    print("Layer10 Memory Pipeline – Thread-Wise Structured Extraction")
    print("=" * 60)

    with open(corpus_path) as f:
        threads = json.load(f)

    # Detect format: if it's a list of dicts with "messages" key → thread format
    # otherwise legacy flat email list (wrap each email as a single-message thread)
    if threads and isinstance(threads[0], dict) and "messages" in threads[0]:
        print(f"\nProcessing {len(threads)} conversation threads …")
    else:
        print(f"\nLegacy format detected – wrapping {len(threads)} emails as threads …")
        threads = [
            {
                "thread_id": em.get("email_id", f"flat_{i}"),
                "subject": em.get("subject", ""),
                "messages": [em],
                "message_count": 1,
                "participants": [em.get("from", "")],
            }
            for i, em in enumerate(threads)
        ]

    total_msgs = sum(t.get("message_count", len(t.get("messages", [])))
                     for t in threads)
    print(f"  Total messages across threads: {total_msgs}")

    if HAS_OLLAMA:
        print(f"  LLM model: {OLLAMA_MODEL} (GPU accelerated)")
    else:
        print("  Ollama not available – using rule-based extraction only")

    # --- Checkpoint resume ---
    checkpoint = _load_checkpoint()
    completed_ids: set[str] = set()
    result_dicts: list[dict] = []

    if checkpoint:
        completed_ids, result_dicts = checkpoint
        print(f"\n  ♻ Resuming from checkpoint: {len(completed_ids)}/{len(threads)} threads done")
    else:
        print()

    total_entities = 0
    total_claims = 0

    for i, thread in enumerate(threads):
        tid = thread.get("thread_id", f"thread_{i}")
        n_msgs = thread.get("message_count", len(thread.get("messages", [])))
        subj = thread.get("subject", "?")[:55]

        if tid in completed_ids:
            # Already extracted in a previous run – skip
            print(f"  [{i+1}/{len(threads)}] {subj}  (cached ✓)")
            continue

        print(f"  [{i+1}/{len(threads)}] {subj}  ({n_msgs} msgs)")

        t0 = time.time()
        result = extract_thread(thread)
        dt = time.time() - t0

        total_entities += len(result.entities)
        total_claims += len(result.claims)
        print(f"           → {len(result.entities)} entities, "
              f"{len(result.claims)} claims  ({dt:.1f}s)")

        result_dicts.append(result.model_dump())
        completed_ids.add(tid)

        # Save checkpoint after each thread
        _save_checkpoint(list(completed_ids), result_dicts)

    # Reconstruct ExtractionResult objects
    results = [ExtractionResult(**r) for r in result_dicts]

    # Totals (include cached)
    for r in results:
        if r.source_id not in {t.get("thread_id", "") for t in threads
                                if t.get("thread_id", "") not in completed_ids}:
            pass  # already counted above for new threads

    print(f"\nExtraction complete:")
    print(f"  Threads processed : {len(threads)}")
    print(f"  Total raw entities: {sum(len(r.entities) for r in results)}")
    print(f"  Total raw claims  : {sum(len(r.claims) for r in results)}")

    # Save raw extractions
    output = [r.model_dump() for r in results]
    with open(EXTRACTION_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to: {EXTRACTION_PATH}")

    # Clean up checkpoint on success
    _remove_checkpoint()
    print("  Checkpoint cleared ✓")

    return results


if __name__ == "__main__":
    run_extraction()
