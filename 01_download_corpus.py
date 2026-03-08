"""
Step 1 – Download & prepare Enron email corpus as *conversation threads*.

Groups real maildir emails by subject-line (normalised "Re:/Fwd:" prefix)
and In-Reply-To / References headers into threads of
MIN_MESSAGES_PER_THREAD .. MAX_MESSAGES_PER_THREAD messages, then selects
the TARGET_THREAD_COUNT richest threads from executive mailboxes.

Sources (in order of preference):
  1. Already-extracted  data/maildir/   (from a prior download)
  2. Download tarball from CMU  →  extract  →  parse
  3. Synthetic fallback threads (last resort)

Reproduction:
    python 01_download_corpus.py
"""

import json, os, sys, email, tarfile, random, re, hashlib
import urllib.request
from collections import defaultdict
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATA_DIR, CORPUS_RAW_PATH,
    MIN_MESSAGES_PER_THREAD, MAX_MESSAGES_PER_THREAD, TARGET_THREAD_COUNT,
)

ENRON_MAILDIR  = os.path.join(DATA_DIR, "maildir")
ENRON_TAR_PATH = os.path.join(DATA_DIR, "enron_mail_20150507.tar.gz")

ENRON_URLS = [
    "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz",
]

# Executive / high-value mailboxes – produce the richest threads
PRIORITY_MAILBOXES = [
    "lay-k", "skilling-j", "kaminski-v", "kitchen-l",
    "lavorato-j", "buy-r", "beck-s", "shankman-j",
    "delainey-d", "haedicke-m", "hayslett-r", "presto-k",
    "derrick-j", "mann-k", "kean-s", "steffes-j",
    "dasovich-j", "germany-c", "ring-r", "sanders-r",
]


# ═════════════════════════════════════════════════════════════════════════════
# Parsing helpers
# ═════════════════════════════════════════════════════════════════════════════
_RE_PREFIX = re.compile(r"^(?:(?:re|fw|fwd)\s*:\s*)+", re.IGNORECASE)


def normalise_subject(subj: str) -> str:
    """Strip Re:/Fwd: prefixes, collapse whitespace, lower-case."""
    s = _RE_PREFIX.sub("", subj).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()[:120]


def parse_email_file(fp: Path) -> dict | None:
    """Parse a single maildir file into a structured dict."""
    try:
        raw = fp.read_text(encoding="utf-8", errors="replace")
        msg = email.message_from_string(raw)

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode("utf-8", errors="replace")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="replace")

        body = body.strip()
        if len(body) < 60:
            return None

        date_str = msg.get("Date", "")
        try:
            ts = parsedate_to_datetime(date_str).isoformat()
        except Exception:
            ts = date_str

        return {
            "email_id":     hashlib.md5(str(fp).encode()).hexdigest()[:12],
            "message_id":   msg.get("Message-ID", ""),
            "in_reply_to":  msg.get("In-Reply-To", ""),
            "references":   msg.get("References", ""),
            "from":         msg.get("From", "unknown"),
            "to":           msg.get("To", ""),
            "subject":      msg.get("Subject", ""),
            "timestamp":    ts,
            "body":         body[:3000],
        }
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Thread builder
# ═════════════════════════════════════════════════════════════════════════════
def build_threads(emails: list[dict]) -> dict[str, list[dict]]:
    """
    Group emails into conversation threads using:
      1) In-Reply-To / References header chains
      2) Normalised subject fallback
    Returns {thread_key: [sorted messages]}.
    """
    # --- Phase 1: msg-id lookup ---
    by_msgid: dict[str, dict] = {}
    for em in emails:
        mid = em.get("message_id", "").strip()
        if mid:
            by_msgid[mid] = em

    # --- Phase 2: Union-Find on message-id references ---
    parent: dict[str, str] = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for em in emails:
        mid = em.get("message_id", "").strip()
        irt = em.get("in_reply_to", "").strip()
        refs = em.get("references", "").strip().split()
        if mid and irt:
            union(mid, irt)
        for ref in refs:
            ref = ref.strip()
            if mid and ref:
                union(mid, ref)

    # --- Phase 3: group by union-find root or subject fallback ---
    threads: dict[str, list[dict]] = defaultdict(list)
    for em in emails:
        mid = em.get("message_id", "").strip()
        if mid and mid in parent:
            key = find(mid)
        else:
            key = "subj::" + normalise_subject(em.get("subject", ""))
        threads[key].append(em)

    # --- Phase 4: merge tiny subject-alike threads ---
    subj_groups: dict[str, list[str]] = defaultdict(list)
    for key in list(threads.keys()):
        subj_key = normalise_subject(
            threads[key][0].get("subject", "") if threads[key] else ""
        )
        if subj_key:
            subj_groups[subj_key].append(key)

    for subj_key, keys in subj_groups.items():
        if len(keys) > 1:
            primary = keys[0]
            for other in keys[1:]:
                threads[primary].extend(threads.pop(other))

    # --- Phase 5: sort messages within each thread by timestamp ---
    for key in threads:
        threads[key].sort(key=lambda e: e.get("timestamp", ""))

    return dict(threads)


# ═════════════════════════════════════════════════════════════════════════════
# Select best threads
# ═════════════════════════════════════════════════════════════════════════════
def select_threads(
    threads: dict[str, list[dict]],
    min_msgs: int = MIN_MESSAGES_PER_THREAD,
    max_msgs: int = MAX_MESSAGES_PER_THREAD,
    target: int = TARGET_THREAD_COUNT,
) -> list[dict]:
    """
    Pick the best threads:
      • At least min_msgs messages
      • Cap at max_msgs
      • Score by message count, distinct senders, body richness
    Returns a flat list of thread dicts:
        { thread_id, subject, messages: [...], message_count, participants }
    """
    candidates = []
    for key, msgs in threads.items():
        if len(msgs) < min_msgs:
            continue
        msgs = msgs[:max_msgs]

        senders = {m.get("from", "") for m in msgs}
        total_body = sum(len(m.get("body", "")) for m in msgs)
        subj = msgs[0].get("subject", "(no subject)")

        # Score: prefer more messages, more participants, longer bodies
        score = len(msgs) * 3 + len(senders) * 5 + total_body / 500

        # Boost priority mailboxes
        for m in msgs:
            sender = m.get("from", "").lower()
            for mb in PRIORITY_MAILBOXES:
                user = mb.replace("-", ".").split("-")[0]
                if user in sender:
                    score += 20
                    break

        candidates.append({
            "key": key,
            "subject": subj,
            "messages": msgs,
            "score": score,
            "senders": senders,
        })

    candidates.sort(key=lambda c: -c["score"])

    # Pick diverse threads (avoid same subject cluster)
    selected: list[dict] = []
    seen_subjects: set[str] = set()
    for c in candidates:
        nsubj = normalise_subject(c["subject"])
        if nsubj in seen_subjects:
            continue
        seen_subjects.add(nsubj)

        thread_id = hashlib.md5(c["key"].encode()).hexdigest()[:10]
        selected.append({
            "thread_id":     thread_id,
            "subject":       c["subject"],
            "messages":      c["messages"],
            "message_count": len(c["messages"]),
            "participants":  sorted(c["senders"]),
        })
        if len(selected) >= target:
            break

    return selected


# ═════════════════════════════════════════════════════════════════════════════
# Maildir loader
# ═════════════════════════════════════════════════════════════════════════════
def load_from_maildir(maildir: str,
                      max_per_mailbox: int = 500) -> list[dict]:
    """Parse emails from priority mailboxes (capped per box for speed)."""
    all_files: list[Path] = []

    # Phase 1 – priority mailboxes only, cap per mailbox for speed
    for mb in PRIORITY_MAILBOXES:
        mb_path = Path(maildir) / mb
        if mb_path.is_dir():
            # Prefer 'sent' and '_sent_mail' subfolders for richer threads
            sent_dirs = [mb_path / "sent", mb_path / "_sent_mail",
                         mb_path / "sent_items"]
            inbox_dirs = [mb_path / "inbox", mb_path / "all_documents"]
            priority = []
            for sd in sent_dirs + inbox_dirs:
                if sd.is_dir():
                    priority.extend(f for f in sd.rglob("*")
                                    if f.is_file() and f.stat().st_size > 200)

            if len(priority) >= max_per_mailbox:
                all_files.extend(priority[:max_per_mailbox])
            else:
                # Fill up from rest of mailbox
                priority_set = set(priority)
                rest = []
                needed = max_per_mailbox - len(priority)
                for f in mb_path.rglob("*"):
                    if f.is_file() and f.stat().st_size > 200 \
                       and f not in priority_set:
                        rest.append(f)
                        if len(rest) >= needed:
                            break
                all_files.extend(priority + rest)

    total = len(all_files)
    print(f"  Scanning {total:,} files from {len(PRIORITY_MAILBOXES)} "
          f"mailboxes (cap {max_per_mailbox}/box) …")

    emails: list[dict] = []
    errors = 0
    for i, fp in enumerate(all_files):
        if i % 5000 == 0 and i > 0:
            print(f"    … {i:,}/{total:,} parsed ({len(emails):,} valid)")
        try:
            parsed = parse_email_file(fp)
            if parsed:
                emails.append(parsed)
        except Exception:
            errors += 1

    print(f"  Parsed {len(emails):,} valid emails ({errors} errors)")
    return emails


# ═════════════════════════════════════════════════════════════════════════════
# Download & extract tarball
# ═════════════════════════════════════════════════════════════════════════════
def _progress(block_num, block_size, total_size):
    done = block_num * block_size
    if total_size > 0:
        pct = min(done / total_size * 100, 100)
        print(f"\r  ↓ {done/1024/1024:.0f}/{total_size/1024/1024:.0f} MB "
              f"({pct:.0f}%)", end="", flush=True)
    else:
        print(f"\r  ↓ {done/1024/1024:.0f} MB", end="", flush=True)


def download_enron_tarball(dest: str = ENRON_TAR_PATH) -> str | None:
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000_000:
        print(f"  Tarball already cached ({os.path.getsize(dest)/1024/1024:.0f} MB)")
        return dest
    for url in ENRON_URLS:
        try:
            print(f"  Mirror: {url}")
            urllib.request.urlretrieve(url, dest, reporthook=_progress)
            print()
            if os.path.getsize(dest) > 100_000_000:
                return dest
            os.remove(dest)
        except Exception as e:
            print(f"\n  ✗ {e}")
            if os.path.exists(dest):
                os.remove(dest)
    return None


def extract_tarball(tar_path: str, dest_dir: str = DATA_DIR) -> str | None:
    target = os.path.join(dest_dir, "maildir")
    if os.path.isdir(target):
        cnt = sum(1 for _ in Path(target).rglob("*") if _.is_file())
        if cnt > 1000:
            print(f"  Maildir already extracted ({cnt:,} files)")
            return target
    print("  Extracting tar.gz (may take several minutes) …")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=dest_dir, filter="data")
        if os.path.isdir(target):
            return target
    except Exception as e:
        print(f"  ✗ Extraction error: {e}")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Synthetic fallback threads
# ═════════════════════════════════════════════════════════════════════════════
def create_synthetic_threads() -> list[dict]:
    """
    15 synthetic Enron-style conversation threads, 5-8 messages each.
    Used only when real maildir is unavailable.
    """
    print("  ⚠  Generating synthetic Enron-style threads (download unavailable)")

    threads = []

    # ── Thread 1: Q3 Earnings Strategy ──────────────────────────────
    threads.append({"thread_id":"t01","subject":"Q3 Earnings Strategy","participants":["kenneth.lay@enron.com","jeff.skilling@enron.com","andrew.fastow@enron.com","rick.causey@enron.com","greg.whalley@enron.com"],"messages":[
        {"email_id":"t01_01","message_id":"<t01_01@enron.com>","in_reply_to":"","references":"","from":"kenneth.lay@enron.com","to":"jeff.skilling@enron.com","subject":"Q3 Earnings Strategy","timestamp":"2001-03-15T09:30:00","body":"Jeff,\n\nWe need to discuss Q3 earnings projections. The California energy market is creating significant revenue opportunities but also major regulatory exposure. Please prepare an overview of our current positions in the Western power markets.\n\nI also need an update on the LJM partnership structures and how they're being used to manage our balance sheet. The board wants a full presentation next month.\n\nKen"},
        {"email_id":"t01_02","message_id":"<t01_02@enron.com>","in_reply_to":"<t01_01@enron.com>","references":"<t01_01@enron.com>","from":"jeff.skilling@enron.com","to":"kenneth.lay@enron.com","subject":"Re: Q3 Earnings Strategy","timestamp":"2001-03-15T14:20:00","body":"Ken,\n\nQ3 is looking strong. Revenue from the trading floor is up 45% YoY. Our mark-to-market accounting allows us to book projected future profits now, which is giving us excellent numbers.\n\nFor California, Tim Belden's team is generating significant profits from the Western desk. I'll have Vince Kaminski's research group run the risk models.\n\nAndy Fastow can brief you on LJM. The structures are complex but they're keeping our debt ratios looking clean.\n\nJeff"},
        {"email_id":"t01_03","message_id":"<t01_03@enron.com>","in_reply_to":"<t01_02@enron.com>","references":"<t01_01@enron.com> <t01_02@enron.com>","from":"andrew.fastow@enron.com","to":"kenneth.lay@enron.com","subject":"Re: Q3 Earnings Strategy","timestamp":"2001-03-16T08:15:00","body":"Ken,\n\nLJM2 partnership update for Q3 planning:\n- Current assets under management: $394M\n- 3 new hedging transactions completed this quarter\n- Raptor vehicles are performing as designed\n- Off-balance-sheet treatment confirmed by Arthur Andersen\n\nBen Glisan has raised some concerns about the accounting treatment of certain SPEs. I've asked Rick Causey to review the specifics with Andersen.\n\nAndy Fastow\nCFO"},
        {"email_id":"t01_04","message_id":"<t01_04@enron.com>","in_reply_to":"<t01_03@enron.com>","references":"<t01_01@enron.com> <t01_02@enron.com> <t01_03@enron.com>","from":"rick.causey@enron.com","to":"kenneth.lay@enron.com","subject":"Re: Q3 Earnings Strategy","timestamp":"2001-03-17T10:30:00","body":"Ken,\n\nI've reviewed the structures with Arthur Andersen. They are comfortable with the current accounting treatment. However, I want to flag that the 3% outside equity requirement for several SPEs is being met with a very narrow margin. If Enron stock drops significantly, some of these structures could trigger credit provisions.\n\nRick Causey\nChief Accounting Officer"},
        {"email_id":"t01_05","message_id":"<t01_05@enron.com>","in_reply_to":"<t01_04@enron.com>","references":"<t01_01@enron.com> <t01_02@enron.com> <t01_03@enron.com> <t01_04@enron.com>","from":"greg.whalley@enron.com","to":"kenneth.lay@enron.com","subject":"Re: Q3 Earnings Strategy","timestamp":"2001-03-18T09:00:00","body":"Ken,\n\nAdding my perspective on the trading side. EnronOnline volumes are hitting new records - $2.8B in notional value this month alone. Our dominant position in the electronic energy trading market is a genuine competitive advantage.\n\nHowever, I share Vince's concern about concentration risk in California. We should consider reducing exposure there by at least 20%.\n\nGreg Whalley\nPresident & COO"},
    ],"message_count":5})

    # ── Thread 2: California Energy Crisis ──────────────────────────
    threads.append({"thread_id":"t02","subject":"California Energy Crisis - Risk Assessment","participants":["vince.kaminski@enron.com","jeff.skilling@enron.com","tim.belden@enron.com","rick.buy@enron.com","john.lavorato@enron.com"],"messages":[
        {"email_id":"t02_01","message_id":"<t02_01@enron.com>","in_reply_to":"","references":"","from":"vince.kaminski@enron.com","to":"jeff.skilling@enron.com","subject":"California Energy Crisis - Risk Assessment","timestamp":"2001-03-18T14:20:00","body":"Jeff,\n\nOur Value-at-Risk in Western power markets has increased 340% over the past quarter. The regulatory risk in California is substantial - FERC may intervene at any time.\n\nTim Belden's trading strategies are generating enormous profits but carry significant concentration risk. The 'Death Star' and 'Fat Boy' strategies exploit grid congestion rules. If regulators investigate, exposure could be severe.\n\nI strongly recommend we reduce California exposure by at least 25%.\n\nVince Kaminski\nManaging Director, Research"},
        {"email_id":"t02_02","message_id":"<t02_02@enron.com>","in_reply_to":"<t02_01@enron.com>","references":"<t02_01@enron.com>","from":"jeff.skilling@enron.com","to":"vince.kaminski@enron.com","subject":"Re: California Energy Crisis - Risk Assessment","timestamp":"2001-03-19T08:45:00","body":"Vince,\n\nI appreciate the analysis but the California desk is our single highest-margin operation right now. We can't pull back when the market is this favorable. Tim's strategies are legal - they work within the existing market rules.\n\nKeep monitoring but I don't want to reduce positions. The profits from the Western desk are critical for our Q3 numbers.\n\nJeff"},
        {"email_id":"t02_03","message_id":"<t02_03@enron.com>","in_reply_to":"<t02_02@enron.com>","references":"<t02_01@enron.com> <t02_02@enron.com>","from":"tim.belden@enron.com","to":"jeff.skilling@enron.com","subject":"Re: California Energy Crisis - Risk Assessment","timestamp":"2001-03-20T07:30:00","body":"Jeff,\n\nWestern desk update: Our active strategies for the California market continue to perform well.\n- 'Death Star': simultaneous scheduling creating congestion revenue. $45M YTD.\n- 'Fat Boy': overscheduling load for excess power. $35M YTD.\n- 'Get Shorty': selling ancillary services short. $20M YTD.\n\nTotal desk P&L: $120M YTD. We're keeping documentation minimal as discussed.\n\nTim Belden\nHead of Western Trading"},
        {"email_id":"t02_04","message_id":"<t02_04@enron.com>","in_reply_to":"<t02_03@enron.com>","references":"<t02_01@enron.com> <t02_02@enron.com> <t02_03@enron.com>","from":"rick.buy@enron.com","to":"vince.kaminski@enron.com","subject":"Re: California Energy Crisis - Risk Assessment","timestamp":"2001-03-21T11:15:00","body":"Vince,\n\nI've reviewed your risk report. You're right that the concentration is concerning. Our total California exposure is now $1.2B with a VaR of $180M. That's well outside our normal risk appetite.\n\nI've escalated to the Risk Management Committee but Jeff overruled the recommendation to reduce positions.\n\nRick Buy\nChief Risk Officer"},
        {"email_id":"t02_05","message_id":"<t02_05@enron.com>","in_reply_to":"<t02_04@enron.com>","references":"<t02_01@enron.com> <t02_02@enron.com> <t02_03@enron.com> <t02_04@enron.com>","from":"john.lavorato@enron.com","to":"jeff.skilling@enron.com","subject":"Re: California Energy Crisis - Risk Assessment","timestamp":"2001-03-22T16:00:00","body":"Jeff,\n\nFYI - FERC has announced they're opening a formal investigation into Western power markets. This could mean subpoenas for our trading records. Tim's team needs to know about document retention policies.\n\nAlso, the California PUC is pushing for price caps. If caps are imposed, several of our forward positions could lose $200M+.\n\nJohn Lavorato\nCEO Enron Americas"},
    ],"message_count":5})

    # ── Thread 3: Sherron Watkins Whistleblower ─────────────────────
    threads.append({"thread_id":"t03","subject":"Accounting Concerns - CONFIDENTIAL","participants":["sherron.watkins@enron.com","kenneth.lay@enron.com","james.derrick@enron.com","vinson.elkins@external.com","richard.causey@enron.com"],"messages":[
        {"email_id":"t03_01","message_id":"<t03_01@enron.com>","in_reply_to":"","references":"","from":"sherron.watkins@enron.com","to":"kenneth.lay@enron.com","subject":"Accounting Concerns - CONFIDENTIAL","timestamp":"2001-08-15T10:30:00","body":"Dear Mr. Lay,\n\nI am writing because I am incredibly nervous that we will implode in a wave of accounting scandals. I have been trying to gauge whether or not this is a serious concern but the more I look the worse it gets.\n\nThe Raptor vehicles are hedging Enron investments with Enron's own stock. This is circular - if the stock drops, the hedges fail precisely when we need them.\n\nLJM partnerships have fundamental conflicts of interest with Andy Fastow managing both sides of transactions. Several SPE structures may not meet the 3% outside equity requirement.\n\nI beg you to examine these issues carefully before the company faces irreversible damage.\n\nSherron Watkins\nVP Corporate Development"},
        {"email_id":"t03_02","message_id":"<t03_02@enron.com>","in_reply_to":"<t03_01@enron.com>","references":"<t03_01@enron.com>","from":"kenneth.lay@enron.com","to":"james.derrick@enron.com","subject":"Fwd: Accounting Concerns - CONFIDENTIAL","timestamp":"2001-08-16T09:00:00","body":"Jim,\n\nPlease review the attached letter from Sherron Watkins. She's raising concerns about our SPE structures and Andy's role in LJM. I want Vinson & Elkins to conduct an independent review.\n\nKeep this confidential. We need to understand our exposure before the next earnings call.\n\nKen"},
        {"email_id":"t03_03","message_id":"<t03_03@enron.com>","in_reply_to":"<t03_02@enron.com>","references":"<t03_01@enron.com> <t03_02@enron.com>","from":"james.derrick@enron.com","to":"kenneth.lay@enron.com","subject":"Re: Accounting Concerns - CONFIDENTIAL","timestamp":"2001-08-20T15:30:00","body":"Ken,\n\nVinson & Elkins has begun their review. Initial assessment: the structures are technically compliant but the optics are terrible. Their preliminary view is that the Raptor vehicles' dependence on Enron stock is a systemic risk.\n\nThey've also flagged that Arthur Andersen approved all of these structures, so any investigation would implicate our auditors as well.\n\nRecommendation: do NOT fire Sherron Watkins - that would create a retaliation claim.\n\nJim Derrick\nGeneral Counsel"},
        {"email_id":"t03_04","message_id":"<t03_04@enron.com>","in_reply_to":"<t03_03@enron.com>","references":"<t03_01@enron.com> <t03_02@enron.com> <t03_03@enron.com>","from":"richard.causey@enron.com","to":"kenneth.lay@enron.com","subject":"Re: Accounting Concerns - CONFIDENTIAL","timestamp":"2001-08-22T11:00:00","body":"Ken,\n\nI've re-examined the Raptor structures in light of Ms. Watkins' concerns. While Arthur Andersen has signed off on the accounting treatment, the reality is:\n\n- Raptor I: hedging $500M in merchant investments\n- Raptor II: hedging $300M in broadband assets\n- Raptor III: hedging $800M combined\n- Total credit support from Enron stock: $1.2B\n\nIf ENE drops below $40, credit triggers activate and we'd need to issue stock or take charges.\n\nRick Causey\nCAO"},
        {"email_id":"t03_05","message_id":"<t03_05@enron.com>","in_reply_to":"<t03_04@enron.com>","references":"<t03_01@enron.com> <t03_02@enron.com> <t03_03@enron.com> <t03_04@enron.com>","from":"kenneth.lay@enron.com","to":"sherron.watkins@enron.com","subject":"Re: Accounting Concerns - CONFIDENTIAL","timestamp":"2001-08-28T09:30:00","body":"Sherron,\n\nThank you for bringing these concerns to my attention. I've asked Vinson & Elkins to conduct a thorough review and our accounting team is re-examining the structures.\n\nI take these matters very seriously. We will address any issues identified through proper channels.\n\nKen Lay\nChairman & CEO"},
    ],"message_count":5})

    # ── Thread 4: EnronOnline Platform ──────────────────────────────
    threads.append({"thread_id":"t04","subject":"EnronOnline Platform Performance","participants":["louise.kitchen@enron.com","greg.whalley@enron.com","jeff.shankman@enron.com","sally.beck@enron.com","john.lavorato@enron.com"],"messages":[
        {"email_id":"t04_01","message_id":"<t04_01@enron.com>","in_reply_to":"","references":"","from":"louise.kitchen@enron.com","to":"greg.whalley@enron.com","subject":"EnronOnline Platform Performance","timestamp":"2001-04-10T16:45:00","body":"Greg,\n\nEnronOnline monthly update:\n- Daily transaction volume: 6,200 trades\n- Total notional value: $2.8B this month\n- Platform uptime: 99.97%\n- New counterparties onboarded: 45\n- Active products: 1,800+\n\nWe are now the dominant electronic trading platform for energy commodities. Competitors ICE and TradeSpark are far behind.\n\nLouise Kitchen\nPresident, Enron Online"},
        {"email_id":"t04_02","message_id":"<t04_02@enron.com>","in_reply_to":"<t04_01@enron.com>","references":"<t04_01@enron.com>","from":"greg.whalley@enron.com","to":"louise.kitchen@enron.com","subject":"Re: EnronOnline Platform Performance","timestamp":"2001-04-11T09:00:00","body":"Louise,\n\nExcellent numbers. This is exactly what we need for the investor presentation. A few questions:\n\n1. What's our market share in natural gas trading?\n2. Can we expand to bandwidth and weather derivatives?\n3. What's the credit exposure per counterparty?\n\nThe board loves the technology story - it differentiates us from traditional utilities.\n\nGreg"},
        {"email_id":"t04_03","message_id":"<t04_03@enron.com>","in_reply_to":"<t04_02@enron.com>","references":"<t04_01@enron.com> <t04_02@enron.com>","from":"jeff.shankman@enron.com","to":"greg.whalley@enron.com","subject":"Re: EnronOnline Platform Performance","timestamp":"2001-04-12T14:20:00","body":"Greg,\n\nOn the Global Markets side, EOL is transforming how we do business internationally. We've launched natural gas products in the UK and power products across continental Europe.\n\nKey concern: every trade on EOL has Enron as the counterparty. If our credit rating drops, counterparties will stop trading with us immediately. This is a single point of failure.\n\nJeff Shankman\nHead of Global Markets"},
        {"email_id":"t04_04","message_id":"<t04_04@enron.com>","in_reply_to":"<t04_03@enron.com>","references":"<t04_01@enron.com> <t04_02@enron.com> <t04_03@enron.com>","from":"sally.beck@enron.com","to":"louise.kitchen@enron.com","subject":"Re: EnronOnline Platform Performance","timestamp":"2001-04-13T10:30:00","body":"Louise,\n\nRisk controls update for EOL: Our back-office systems are struggling to keep up with the volume. We're processing 6000+ trades daily but settlement and confirmation are lagging by 2-3 days.\n\nI need additional headcount and system upgrades. Current error rate is 1.2% which is unacceptable for these volumes.\n\nSally Beck\nCOO, Enron Americas"},
        {"email_id":"t04_05","message_id":"<t04_05@enron.com>","in_reply_to":"<t04_04@enron.com>","references":"<t04_01@enron.com> <t04_02@enron.com> <t04_03@enron.com> <t04_04@enron.com>","from":"john.lavorato@enron.com","to":"greg.whalley@enron.com","subject":"Re: EnronOnline Platform Performance","timestamp":"2001-04-14T08:00:00","body":"Greg,\n\nJeff Shankman raises a critical point about counterparty risk. Our entire trading business depends on maintaining investment-grade credit. If we lose that, EOL shuts down overnight.\n\nWe should develop contingency plans. Perhaps a joint venture structure or clearing house model to reduce single-counterparty concentration.\n\nJohn Lavorato"},
    ],"message_count":5})

    # ── Thread 5: Raptor Vehicle Crisis ─────────────────────────────
    threads.append({"thread_id":"t05","subject":"Raptor Vehicle Status - CRITICAL","participants":["ben.glisan@enron.com","andrew.fastow@enron.com","rick.causey@enron.com","jeff.mcmahon@enron.com","david.duncan@andersen.com"],"messages":[
        {"email_id":"t05_01","message_id":"<t05_01@enron.com>","in_reply_to":"","references":"","from":"ben.glisan@enron.com","to":"andrew.fastow@enron.com","subject":"Raptor Vehicle Status - CRITICAL","timestamp":"2001-09-15T07:00:00","body":"Andy,\n\nThe Raptor vehicles are in serious trouble. Current status:\n- Raptor I: underwater by $130M, hedging New Power Company\n- Raptor II: underwater by $95M, hedging broadband assets\n- Raptor III (Talon): underwater by $475M, hedging merchant portfolio\n- Total exposure: approximately $700M\n\nThe credit triggers tied to Enron's stock price have been breached. Arthur Andersen is asking increasingly difficult questions about the capitalization requirements.\n\nBen Glisan\nTreasurer"},
        {"email_id":"t05_02","message_id":"<t05_02@enron.com>","in_reply_to":"<t05_01@enron.com>","references":"<t05_01@enron.com>","from":"andrew.fastow@enron.com","to":"ben.glisan@enron.com","subject":"Re: Raptor Vehicle Status - CRITICAL","timestamp":"2001-09-15T11:30:00","body":"Ben,\n\nWe need to restructure the Raptors immediately. Options:\n1. Inject additional Enron stock to meet capitalization - dilutive\n2. Unwind the hedges and take the losses - $700M charge\n3. Cross-collateralize the Raptors - buys time but increases systemic risk\n\nI'm recommending option 3 to buy time. We can restructure in Q4.\n\nDo NOT share this analysis outside of Treasury and LJM teams.\n\nAndy"},
        {"email_id":"t05_03","message_id":"<t05_03@enron.com>","in_reply_to":"<t05_02@enron.com>","references":"<t05_01@enron.com> <t05_02@enron.com>","from":"rick.causey@enron.com","to":"andrew.fastow@enron.com","subject":"Re: Raptor Vehicle Status - CRITICAL","timestamp":"2001-09-16T09:00:00","body":"Andy,\n\nI cannot support option 3. Cross-collateralizing creates additional undisclosed risk that Arthur Andersen will flag. We've already received a management representation letter from David Duncan asking about off-balance-sheet exposure.\n\nMy recommendation: disclose the Raptor losses in Q3 earnings. Taking the hit now is better than having it discovered later. A $700M restatement voluntarily is survivable. Being caught is not.\n\nRick Causey\nCAO"},
        {"email_id":"t05_04","message_id":"<t05_04@enron.com>","in_reply_to":"<t05_03@enron.com>","references":"<t05_01@enron.com> <t05_02@enron.com> <t05_03@enron.com>","from":"jeff.mcmahon@enron.com","to":"andrew.fastow@enron.com","subject":"Re: Raptor Vehicle Status - CRITICAL","timestamp":"2001-09-17T14:00:00","body":"Andy,\n\nAs Treasurer (prior to Ben), I want to go on record that I raised concerns about the LJM conflict of interest directly with Jeff Skilling in March 2000. The dual-role structure was problematic from inception.\n\nThe Raptor unwinding will require board notification. Audit committee needs to be briefed before any public disclosure.\n\nJeff McMahon"},
        {"email_id":"t05_05","message_id":"<t05_05@enron.com>","in_reply_to":"<t05_04@enron.com>","references":"<t05_01@enron.com> <t05_02@enron.com> <t05_03@enron.com> <t05_04@enron.com>","from":"ben.glisan@enron.com","to":"andrew.fastow@enron.com","subject":"Re: Raptor Vehicle Status - CRITICAL","timestamp":"2001-09-20T08:00:00","body":"Andy,\n\nUpdate: David Duncan from Arthur Andersen has requested a meeting about the Raptor capitalization. He's bringing their risk consulting team.\n\nAlso, the SEC has sent preliminary inquiries about our SPE disclosures in the 10-K. This is getting serious.\n\nWe need Ken Lay and the board involved now.\n\nBen Glisan\nTreasurer"},
    ],"message_count":5})

    # ── Thread 6: Skilling Resignation ──────────────────────────────
    threads.append({"thread_id":"t06","subject":"CEO Resignation - Internal","participants":["jeff.skilling@enron.com","kenneth.lay@enron.com","mark.koenig@enron.com","greg.whalley@enron.com","mark.frevert@enron.com"],"messages":[
        {"email_id":"t06_01","message_id":"<t06_01@enron.com>","in_reply_to":"","references":"","from":"jeff.skilling@enron.com","to":"kenneth.lay@enron.com","subject":"Personal Decision - CONFIDENTIAL","timestamp":"2001-08-13T20:00:00","body":"Ken,\n\nI have decided to resign as CEO effective tomorrow. This is a personal decision. I want you to know that I believe Enron is in the strongest shape it has ever been.\n\nI will announce to all employees tomorrow at noon. You should be prepared to resume the CEO role.\n\nI'm sorry for the short notice.\n\nJeff"},
        {"email_id":"t06_02","message_id":"<t06_02@enron.com>","in_reply_to":"<t06_01@enron.com>","references":"<t06_01@enron.com>","from":"kenneth.lay@enron.com","to":"jeff.skilling@enron.com","subject":"Re: Personal Decision - CONFIDENTIAL","timestamp":"2001-08-14T07:00:00","body":"Jeff,\n\nI'm shocked by this decision. The timing couldn't be worse - we have ongoing regulatory inquiries, the stock has dropped from $80 to $42, and analysts are already questioning our earnings quality.\n\nI'll take over as CEO but I need a full briefing on everything. Every trading position, every SPE, every off-balance-sheet arrangement.\n\nKen"},
        {"email_id":"t06_03","message_id":"<t06_03@enron.com>","in_reply_to":"<t06_02@enron.com>","references":"<t06_01@enron.com> <t06_02@enron.com>","from":"jeff.skilling@enron.com","to":"kenneth.lay@enron.com","subject":"Personal Announcement","timestamp":"2001-08-14T12:00:00","body":"To all Enron employees,\n\nI am resigning as CEO of Enron effective today, August 14, 2001. This is a personal decision and has nothing to do with the company's financial condition or any pending regulatory matter. Enron is in the strongest shape it has ever been.\n\nKen Lay will resume the role of Chairman and CEO. I have full confidence in Ken and the leadership team.\n\nThank you for making Enron the world's leading energy company.\n\nJeff Skilling"},
        {"email_id":"t06_04","message_id":"<t06_04@enron.com>","in_reply_to":"<t06_03@enron.com>","references":"<t06_01@enron.com> <t06_02@enron.com> <t06_03@enron.com>","from":"mark.koenig@enron.com","to":"kenneth.lay@enron.com","subject":"Re: Personal Announcement","timestamp":"2001-08-14T14:00:00","body":"Ken,\n\nThe market reaction is severe. ENE stock dropped 6% on the news. Analysts are calling this a red flag. We need an immediate investor relations response.\n\nI recommend an analyst conference call within 48 hours. Key messages: continuity of strategy, no financial issues, Lay's deep knowledge of the business.\n\nMark Koenig\nHead of Investor Relations"},
        {"email_id":"t06_05","message_id":"<t06_05@enron.com>","in_reply_to":"<t06_04@enron.com>","references":"<t06_01@enron.com> <t06_02@enron.com> <t06_03@enron.com> <t06_04@enron.com>","from":"greg.whalley@enron.com","to":"kenneth.lay@enron.com","subject":"Re: Personal Announcement","timestamp":"2001-08-14T15:30:00","body":"Ken,\n\nInternal reaction is mixed. The trading floor is nervous. Some senior people are asking questions about whether Jeff's departure is related to the accounting structures.\n\nMark Frevert and I can handle day-to-day operations. But you need to get ahead of the narrative - Wall Street is going to dig into everything now.\n\nGreg Whalley\nPresident & COO"},
    ],"message_count":5})

    # ── Thread 7: Dynegy Merger ─────────────────────────────────────
    threads.append({"thread_id":"t07","subject":"Dynegy Merger Negotiations","participants":["kenneth.lay@enron.com","greg.whalley@enron.com","mark.frevert@enron.com","steve.bergstrom@dynegy.com","chuck.watson@dynegy.com"],"messages":[
        {"email_id":"t07_01","message_id":"<t07_01@enron.com>","in_reply_to":"","references":"","from":"greg.whalley@enron.com","to":"kenneth.lay@enron.com","subject":"Dynegy Merger Negotiations","timestamp":"2001-11-08T06:00:00","body":"Ken,\n\nDynegy has offered to acquire Enron in a stock-for-stock deal valued at approximately $9B. Chuck Watson is willing to move fast. Key terms:\n- Exchange ratio: 0.2685 Dynegy shares per ENE share\n- $1.5B cash infusion immediately from ChevronTexaco\n- Dynegy assumes $13B in debt\n- Enron's pipeline assets (Northern Natural Gas) as collateral\n\nThis may be our best option given the current situation.\n\nGreg"},
        {"email_id":"t07_02","message_id":"<t07_02@enron.com>","in_reply_to":"<t07_01@enron.com>","references":"<t07_01@enron.com>","from":"kenneth.lay@enron.com","to":"greg.whalley@enron.com","subject":"Re: Dynegy Merger Negotiations","timestamp":"2001-11-09T08:00:00","body":"Greg,\n\nThe terms are painful but we may have no choice. Our credit has been downgraded to BBB and another notch puts us at junk. That would trigger $3.9B in payment obligations and collapse EnronOnline.\n\nProceed with negotiations but push for a better ratio. And the $1.5B from Chevron needs to come in immediately - we need it for liquidity.\n\nKen"},
        {"email_id":"t07_03","message_id":"<t07_03@enron.com>","in_reply_to":"<t07_02@enron.com>","references":"<t07_01@enron.com> <t07_02@enron.com>","from":"mark.frevert@enron.com","to":"kenneth.lay@enron.com","subject":"Re: Dynegy Merger Negotiations","timestamp":"2001-11-15T10:00:00","body":"Ken,\n\nDue diligence update: Dynegy's team is finding more off-balance-sheet obligations than we disclosed. They've flagged:\n- Additional $690M in obligations from Marlin and Osprey trusts\n- The Whitewing SPE structure they weren't told about\n- Potential regulatory liabilities from California trading\n\nChuck Watson called me directly expressing concern. The deal is getting shaky.\n\nMark Frevert\nVice Chairman"},
        {"email_id":"t07_04","message_id":"<t07_04@enron.com>","in_reply_to":"<t07_03@enron.com>","references":"<t07_01@enron.com> <t07_02@enron.com> <t07_03@enron.com>","from":"greg.whalley@enron.com","to":"kenneth.lay@enron.com","subject":"Re: Dynegy Merger Negotiations - DEAL STATUS","timestamp":"2001-11-26T14:00:00","body":"Ken,\n\nBad news. Moody's and S&P have both downgraded us to junk (Ba2/BB). This triggered $3.9B in immediate payment obligations. Our cash position is critical.\n\nDynegy is now demanding we renegotiate the merger ratio to 0.15 from 0.2685, effectively cutting the deal value by 44%.\n\nChevron is threatening to pull the $1.5B bridge loan.\n\nGreg"},
        {"email_id":"t07_05","message_id":"<t07_05@enron.com>","in_reply_to":"<t07_04@enron.com>","references":"<t07_01@enron.com> <t07_02@enron.com> <t07_03@enron.com> <t07_04@enron.com>","from":"greg.whalley@enron.com","to":"kenneth.lay@enron.com","subject":"Dynegy Merger - DEAL IS OFF","timestamp":"2001-11-28T06:00:00","body":"Ken,\n\nDynegy has officially terminated the merger agreement effective immediately. Stated reasons:\n1. Discovery of additional off-balance-sheet obligations not previously disclosed\n2. Credit downgrade to junk status\n3. SEC investigation expanding in scope\n4. Material deterioration in Enron's business\n\nWe have no choice but to file Chapter 11. Legal is preparing the filing for December 2. $63B in assets - this will be the largest bankruptcy in US history.\n\nGreg Whalley\nPresident & COO"},
    ],"message_count":5})

    # ── Thread 8: Arthur Andersen Document Destruction ──────────────
    threads.append({"thread_id":"t08","subject":"Document Retention Policy - URGENT","participants":["nancy.temple@andersen.com","david.duncan@andersen.com","rick.causey@enron.com","james.derrick@enron.com","thomas.bauer@andersen.com"],"messages":[
        {"email_id":"t08_01","message_id":"<t08_01@enron.com>","in_reply_to":"","references":"","from":"nancy.temple@andersen.com","to":"david.duncan@andersen.com","subject":"Document Retention Policy - URGENT","timestamp":"2001-10-12T09:00:00","body":"David,\n\nGiven the current situation with Enron and the SEC inquiry, I want to remind you of our firm's document retention policy. Please ensure that all engagement team members are complying with the policy.\n\nSpecifically, all working papers and correspondence should be reviewed. Items that are not part of the permanent audit file should be handled per our standard retention schedule.\n\nPlease confirm receipt.\n\nNancy Temple\nIn-House Counsel, Arthur Andersen"},
        {"email_id":"t08_02","message_id":"<t08_02@enron.com>","in_reply_to":"<t08_01@enron.com>","references":"<t08_01@enron.com>","from":"david.duncan@andersen.com","to":"nancy.temple@andersen.com","subject":"Re: Document Retention Policy - URGENT","timestamp":"2001-10-12T14:00:00","body":"Nancy,\n\nUnderstood. I have instructed the Enron engagement team to conduct an immediate review of all working papers. Non-essential documents, drafts, and preliminary analyses are being processed per the retention policy.\n\nThe team has been working around the clock on this.\n\nDavid Duncan\nLead Engagement Partner, Enron Account"},
        {"email_id":"t08_03","message_id":"<t08_03@enron.com>","in_reply_to":"<t08_02@enron.com>","references":"<t08_01@enron.com> <t08_02@enron.com>","from":"thomas.bauer@andersen.com","to":"david.duncan@andersen.com","subject":"Re: Document Retention Policy - URGENT","timestamp":"2001-10-15T11:30:00","body":"David,\n\nThe team has shredded approximately 1 ton of Enron-related documents over the past 3 days. We've also deleted a significant volume of electronic files including emails and working paper drafts.\n\nI want to confirm this is appropriate given the SEC situation. Some team members are uncomfortable.\n\nThomas Bauer\nSenior Manager, Andersen Houston"},
        {"email_id":"t08_04","message_id":"<t08_04@enron.com>","in_reply_to":"<t08_03@enron.com>","references":"<t08_01@enron.com> <t08_02@enron.com> <t08_03@enron.com>","from":"james.derrick@enron.com","to":"rick.causey@enron.com","subject":"Re: Document Retention Policy - URGENT","timestamp":"2001-10-16T08:00:00","body":"Rick,\n\nI've been informed that Arthur Andersen's Houston office is conducting a large-scale document destruction. Given that we have received SEC inquiries, this is extremely concerning.\n\nWe need to issue an immediate litigation hold on all Enron documents, electronic and physical. Any continued destruction could constitute obstruction.\n\nJim Derrick\nGeneral Counsel"},
        {"email_id":"t08_05","message_id":"<t08_05@enron.com>","in_reply_to":"<t08_04@enron.com>","references":"<t08_01@enron.com> <t08_02@enron.com> <t08_03@enron.com> <t08_04@enron.com>","from":"rick.causey@enron.com","to":"james.derrick@enron.com","subject":"Re: Document Retention Policy - URGENT","timestamp":"2001-10-17T10:00:00","body":"Jim,\n\nAgreed. I've contacted David Duncan directly and demanded that all document destruction cease immediately. Arthur Andersen's actions could expose both their firm and Enron to obstruction charges.\n\nI'm issuing a company-wide litigation hold today. All employees must preserve all documents, emails, and electronic files related to any financial transaction, SPE, or trading activity.\n\nRick Causey\nCAO"},
    ],"message_count":5})

    # ── Thread 9: Broadband Venture Failure ─────────────────────────
    threads.append({"thread_id":"t09","subject":"Enron Broadband - Strategic Review","participants":["kenneth.rice@enron.com","jeff.skilling@enron.com","kevin.hannon@enron.com","joe.hirko@enron.com","scott.yeager@enron.com"],"messages":[
        {"email_id":"t09_01","message_id":"<t09_01@enron.com>","in_reply_to":"","references":"","from":"kenneth.rice@enron.com","to":"jeff.skilling@enron.com","subject":"Enron Broadband - Strategic Review","timestamp":"2001-01-20T14:30:00","body":"Jeff,\n\nEnron Broadband Services Q4 review:\n- Revenue: $408M (vs $30B projection from analyst day)\n- Losses: $137M operating loss\n- Blockbuster VOD deal: technology not ready, streaming quality poor\n- Fiber network utilization: only 5% of capacity\n\nThe analyst day projections from January 2000 were far too aggressive. We told Wall Street this would be a $30B business by 2005 and we're nowhere close.\n\nKen Rice\nCEO, Enron Broadband"},
        {"email_id":"t09_02","message_id":"<t09_02@enron.com>","in_reply_to":"<t09_01@enron.com>","references":"<t09_01@enron.com>","from":"jeff.skilling@enron.com","to":"kenneth.rice@enron.com","subject":"Re: Enron Broadband - Strategic Review","timestamp":"2001-01-21T09:00:00","body":"Ken,\n\nThe Blockbuster deal was supposed to be the proof point. We booked $110M in mark-to-market profits on that deal alone before we even delivered a working service. The market will crucify us if this unravels.\n\nWe need to pivot the narrative. Focus on bandwidth trading - that's where the margin is. The content delivery story isn't working.\n\nJeff"},
        {"email_id":"t09_03","message_id":"<t09_03@enron.com>","in_reply_to":"<t09_02@enron.com>","references":"<t09_01@enron.com> <t09_02@enron.com>","from":"kevin.hannon@enron.com","to":"jeff.skilling@enron.com","subject":"Re: Enron Broadband - Strategic Review","timestamp":"2001-02-15T11:00:00","body":"Jeff,\n\nBandwidth trading update: daily volumes are improving but we're still the only real market maker. Without other participants, this is more like mark-to-model than mark-to-market.\n\nThe Portland fiber network team has reduced the engineering team from 500 to 200. Morale is terrible.\n\nKevin Hannon\nCOO, Enron Broadband"},
        {"email_id":"t09_04","message_id":"<t09_04@enron.com>","in_reply_to":"<t09_03@enron.com>","references":"<t09_01@enron.com> <t09_02@enron.com> <t09_03@enron.com>","from":"joe.hirko@enron.com","to":"jeff.skilling@enron.com","subject":"Re: Enron Broadband - Strategic Review","timestamp":"2001-03-10T08:00:00","body":"Jeff,\n\nI need to flag a serious accounting concern. We booked $110M from the Blockbuster deal using mark-to-market accounting on a 20-year forward projection. The deal has effectively collapsed - Blockbuster pulled out.\n\nWe haven't reversed the gains. If we do, it's a $110M charge that will raise questions about our entire MTM methodology.\n\nJoe Hirko\nCo-CEO, Enron Broadband"},
        {"email_id":"t09_05","message_id":"<t09_05@enron.com>","in_reply_to":"<t09_04@enron.com>","references":"<t09_01@enron.com> <t09_02@enron.com> <t09_03@enron.com> <t09_04@enron.com>","from":"scott.yeager@enron.com","to":"jeff.skilling@enron.com","subject":"Re: Enron Broadband - Strategic Review","timestamp":"2001-04-05T16:00:00","body":"Jeff,\n\nThe broadband trading floor is not sustainable. We have:\n- $500M invested in fiber infrastructure with 5% utilization\n- $110M in phantom Blockbuster profits that need reversal\n- An engineering team that's been cut by 60%\n\nRecommendation: shut down the broadband trading operation and focus on core energy business. This is burning cash.\n\nScott Yeager\nSVP Strategy"},
    ],"message_count":5})

    # ── Thread 10: Employee Pension Fund ─────────────────────────────
    threads.append({"thread_id":"t10","subject":"401(k) Plan - Enron Stock Lockdown","participants":["cindy.olson@enron.com","kenneth.lay@enron.com","employee.rep@enron.com","legal.dept@enron.com","jeff.skilling@enron.com"],"messages":[
        {"email_id":"t10_01","message_id":"<t10_01@enron.com>","in_reply_to":"","references":"","from":"cindy.olson@enron.com","to":"kenneth.lay@enron.com","subject":"401(k) Plan - Enron Stock Lockdown","timestamp":"2001-10-26T09:00:00","body":"Ken,\n\nAs we transition the 401(k) plan administrator from our current provider, there will be a mandatory 'blackout period' starting October 29 through November 12. During this time, employees cannot sell or transfer any holdings including Enron stock.\n\nGiven the current stock price volatility ($33 and falling), this will create significant employee concern. 60% of the 401(k) assets are in Enron stock - approximately $2.1B.\n\nCindy Olson\nEVP Human Resources"},
        {"email_id":"t10_02","message_id":"<t10_02@enron.com>","in_reply_to":"<t10_01@enron.com>","references":"<t10_01@enron.com>","from":"kenneth.lay@enron.com","to":"cindy.olson@enron.com","subject":"Re: 401(k) Plan - Enron Stock Lockdown","timestamp":"2001-10-27T08:00:00","body":"Cindy,\n\nThis is terrible timing. Employees will be trapped in Enron stock during what could be the worst period for our share price. We have a fiduciary duty.\n\nCan we accelerate the transition? Or at minimum, communicate clearly about the blackout period?\n\nAlso - I need to disclose that I sold $70M in Enron stock over the past year through my personal accounts, even while telling employees to buy more.\n\nKen"},
        {"email_id":"t10_03","message_id":"<t10_03@enron.com>","in_reply_to":"<t10_02@enron.com>","references":"<t10_01@enron.com> <t10_02@enron.com>","from":"cindy.olson@enron.com","to":"kenneth.lay@enron.com","subject":"Re: 401(k) Plan - Enron Stock Lockdown","timestamp":"2001-10-29T10:00:00","body":"Ken,\n\nThe blackout has begun. We're already getting hundreds of calls from employees who want to sell. Many are watching their retirement savings evaporate - some employees had 100% of their 401(k) in Enron stock.\n\nTotal employee 401(k) exposure: $2.1B in Enron stock. At today's price ($27), that's already down from $3.5B at the peak.\n\nWe cannot accelerate the transition.\n\nCindy Olson"},
        {"email_id":"t10_04","message_id":"<t10_04@enron.com>","in_reply_to":"<t10_03@enron.com>","references":"<t10_01@enron.com> <t10_02@enron.com> <t10_03@enron.com>","from":"employee.rep@enron.com","to":"kenneth.lay@enron.com","subject":"Re: 401(k) Plan - Enron Stock Lockdown","timestamp":"2001-11-05T09:00:00","body":"Mr. Lay,\n\nI am writing on behalf of Enron employees who are unable to protect their retirement savings. During the past week:\n\n- ENE stock dropped from $27 to $10\n- Employee 401(k) losses exceed $1.3B\n- We are locked out and cannot sell\n- Meanwhile, executives sold millions in stock before the decline\n\nEmployees feel betrayed. Many are close to retirement and have lost everything. We are exploring legal options.\n\nEmployee Representative"},
        {"email_id":"t10_05","message_id":"<t10_05@enron.com>","in_reply_to":"<t10_04@enron.com>","references":"<t10_01@enron.com> <t10_02@enron.com> <t10_03@enron.com> <t10_04@enron.com>","from":"legal.dept@enron.com","to":"kenneth.lay@enron.com","subject":"Re: 401(k) Plan - Enron Stock Lockdown","timestamp":"2001-11-12T14:00:00","body":"Ken,\n\nThe blackout period has ended. The damage:\n- Enron stock: $8.41 (down from $33 at start of blackout)\n- Total 401(k) losses during blackout: approximately $1.7B\n- Multiple class-action lawsuits have been filed by employees\n- DOL is investigating potential ERISA violations\n\nThe optics of executive stock sales during this period are extremely damaging. Several executives sold $100M+ in stock over the past year while employees were locked out.\n\nLegal Department"},
    ],"message_count":5})

    # ── Thread 11: International Operations ─────────────────────────
    threads.append({"thread_id":"t11","subject":"Dabhol Power Plant - India Operations","participants":["rebecca.mark@enron.com","kenneth.lay@enron.com","joe.sutton@enron.com","wade.cline@enron.com","sanjay.bhatnagar@enron.com"],"messages":[
        {"email_id":"t11_01","message_id":"<t11_01@enron.com>","in_reply_to":"","references":"","from":"rebecca.mark@enron.com","to":"kenneth.lay@enron.com","subject":"Dabhol Power Plant - India Operations","timestamp":"2001-02-10T08:00:00","body":"Ken,\n\nThe Dabhol power plant situation in Maharashtra, India is deteriorating rapidly. The state government is refusing to pay for electricity, claiming the tariffs are too high at $0.07/kWh when domestic power costs $0.03/kWh.\n\nTotal investment: $2.9B (Enron's share: $900M)\nOutstanding receivables: $280M\nMonthly operating cost: $25M\n\nI recommend we invoke the sovereign guarantee and engage the Indian federal government directly.\n\nRebecca Mark\nVice Chairman, Enron International"},
        {"email_id":"t11_02","message_id":"<t11_02@enron.com>","in_reply_to":"<t11_01@enron.com>","references":"<t11_01@enron.com>","from":"joe.sutton@enron.com","to":"kenneth.lay@enron.com","subject":"Re: Dabhol Power Plant - India Operations","timestamp":"2001-03-05T11:00:00","body":"Ken,\n\nUpdate on Dabhol: Maharashtra MSEB has officially stopped payments. We're owed $280M and counting. The plant is running at 10% capacity because there's no buyer for the power at our contracted rate.\n\nOur options are limited:\n1. Negotiate a reduced tariff (take a loss)\n2. Invoke arbitration under bilateral investment treaty\n3. Sell our stake at a massive loss\n\nThis was always the risk of a single-customer power project in an emerging market.\n\nJoe Sutton\nCEO, Enron International"},
        {"email_id":"t11_03","message_id":"<t11_03@enron.com>","in_reply_to":"<t11_02@enron.com>","references":"<t11_01@enron.com> <t11_02@enron.com>","from":"wade.cline@enron.com","to":"kenneth.lay@enron.com","subject":"Re: Dabhol Power Plant - India Operations","timestamp":"2001-04-20T09:30:00","body":"Ken,\n\nPolitical situation update: The new Maharashtra government led by BJP is openly hostile to Enron. They view the Dabhol deal as an example of 'Western exploitation' and are using it as a political issue.\n\nUS Ambassador has raised the issue with Delhi but getting limited traction. The $900M write-down may be unavoidable.\n\nWade Cline\nManaging Director, Enron India"},
        {"email_id":"t11_04","message_id":"<t11_04@enron.com>","in_reply_to":"<t11_03@enron.com>","references":"<t11_01@enron.com> <t11_02@enron.com> <t11_03@enron.com>","from":"sanjay.bhatnagar@enron.com","to":"joe.sutton@enron.com","subject":"Re: Dabhol Power Plant - India Operations","timestamp":"2001-06-15T07:00:00","body":"Joe,\n\nDabhol plant has been shut down completely. We cannot continue operating at a loss with no payment from MSEB. 1,500 Indian employees are on paid leave.\n\nThe Indian press coverage is devastating - we're being portrayed as a predatory multinational. Any future business in India will be extremely difficult.\n\nSanjay Bhatnagar\nCountry Manager, Enron India"},
        {"email_id":"t11_05","message_id":"<t11_05@enron.com>","in_reply_to":"<t11_04@enron.com>","references":"<t11_01@enron.com> <t11_02@enron.com> <t11_03@enron.com> <t11_04@enron.com>","from":"kenneth.lay@enron.com","to":"rebecca.mark@enron.com","subject":"Re: Dabhol Power Plant - India Operations","timestamp":"2001-07-10T15:00:00","body":"Rebecca,\n\nThe board has decided to write down the Dabhol investment. We're taking a $900M charge against Enron International. This, combined with the Azurix water utility write-down of $350M, means International has lost $1.25B.\n\nI know you championed both projects. Given the losses, we need to restructure the International division entirely.\n\nKen Lay"},
    ],"message_count":5})

    # ── Thread 12: Mark-to-Market Accounting Questions ──────────────
    threads.append({"thread_id":"t12","subject":"MTM Accounting Methodology Review","participants":["wes.colwell@enron.com","rick.causey@enron.com","david.duncan@andersen.com","jeff.skilling@enron.com","vince.kaminski@enron.com"],"messages":[
        {"email_id":"t12_01","message_id":"<t12_01@enron.com>","in_reply_to":"","references":"","from":"wes.colwell@enron.com","to":"rick.causey@enron.com","subject":"MTM Accounting Methodology Review","timestamp":"2001-05-10T10:00:00","body":"Rick,\n\nI need to flag concerns about our mark-to-market accounting practices across several business units:\n\n1. Wholesale Services: MTM gains of $1.4B on long-dated contracts with no liquid market\n2. Broadband: $110M booked on Blockbuster deal that's collapsed\n3. Retail Energy: $180M in projected savings booked upfront on multi-year contracts\n\nWe're using internal models to value positions where no external market prices exist. This is technically permitted but aggressive.\n\nWes Colwell\nChief Accounting Officer, Enron Americas"},
        {"email_id":"t12_02","message_id":"<t12_02@enron.com>","in_reply_to":"<t12_01@enron.com>","references":"<t12_01@enron.com>","from":"rick.causey@enron.com","to":"david.duncan@andersen.com","subject":"Re: MTM Accounting Methodology Review","timestamp":"2001-05-12T14:00:00","body":"David,\n\nForwarding Wes's concerns for your review. We need Arthur Andersen's opinion on whether our MTM methodology is defensible for:\n- Long-dated energy contracts (10-20 year forwards)\n- Bandwidth trading (essentially illiquid market)\n- Retail energy savings projections\n\nOur total MTM-related revenue recognition is approximately $1.7B. If this methodology is challenged, the impact would be material.\n\nRick Causey"},
        {"email_id":"t12_03","message_id":"<t12_03@enron.com>","in_reply_to":"<t12_02@enron.com>","references":"<t12_01@enron.com> <t12_02@enron.com>","from":"david.duncan@andersen.com","to":"rick.causey@enron.com","subject":"Re: MTM Accounting Methodology Review","timestamp":"2001-05-15T09:30:00","body":"Rick,\n\nWe've reviewed the MTM methodology. Our assessment:\n- Energy forward curves beyond 5 years: insufficient market data, models are subjective\n- Broadband: no established market, valuations are essentially management estimates\n- Retail: early recognition is aggressive but within GAAP interpretation\n\nWe can defend the current treatment but I'll note we classified Enron as a 'high risk' client internally. Any future scrutiny would focus on these exact issues.\n\nDavid Duncan\nLead Partner, Andersen"},
        {"email_id":"t12_04","message_id":"<t12_04@enron.com>","in_reply_to":"<t12_03@enron.com>","references":"<t12_01@enron.com> <t12_02@enron.com> <t12_03@enron.com>","from":"vince.kaminski@enron.com","to":"rick.causey@enron.com","subject":"Re: MTM Accounting Methodology Review","timestamp":"2001-05-18T16:00:00","body":"Rick,\n\nFrom the Research group's perspective, the forward curves used for MTM valuation beyond Year 5 are essentially fabricated. There is no liquid market data. The curves are generated by our own models and then used to book billions in revenue.\n\nThis is circular - we're using our own assumptions to validate our own profits. My team has been raising this issue for two years.\n\nVince Kaminski\nManaging Director, Research"},
        {"email_id":"t12_05","message_id":"<t12_05@enron.com>","in_reply_to":"<t12_04@enron.com>","references":"<t12_01@enron.com> <t12_02@enron.com> <t12_03@enron.com> <t12_04@enron.com>","from":"jeff.skilling@enron.com","to":"rick.causey@enron.com","subject":"Re: MTM Accounting Methodology Review","timestamp":"2001-05-20T08:00:00","body":"Rick,\n\nOur MTM accounting is our competitive advantage. It allows us to recognize the value we create immediately rather than waiting for cash flows over decades. Every major Wall Street firm uses similar methods.\n\nVince's concerns are noted but the methodology has been approved by Arthur Andersen. I don't want this relitigated.\n\nKeep the current approach.\n\nJeff Skilling\nCEO"},
    ],"message_count":5})

    # ── Thread 13: Bankruptcy Preparation ────────────────────────────
    threads.append({"thread_id":"t13","subject":"Chapter 11 Filing Preparation","participants":["james.derrick@enron.com","kenneth.lay@enron.com","greg.whalley@enron.com","jeff.mcmahon@enron.com","ray.bowen@enron.com"],"messages":[
        {"email_id":"t13_01","message_id":"<t13_01@enron.com>","in_reply_to":"","references":"","from":"james.derrick@enron.com","to":"kenneth.lay@enron.com","subject":"Chapter 11 Filing Preparation","timestamp":"2001-11-29T06:00:00","body":"Ken,\n\nWith the Dynegy deal collapsed and credit at junk, we need to prepare the Chapter 11 filing immediately. Key items:\n\n- Total assets: $63.4B (as stated, actual value is debatable)\n- Total debt: $31.2B (including off-balance-sheet)\n- Cash on hand: $1.1B (and declining daily)\n- Employees worldwide: 20,600\n\nWe've retained Weil Gotshal & Manges as bankruptcy counsel. Target filing date: December 2, 2001.\n\nJim Derrick\nGeneral Counsel"},
        {"email_id":"t13_02","message_id":"<t13_02@enron.com>","in_reply_to":"<t13_01@enron.com>","references":"<t13_01@enron.com>","from":"jeff.mcmahon@enron.com","to":"kenneth.lay@enron.com","subject":"Re: Chapter 11 Filing Preparation","timestamp":"2001-11-29T10:00:00","body":"Ken,\n\nTreasury update for the filing:\n- We have $1.1B in cash but $3.9B in payment obligations triggered by the junk downgrade\n- EnronOnline has been shut down - no counterparty will trade with us\n- Northern Natural Gas Pipeline (our best asset) has been claimed by Dynegy as merger break-up fee\n- Employee severance obligations: estimated $300M\n\nWe may need DIP financing of $1.5B to operate through bankruptcy.\n\nJeff McMahon\nCFO"},
        {"email_id":"t13_03","message_id":"<t13_03@enron.com>","in_reply_to":"<t13_02@enron.com>","references":"<t13_01@enron.com> <t13_02@enron.com>","from":"greg.whalley@enron.com","to":"kenneth.lay@enron.com","subject":"Re: Chapter 11 Filing Preparation","timestamp":"2001-11-30T08:00:00","body":"Ken,\n\nOperations status:\n- Trading floor shut down November 28\n- 4,000 employees terminated immediately, more to follow\n- London and India offices being closed\n- IT systems: we need to preserve all electronic records for SEC/DOJ investigations\n\nI'm recommending we keep a skeleton crew of 3,000 to manage the asset disposition through bankruptcy. The pipeline, power plants, and physical assets still have significant value.\n\nGreg Whalley"},
        {"email_id":"t13_04","message_id":"<t13_04@enron.com>","in_reply_to":"<t13_03@enron.com>","references":"<t13_01@enron.com> <t13_02@enron.com> <t13_03@enron.com>","from":"ray.bowen@enron.com","to":"greg.whalley@enron.com","subject":"Re: Chapter 11 Filing Preparation","timestamp":"2001-11-30T14:00:00","body":"Greg,\n\nFinal financial summary for the filing:\n- Stock price: $0.61 (from peak of $90.75 in August 2000)\n- Market cap loss: $74B in 12 months\n- 401(k) employee losses: approximately $2B\n- Total creditor claims: estimated $67B\n\nThis will be the largest corporate bankruptcy in US history, surpassing WorldCom.\n\nRay Bowen\nCFO, Enron Industrial Markets"},
        {"email_id":"t13_05","message_id":"<t13_05@enron.com>","in_reply_to":"<t13_04@enron.com>","references":"<t13_01@enron.com> <t13_02@enron.com> <t13_03@enron.com> <t13_04@enron.com>","from":"kenneth.lay@enron.com","to":"james.derrick@enron.com","subject":"Re: Chapter 11 Filing Preparation","timestamp":"2001-12-01T20:00:00","body":"Jim,\n\nProceed with the Chapter 11 filing tomorrow. I will address employees and the press.\n\nI still believe the core energy business has value and can emerge from bankruptcy. But I accept responsibility for what has happened. We built something extraordinary and then failed to manage the risks.\n\nKen Lay\nChairman & CEO"},
    ],"message_count":5})

    # ── Thread 14: SEC Investigation ────────────────────────────────
    threads.append({"thread_id":"t14","subject":"SEC Formal Investigation - Response Strategy","participants":["james.derrick@enron.com","kenneth.lay@enron.com","mark.koenig@enron.com","jordan.mintz@enron.com","robert.mueller@sec.gov"],"messages":[
        {"email_id":"t14_01","message_id":"<t14_01@enron.com>","in_reply_to":"","references":"","from":"james.derrick@enron.com","to":"kenneth.lay@enron.com","subject":"SEC Formal Investigation - Response Strategy","timestamp":"2001-10-31T08:00:00","body":"Ken,\n\nThe SEC has upgraded their preliminary inquiry to a formal investigation. They're demanding:\n1. All documents related to LJM partnerships and SPE transactions\n2. Internal communications about Raptor vehicles\n3. Trading records for California energy markets\n4. Personal trading records of all officers and directors\n5. All Arthur Andersen work papers and correspondence\n\nWe must comply fully. Any obstruction at this point would be catastrophic.\n\nJim Derrick\nGeneral Counsel"},
        {"email_id":"t14_02","message_id":"<t14_02@enron.com>","in_reply_to":"<t14_01@enron.com>","references":"<t14_01@enron.com>","from":"kenneth.lay@enron.com","to":"james.derrick@enron.com","subject":"Re: SEC Formal Investigation - Response Strategy","timestamp":"2001-10-31T15:00:00","body":"Jim,\n\nFull cooperation. Instruct all departments to preserve everything. No document destruction of any kind.\n\nI also need to disclose my personal stock sales. Over the past 12 months I sold approximately $70M in Enron stock while publicly encouraging employees and investors to buy. The SEC will focus on this.\n\nRetain separate criminal defense counsel for me personally.\n\nKen"},
        {"email_id":"t14_03","message_id":"<t14_03@enron.com>","in_reply_to":"<t14_02@enron.com>","references":"<t14_01@enron.com> <t14_02@enron.com>","from":"mark.koenig@enron.com","to":"kenneth.lay@enron.com","subject":"Re: SEC Formal Investigation - Response Strategy","timestamp":"2001-11-01T09:00:00","body":"Ken,\n\nThe SEC investigation combined with our $618M Q3 charge and the revelation of the $1.2B equity reduction has destroyed market confidence. S&P has us on negative credit watch.\n\nThe Q3 earnings call was a disaster. Analysts are now questioning every transaction we've done. Our credibility is zero.\n\nMoody's is threatening a downgrade to junk. If that happens, it triggers $3.9B in obligations.\n\nMark Koenig\nIR"},
        {"email_id":"t14_04","message_id":"<t14_04@enron.com>","in_reply_to":"<t14_03@enron.com>","references":"<t14_01@enron.com> <t14_02@enron.com> <t14_03@enron.com>","from":"jordan.mintz@enron.com","to":"james.derrick@enron.com","subject":"Re: SEC Formal Investigation - Response Strategy","timestamp":"2001-11-05T11:00:00","body":"Jim,\n\nAs General Counsel for Enron Global Finance, I want to go on record that I raised concerns about the LJM-related transactions to Jeff Skilling's office on multiple occasions in 2000 and 2001. I was unable to get sign-off from Rick Causey on several deals that I believed had inadequate documentation.\n\nI want to ensure my communications are preserved as part of the SEC response. I may need my own counsel.\n\nJordan Mintz"},
        {"email_id":"t14_05","message_id":"<t14_05@enron.com>","in_reply_to":"<t14_04@enron.com>","references":"<t14_01@enron.com> <t14_02@enron.com> <t14_03@enron.com> <t14_04@enron.com>","from":"james.derrick@enron.com","to":"kenneth.lay@enron.com","subject":"Re: SEC Formal Investigation - Response Strategy","timestamp":"2001-11-10T16:00:00","body":"Ken,\n\nUpdate: The SEC has now referred the matter to the Department of Justice for potential criminal prosecution. This means:\n- Grand jury investigation likely\n- Individual executives may face criminal charges\n- Andy Fastow, Ben Glisan, and potentially others are targets\n- Arthur Andersen's document destruction is now a separate obstruction case\n\nEvery executive needs personal criminal defense counsel immediately.\n\nJim Derrick"},
    ],"message_count":5})

    # ── Thread 15: Post-Bankruptcy Legacy ───────────────────────────
    threads.append({"thread_id":"t15","subject":"Year End Review - Lessons Learned","participants":["rick.buy@enron.com","vince.kaminski@enron.com","sally.beck@enron.com","john.lavorato@enron.com","louise.kitchen@enron.com"],"messages":[
        {"email_id":"t15_01","message_id":"<t15_01@enron.com>","in_reply_to":"","references":"","from":"rick.buy@enron.com","to":"vince.kaminski@enron.com","subject":"RE: Risk Assessment - Year End Review","timestamp":"2001-12-01T10:00:00","body":"Vince,\n\nYou were right. About everything. The California risk warnings, the Raptor concerns, the VaR concentration issues - all materialized exactly as your research team predicted.\n\nThe bankruptcy filing happened yesterday. $63B in assets, largest in US history. 20,000 people losing their jobs. Retirement savings wiped out.\n\nAs Chief Risk Officer, I failed to push hard enough when my concerns were overruled.\n\nRick Buy\nChief Risk Officer"},
        {"email_id":"t15_02","message_id":"<t15_02@enron.com>","in_reply_to":"<t15_01@enron.com>","references":"<t15_01@enron.com>","from":"vince.kaminski@enron.com","to":"rick.buy@enron.com","subject":"Re: Year End Review","timestamp":"2001-12-02T08:00:00","body":"Rick,\n\nThe problem was structural. The risk management function reported to the business units it was supposed to monitor. When I raised concerns about Raptor to Andy Fastow, my team was removed from reviewing LJM transactions.\n\nThe culture rewarded revenue generation and punished dissent. My researchers who flagged issues were marginalized.\n\nThis is what happens when risk management has no independence.\n\nVince Kaminski"},
        {"email_id":"t15_03","message_id":"<t15_03@enron.com>","in_reply_to":"<t15_02@enron.com>","references":"<t15_01@enron.com> <t15_02@enron.com>","from":"sally.beck@enron.com","to":"vince.kaminski@enron.com","subject":"Re: Year End Review","timestamp":"2001-12-03T08:00:00","body":"Vince,\n\nFrom the operations side, I saw the same pattern. The trading floor was generating enormous reported profits but the back office couldn't verify many of the valuations. We were processing 6,000 trades daily with outdated systems and insufficient staff.\n\nThe entire infrastructure was optimized for speed and revenue, not for accuracy or controls.\n\nSally Beck\nCOO, Enron Americas"},
        {"email_id":"t15_04","message_id":"<t15_04@enron.com>","in_reply_to":"<t15_03@enron.com>","references":"<t15_01@enron.com> <t15_02@enron.com> <t15_03@enron.com>","from":"john.lavorato@enron.com","to":"rick.buy@enron.com","subject":"Re: Year End Review","timestamp":"2001-12-03T14:00:00","body":"Rick,\n\nThe trading business itself was actually profitable and legitimate for the most part. What killed us was the off-balance-sheet structures, the conflicts of interest, and the accounting games.\n\nEnron's core energy trading and pipeline business had real value. It was the 'financial innovation' - SPEs, mark-to-market on illiquid assets, self-hedging - that was fraudulent.\n\nJohn Lavorato"},
        {"email_id":"t15_05","message_id":"<t15_05@enron.com>","in_reply_to":"<t15_04@enron.com>","references":"<t15_01@enron.com> <t15_02@enron.com> <t15_03@enron.com> <t15_04@enron.com>","from":"louise.kitchen@enron.com","to":"sally.beck@enron.com","subject":"Re: Year End Review","timestamp":"2001-12-10T09:00:00","body":"Sally,\n\nEnronOnline processed $880B in transactions in its lifetime. The technology was genuinely revolutionary. We built the first electronic commodity trading platform.\n\nBut Louise Kitchen's EnronOnline needed Enron's credit rating to function. When that disappeared, the entire business model collapsed in 48 hours.\n\nLesson: technology alone isn't enough. The business foundation has to be sound.\n\nLouise Kitchen\nPresident, Enron Online"},
    ],"message_count":5})

    total_msgs = sum(t["message_count"] for t in threads)
    print(f"  Generated {len(threads)} synthetic threads ({total_msgs} messages)")
    return threads


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Layer10 Memory Pipeline – Thread-Wise Corpus Preparation")
    print("  Dataset : Enron Email Corpus (CMU)")
    print(f"  Target  : {TARGET_THREAD_COUNT} threads × "
          f"{MIN_MESSAGES_PER_THREAD}-{MAX_MESSAGES_PER_THREAD} msgs each")
    print("=" * 60)

    threads: list[dict] = []

    # 1. Already-extracted maildir → group into threads
    if os.path.isdir(ENRON_MAILDIR):
        cnt = sum(1 for _ in Path(ENRON_MAILDIR).rglob("*") if _.is_file())
        if cnt > 1000:
            print(f"\n[1] Found existing maildir ({cnt:,} files)")
            raw_emails = load_from_maildir(ENRON_MAILDIR)
            if raw_emails:
                print(f"  Building conversation threads …")
                all_threads = build_threads(raw_emails)
                print(f"  Found {len(all_threads):,} raw threads")
                threads = select_threads(all_threads)

    # 2. Download tarball → extract → parse
    if not threads:
        print(f"\n[2] Downloading Enron maildir tarball (~1.7 GB) …")
        tar = download_enron_tarball()
        if tar:
            md = extract_tarball(tar)
            if md:
                raw_emails = load_from_maildir(md)
                all_threads = build_threads(raw_emails)
                threads = select_threads(all_threads)

    # 3. Synthetic fallback
    if not threads:
        print("\n[3] Using synthetic Enron threads")
        threads = create_synthetic_threads()

    # Summary
    total_msgs = sum(t["message_count"] for t in threads)
    all_participants = set()
    for t in threads:
        all_participants.update(t.get("participants", []))

    print(f"\n  ✓ {len(threads)} threads ready ({total_msgs} messages)")
    if threads:
        print(f"    Participants   : {len(all_participants)}")
        for i, t in enumerate(threads):
            print(f"    [{i+1:2d}] {t['subject'][:55]:<55s}  "
                  f"({t['message_count']} msgs, "
                  f"{len(t.get('participants',[]))} people)")

    # Save
    with open(CORPUS_RAW_PATH, "w") as f:
        json.dump(threads, f, indent=2)
    print(f"\n    Saved to: {CORPUS_RAW_PATH} "
          f"({os.path.getsize(CORPUS_RAW_PATH)/1024:.1f} KB)")
    return threads


if __name__ == "__main__":
    main()
