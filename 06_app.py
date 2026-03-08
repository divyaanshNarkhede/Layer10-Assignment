"""
Step 6 – Streamlit Memory-Graph Explorer.

Interactive front-end for the Layer10 grounded memory system.

Pages (sidebar navigation):
  1. Query        – Semantic search & grounded Q&A (with LLM synthesis)
  2. Graph        – Interactive vis-network.js visualisation
  3. Evidence     – Browse all entities, claims & evidence chains
  4. Merges       – Audit log of entity/claim deduplication (with undo)
  5. Statistics   – High-level graph metrics

Launch:
    streamlit run 06_app.py
"""
import json, os, sys, html, hashlib
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    GRAPH_PATH, CONTEXT_PACKS_PATH, OUTPUT_DIR, EMBEDDING_MODEL,
    OLLAMA_MODEL
)

import streamlit as st

# ═════════════════════════════════════════════════════════════════════════════
# Page config
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Layer10 | Memory Graph",
    page_icon="L10",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# Custom CSS  (no tab hacks — sidebar radio nav is used instead)
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ─── Sidebar ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] { min-width: 270px; max-width: 330px; }
[data-testid="stSidebar"] .stRadio > label {
    font-size: 1.05rem !important;
    font-weight: 600;
}
[data-testid="stSidebar"] .stRadio > div {
    gap: 2px;
}
[data-testid="stSidebar"] .stRadio > div label {
    padding: 0.45rem 0.6rem !important;
    border-radius: 8px;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stRadio > div label:hover {
    background: rgba(255,255,255,0.06);
}

/* ─── Main area ─────────────────────────────────────────────────── */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 1200px;
}

/* ─── Cards ─────────────────────────────────────────────────────── */
.card {
    background: var(--background-color, #1e1e2f);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: .9rem;
    transition: box-shadow .2s;
    word-wrap: break-word;
    overflow-wrap: break-word;
}
.card:hover { box-shadow: 0 4px 20px rgba(0,0,0,.25); }
.card h4 { margin: 0 0 .4rem; font-size: 1rem; word-break: break-word; }
.card p  { margin: 0; font-size: .88rem; opacity: .85; line-height: 1.5; }
.card .meta { font-size: .78rem; opacity: .55; margin-top: .3rem; }

/* ─── Badges ────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 30px;
    font-size: .73rem;
    font-weight: 600;
    letter-spacing: .3px;
    margin-right: 4px;
    color: #fff;
    white-space: nowrap;
}

/* ─── Metric tiles ──────────────────────────────────────────────── */
.metric-tile {
    text-align: center;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,.06);
}
.metric-tile .val {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
}
.metric-tile .lbl {
    font-size: .78rem;
    opacity: .6;
    margin-top: .2rem;
}

/* ─── Evidence box ──────────────────────────────────────────────── */
.evidence-box {
    background: rgba(255,255,255,.04);
    border-left: 3px solid #2196F3;
    padding: .6rem .9rem;
    margin: .4rem 0;
    border-radius: 0 8px 8px 0;
    font-size: .84rem;
    line-height: 1.45;
    word-wrap: break-word;
}

/* ─── Query result ──────────────────────────────────────────────── */
.answer-box {
    background: linear-gradient(135deg, rgba(33,150,243,.08), rgba(76,175,80,.06));
    border: 1px solid rgba(33,150,243,.25);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: .8rem 0;
    line-height: 1.6;
}

/* ─── Page header ───────────────────────────────────────────────── */
.page-header {
    border-bottom: 2px solid rgba(255,255,255,.06);
    padding-bottom: 0.6rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Type → colour mapping
# ═════════════════════════════════════════════════════════════════════════════
TYPE_COLOURS = {
    "Person":              "#4CAF50",
    "Organisation":        "#2196F3",
    "System":              "#42A5F5",
    "Project":             "#9C27B0",
    "FinancialInstrument": "#1565C0",
    "Event":               "#00BCD4",
    "Role":                "#795548",
    "Location":            "#607D8B",
}

STATUS_COLOURS = {
    "current":    "#4CAF50",
    "historical": "#42A5F5",
    "retracted":  "#1565C0",
    "disputed":   "#9C27B0",
}

def type_badge(t: str) -> str:
    c = TYPE_COLOURS.get(t, "#888")
    return f'<span class="badge" style="background:{c}">{html.escape(t)}</span>'

def status_badge(s: str) -> str:
    c = STATUS_COLOURS.get(s, "#888")
    return f'<span class="badge" style="background:{c}">{html.escape(s)}</span>'


# ═════════════════════════════════════════════════════════════════════════════
# Data loading (cached)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_graph():
    with open(GRAPH_PATH) as f:
        return json.load(f)

@st.cache_data(ttl=300)
def load_stats():
    stats_path = os.path.join(OUTPUT_DIR, "graph_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=300)
def load_context_packs():
    if os.path.exists(CONTEXT_PACKS_PATH):
        with open(CONTEXT_PACKS_PATH) as f:
            return json.load(f)
    return []


# ═════════════════════════════════════════════════════════════════════════════
# Build vis-network HTML (embedded)
# ═════════════════════════════════════════════════════════════════════════════
def build_vis_html(entities, claims, height=620, filter_types=None, search_q=""):
    """Return a self-contained HTML string using vis-network.js."""

    # Read bundled JS/CSS
    base = os.path.dirname(os.path.abspath(__file__))
    vis_js  = Path(base, "lib", "vis-9.1.2", "vis-network.min.js").read_text()
    vis_css = Path(base, "lib", "vis-9.1.2", "vis-network.css").read_text()

    # Prepare nodes
    ent_map = {e["id"]: e for e in entities}
    nodes_js = []
    search_lower = search_q.lower().strip()

    connected_ids = set()
    for c in claims:
        if c.get("status") == "retracted":
            continue
        connected_ids.add(c["subject_id"])
        connected_ids.add(c["object_id"])

    for e in entities:
        if e["id"] not in connected_ids:
            continue
        if filter_types and e["type"] not in filter_types:
            continue
        colour = TYPE_COLOURS.get(e["type"], "#888")
        label = e["name"][:25]
        opacity = 1.0
        if search_lower and search_lower not in e["name"].lower():
            opacity = 0.2
        nodes_js.append(
            f'{{id:"{e["id"]}",label:"{_js(label)}",'
            f'color:{{background:"{colour}",border:"{colour}"}},'
            f'font:{{color:"#eee",size:12}},'
            f'opacity:{opacity},'
            f'title:"{_js(e["name"])} ({e["type"]})",'
            f'shape:"dot",size:12}}'
        )

    # Prepare edges
    edges_js = []
    for c in claims:
        if c.get("status") == "retracted":
            continue
        if c["subject_id"] not in connected_ids or c["object_id"] not in connected_ids:
            continue
        if filter_types:
            s_ent = ent_map.get(c["subject_id"])
            o_ent = ent_map.get(c["object_id"])
            if s_ent and s_ent["type"] not in filter_types:
                continue
            if o_ent and o_ent["type"] not in filter_types:
                continue
        rel = c.get("relation", "related_to")
        conf = c.get("confidence", 0.5)
        colour = STATUS_COLOURS.get(c.get("status", "current"), "#4CAF50")
        width = max(1, conf * 3)
        edges_js.append(
            f'{{from:"{c["subject_id"]}",to:"{c["object_id"]}",'
            f'label:"{_js(rel)}",'
            f'arrows:"to",'
            f'color:{{color:"{colour}",opacity:{max(0.3, conf)}}},'
            f'width:{width:.1f},'
            f'font:{{size:9,color:"#999",strokeWidth:0}},'
            f'title:"{_js(c.get("subject_name",""))} → {_js(rel)} → {_js(c.get("object_name",""))} (conf={conf:.2f})"}}'
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>{vis_css}
html,body{{margin:0;padding:0;overflow:hidden;background:#0e1117;}}
#g{{width:100%;height:{height}px;}}
#legend{{position:absolute;top:10px;right:14px;background:rgba(14,17,23,.85);
  border:1px solid rgba(255,255,255,.1);border-radius:10px;padding:10px 14px;
  font:12px/1.6 system-ui;color:#ccc;}}
#legend span{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:middle;}}
</style></head><body>
<div id="g"></div>
<div id="legend">
  {"".join(f'<div><span style="background:{c}"></span>{t}</div>' for t,c in TYPE_COLOURS.items())}
</div>
<script>{vis_js}</script>
<script>
var nodes=new vis.DataSet([{",".join(nodes_js)}]);
var edges=new vis.DataSet([{",".join(edges_js)}]);
var container=document.getElementById('g');
var data={{nodes:nodes,edges:edges}};
var options={{
  physics:{{barnesHut:{{gravitationalConstant:-4000,centralGravity:0.25,springLength:120,damping:0.15}},
           stabilization:{{iterations:120,fit:true}}}},
  interaction:{{hover:true,tooltipDelay:100,zoomView:true,dragView:true,
               navigationButtons:false,keyboard:true}},
  edges:{{smooth:{{type:"continuous"}}}},
  layout:{{improvedLayout:true}}
}};
var network=new vis.Network(container,data,options);
</script></body></html>"""


def _js(s: str) -> str:
    """Escape for JS string literal."""
    return (
        str(s)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", "")
    )


# ═════════════════════════════════════════════════════════════════════════════
# Retrieval helpers
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)

def embed_text(text: str):
    import numpy as np
    model = get_embed_model()
    return model.encode(text, normalize_embeddings=True)

def cosine_sim(a, b):
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def search_graph(query: str, graph_data: dict, top_k: int = 15):
    """Semantic search over entities + claims, return ranked results."""
    q_emb = embed_text(query)
    scored = []

    for e in graph_data["entities"]:
        text = f"{e['name']} ({e['type']})"
        sim = cosine_sim(q_emb, embed_text(text))
        scored.append(("entity", e, sim))

    for c in graph_data["claims"]:
        text = f"{c.get('subject_name','')} {c.get('relation','')} {c.get('object_name','')}"
        sim = cosine_sim(q_emb, embed_text(text))
        conf = c.get("confidence", 0.5)
        scored.append(("claim", c, sim * 0.6 + conf * 0.4))

    scored.sort(key=lambda x: -x[2])
    return scored[:top_k]

def format_grounded_answer(results):
    """Format search results into a grounded narrative."""
    lines = []
    for kind, item, score in results[:8]:
        if kind == "entity":
            lines.append(f"• **{item['name']}** ({item['type']})")
        else:
            subj = item.get("subject_name", "?")
            rel = item.get("relation", "?").replace("_", " ")
            obj = item.get("object_name", "?")
            conf = item.get("confidence", 0)
            ev_count = len(item.get("evidence", []))
            lines.append(
                f"• {subj} **{rel}** {obj}  "
                f"*(confidence: {conf:.0%}, {ev_count} evidence)*"
            )
    return "\n".join(lines)


# ─── Ollama-powered answer synthesis ─────────────────────────────────────────
_HAS_OLLAMA_UI = False
try:
    if not os.environ.get("DISABLE_OLLAMA"):
        import ollama as _ollama_client
        _HAS_OLLAMA_UI = True
except ImportError:
    pass


def generate_llm_answer(query: str, results: list) -> str | None:
    """
    Call Ollama to synthesize a grounded answer from search results.
    Returns the LLM's response string, or None on failure.
    """
    if not _HAS_OLLAMA_UI:
        return None

    # Build context from search results
    context_lines = []
    for kind, item, score in results[:10]:
        if kind == "entity":
            context_lines.append(
                f"ENTITY: {item['name']} (type={item['type']}, "
                f"first_seen={item.get('first_seen','?')}, "
                f"aliases={item.get('aliases',[])})"
            )
        else:
            evidences = item.get("evidence", [])
            ev_text = ""
            for ev in evidences[:2]:
                ev_text += f' Evidence: "{ev.get("excerpt","")[:200]}" '
                ev_text += f'[{ev.get("author","")}, {ev.get("source_id","")}]'
            context_lines.append(
                f"CLAIM: {item.get('subject_name','')} "
                f"{item.get('relation','').replace('_',' ')} "
                f"{item.get('object_name','')} "
                f"(confidence={item.get('confidence',0):.0%}, "
                f"status={item.get('status','current')}).{ev_text}"
            )

    context_block = "\n".join(context_lines)

    system_msg = (
        "You are a knowledgeable assistant with access to a memory graph of the "
        "Enron email corpus. Answer the user's question using ONLY the provided "
        "context. Cite specific evidence when available. If the context doesn't "
        "contain enough information, say so. Be concise and factual."
    )
    user_msg = (
        f"Question: {query}\n\n"
        f"--- MEMORY GRAPH CONTEXT ---\n{context_block}\n--- END CONTEXT ---\n\n"
        f"Provide a grounded answer based on the above context."
    )

    try:
        response = _ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            options={
                "temperature": 0.2,
                "num_predict": 512,
                "num_gpu": 99,
                "num_ctx": 4096,
            },
            keep_alive="10m",
        )
        return response["message"]["content"]
    except Exception as e:
        return f"LLM error: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar  (navigation + metrics)
# ═════════════════════════════════════════════════════════════════════════════
PAGES = [
    "Query",
    "Graph",
    "Evidence",
    "Merges",
    "Statistics",
]


def render_sidebar(graph_data) -> str:
    """Render the sidebar and return the selected page name."""
    with st.sidebar:
        st.markdown("## Layer10 Memory")
        st.caption("Grounded Long-Term Memory Graph")
        st.divider()

        page = st.radio(
            "Navigate",
            PAGES,
            index=0,
            key="nav_page",
            label_visibility="collapsed",
        )

        st.divider()

        ne = len(graph_data.get("entities", []))
        nc = len(graph_data.get("claims", []))
        nm = len(graph_data.get("merge_log", []))
        cols = st.columns(3)
        cols[0].metric("Entities", ne)
        cols[1].metric("Claims", nc)
        cols[2].metric("Merges", nm)

        st.divider()
        st.caption(f"Schema v{graph_data.get('schema_version','?')}")
        ci = graph_data.get("corpus_info", {})
        if ci:
            st.caption(f"Corpus: {ci.get('email_count',0)} emails")
        st.caption(f"Created: {graph_data.get('created_at','?')[:19]}")

    return page


# ═════════════════════════════════════════════════════════════════════════════
# Page 1: Query
# ═════════════════════════════════════════════════════════════════════════════
def page_query(graph_data):
    st.markdown('<div class="page-header"><h2>Semantic Query</h2></div>',
                unsafe_allow_html=True)
    st.caption("Ask natural-language questions grounded in the memory graph.")

    query = st.text_input(
        "Enter your question",
        placeholder="e.g., Who warned about accounting fraud at Enron?",
        key="query_input",
    )

    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        top_k = st.slider("Results", 5, 30, 12, key="query_k")
    with col3:
        use_llm = st.checkbox("LLM Answer", value=_HAS_OLLAMA_UI, key="use_llm",
                              disabled=not _HAS_OLLAMA_UI,
                              help="Generate a synthesized answer using Ollama LLM")

    if query:
        with st.spinner("Searching memory graph…"):
            results = search_graph(query, graph_data, top_k=top_k)

        if results:
            # --- LLM-synthesized answer ---
            if use_llm:
                with st.spinner("Generating grounded answer..."):
                    llm_answer = generate_llm_answer(query, results)
                if llm_answer:
                    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                    st.markdown(f"**AI Answer:**\n\n{llm_answer}")
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                # Fallback: template-based answer
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown("**Grounded Answer:**\n\n" + format_grounded_answer(results))
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f"**Top {len(results)} results** (by relevance)")
            for kind, item, score in results:
                if kind == "entity":
                    st.markdown(
                        f'<div class="card">'
                        f'<h4>{type_badge(item["type"])} {html.escape(item["name"])}</h4>'
                        f'<p class="meta">Score: {score:.3f} · '
                        f'First seen: {item.get("first_seen","?")[:10]}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    evidences = item.get("evidence", [])
                    ev_html = "".join(
                        f'<div class="evidence-box">"{html.escape(ev.get("excerpt","")[:200])}"'
                        f'<br><span class="meta">— {html.escape(ev.get("author",""))}, '
                        f'{html.escape(ev.get("source_id",""))}</span></div>'
                        for ev in evidences[:2]
                    )
                    st.markdown(
                        f'<div class="card">'
                        f'<h4>{html.escape(item.get("subject_name",""))} '
                        f'→ {item.get("relation","")} → '
                        f'{html.escape(item.get("object_name",""))}</h4>'
                        f'<p>{status_badge(item.get("status","current"))} '
                        f'Confidence: {item.get("confidence",0):.0%} · '
                        f'Score: {score:.3f}</p>'
                        f'{ev_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No results found.")

    # Pre-built context packs
    packs = load_context_packs()
    if packs:
        st.divider()
        st.markdown("#### Pre-built Context Packs")
        for pack in packs:
            with st.expander(f"{pack.get('question', '?')}", expanded=False):
                ans = pack.get("answer", "")
                if ans:
                    st.markdown(ans)
                ctx = pack.get("context_pack", {})
                claims_list = ctx.get("claims", [])
                if claims_list:
                    st.caption(f"{len(claims_list)} grounded claims")
                    for c in claims_list[:5]:
                        st.markdown(
                            f"- **{c.get('subject','')}** {c.get('relation','')} "
                            f"**{c.get('object','')}** *(conf: {c.get('confidence',0):.0%})*"
                        )


# ═════════════════════════════════════════════════════════════════════════════
# Page 2: Graph
# ═════════════════════════════════════════════════════════════════════════════
def page_graph(graph_data):
    st.markdown('<div class="page-header"><h2>Interactive Memory Graph</h2></div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 1])
    with col_a:
        all_types = sorted(set(e["type"] for e in graph_data["entities"]))
        sel_types = st.multiselect(
            "Filter by entity type",
            all_types,
            default=all_types,
            key="graph_types",
        )
    with col_b:
        graph_search = st.text_input(
            "Highlight entity", placeholder="e.g. Kenneth Lay", key="graph_search"
        )

    graph_html = build_vis_html(
        graph_data["entities"],
        graph_data["claims"],
        height=640,
        filter_types=set(sel_types) if sel_types else None,
        search_q=graph_search,
    )
    st.components.v1.html(graph_html, height=660, scrolling=True)

    st.caption(
        f"Showing {len(sel_types)} types · "
        f"{len(graph_data['entities'])} entities · "
        f"{len(graph_data['claims'])} claims"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Page 3: Evidence
# ═════════════════════════════════════════════════════════════════════════════
def page_evidence(graph_data):
    st.markdown('<div class="page-header"><h2>Entity & Claim Browser</h2></div>',
                unsafe_allow_html=True)

    view = st.radio("View", ["Entities", "Claims"], horizontal=True, key="ev_view")

    if view == "Entities":
        entities = graph_data.get("entities", [])
        all_types = sorted(set(e["type"] for e in entities))
        sel = st.multiselect("Filter types", all_types, default=all_types, key="ent_types")
        search = st.text_input("Search entities", key="ent_search")

        filtered = [
            e for e in entities
            if e["type"] in sel
            and (not search or search.lower() in e["name"].lower())
        ]
        st.caption(f"Showing {len(filtered)} / {len(entities)} entities")

        for e in filtered[:50]:
            aliases = ", ".join(e.get("aliases", [])) or "—"
            st.markdown(
                f'<div class="card">'
                f'<h4>{type_badge(e["type"])} {html.escape(e["name"])}</h4>'
                f'<p class="meta">ID: {e["id"]} · Aliases: {html.escape(aliases)}'
                f'<br>Seen: {e.get("first_seen","?")[:10]} – {e.get("last_seen","?")[:10]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    else:  # Claims
        claims = graph_data.get("claims", [])
        all_rels = sorted(set(c.get("relation", "") for c in claims))
        all_statuses = sorted(set(c.get("status", "") for c in claims))

        c1, c2, c3 = st.columns(3)
        with c1:
            sel_rels = st.multiselect("Relations", all_rels, default=all_rels, key="cl_rels")
        with c2:
            sel_status = st.multiselect("Status", all_statuses, default=all_statuses, key="cl_stat")
        with c3:
            min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05, key="cl_conf")

        search = st.text_input("Search claims", key="cl_search")

        filtered = [
            c for c in claims
            if c.get("relation") in sel_rels
            and c.get("status") in sel_status
            and c.get("confidence", 0) >= min_conf
            and (
                not search
                or search.lower() in c.get("subject_name", "").lower()
                or search.lower() in c.get("object_name", "").lower()
                or search.lower() in c.get("relation", "").lower()
            )
        ]
        st.caption(f"Showing {len(filtered)} / {len(claims)} claims")

        for c in filtered[:50]:
            evidences = c.get("evidence", [])
            ev_html = ""
            for ev in evidences[:3]:
                offset_info = ""
                if ev.get("char_offset_start") is not None:
                    offset_info = (
                        f' <span class="badge" style="background:#555;font-size:.65rem">'
                        f'chars {ev["char_offset_start"]}–{ev.get("char_offset_end","?")}</span>'
                    )
                ev_html += (
                    f'<div class="evidence-box">'
                    f'"{html.escape(ev.get("excerpt","")[:250])}"'
                    f'<br><span class="meta">— {html.escape(ev.get("author",""))}'
                    f' [{html.escape(ev.get("source_id",""))}]{offset_info}</span></div>'
                )

            st.markdown(
                f'<div class="card">'
                f'<h4>{html.escape(c.get("subject_name",""))} '
                f'→ <em>{c.get("relation","")}</em> → '
                f'{html.escape(c.get("object_name",""))}</h4>'
                f'<p>{status_badge(c.get("status","current"))} '
                f'Confidence: {c.get("confidence",0):.0%} · '
                f'{len(evidences)} evidence</p>'
                f'{ev_html}'
                f'<p class="meta">ID: {c.get("id","")} · '
                f'Valid: {c.get("valid_from","?")[:10] if c.get("valid_from") else "?"}'
                f' – {c.get("valid_until","?")[:10] if c.get("valid_until") else "now"}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# Page 4: Merges
# ═════════════════════════════════════════════════════════════════════════════
def page_merges(graph_data):
    st.markdown('<div class="page-header"><h2>Deduplication Audit Log</h2></div>',
                unsafe_allow_html=True)

    merges = graph_data.get("merge_log", [])
    if not merges:
        st.info("No merge records found.")
        return

    # Undo merge support
    st.caption("Reversible merges can be undone using the undo_merge() API "
               "with a MemoryStore instance.")

    # Summary
    merge_types = {}
    reasons = {}
    for m in merges:
        mt = m.get("merge_type", "?")
        merge_types[mt] = merge_types.get(mt, 0) + 1
        r = m.get("reason", "?").split("=")[0]
        reasons[r] = reasons.get(r, 0) + 1

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Merge types**")
        for t, cnt in sorted(merge_types.items(), key=lambda x: -x[1]):
            st.markdown(f"- `{t}`: **{cnt}**")
    with c2:
        st.markdown("**Reasons**")
        for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
            st.markdown(f"- `{r}`: **{cnt}**")

    st.divider()

    # Filter
    sel_type = st.selectbox(
        "Filter by type", ["all"] + sorted(merge_types.keys()), key="merge_type_sel"
    )
    filtered = (
        merges if sel_type == "all"
        else [m for m in merges if m.get("merge_type") == sel_type]
    )
    st.caption(f"Showing {len(filtered[:100])} / {len(filtered)} merges")

    for m in filtered[:100]:
        rev = "reversible" if m.get("reversible") else "permanent"
        has_snap = "has snapshots" if m.get("original_snapshots") else "no snapshots"
        st.markdown(
            f'<div class="card">'
            f'<h4>{m.get("merge_type","").title()} Merge</h4>'
            f'<p><b>Winner:</b> {html.escape(m.get("winner_id",""))} ← '
            f'<b>Loser:</b> {html.escape(m.get("loser_id",""))}</p>'
            f'<p class="meta">Reason: {html.escape(m.get("reason",""))} · '
            f'{rev} · {has_snap} · {m.get("timestamp","")[:19]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Page 5: Statistics
# ═════════════════════════════════════════════════════════════════════════════
def page_stats(graph_data):
    st.markdown('<div class="page-header"><h2>Graph Statistics</h2></div>',
                unsafe_allow_html=True)

    stats = load_stats()
    entities = graph_data.get("entities", [])
    claims = graph_data.get("claims", [])
    merges = graph_data.get("merge_log", [])

    # Top metrics row
    cols = st.columns(5)
    items = [
        ("Nodes", stats.get("nodes", len(entities))),
        ("Edges", stats.get("edges", len(claims))),
        ("Merges", len(merges)),
        ("Components", stats.get("connected_components", "?")),
        ("Density", f"{stats.get('density', 0):.4f}"),
    ]
    for col, (label, val) in zip(cols, items):
        col.markdown(
            f'<div class="metric-tile">'
            f'<div class="val">{val}</div>'
            f'<div class="lbl">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Breakdowns
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Entity types**")
        nt = stats.get("node_types", {})
        if not nt:
            nt = {}
            for e in entities:
                nt[e["type"]] = nt.get(e["type"], 0) + 1
        for t in sorted(nt, key=lambda x: -nt[x]):
            pct = nt[t] / max(sum(nt.values()), 1) * 100
            st.markdown(
                f'{type_badge(t)} **{nt[t]}** ({pct:.0f}%)',
                unsafe_allow_html=True,
            )

    with c2:
        st.markdown("**Relation types**")
        rt = stats.get("relation_types", {})
        if not rt:
            rt = {}
            for c in claims:
                r = c.get("relation", "?")
                rt[r] = rt.get(r, 0) + 1
        for r in sorted(rt, key=lambda x: -rt[x]):
            st.markdown(f"- `{r}`: **{rt[r]}**")

    with c3:
        st.markdown("**Status distribution**")
        sd = stats.get("status_distribution", {})
        if not sd:
            sd = {}
            for c in claims:
                s = c.get("status", "?")
                sd[s] = sd.get(s, 0) + 1
        for s in sorted(sd, key=lambda x: -sd[x]):
            st.markdown(
                f'{status_badge(s)} **{sd[s]}**',
                unsafe_allow_html=True,
            )

    # Top entities by degree
    st.divider()
    st.markdown("**Top entities by connection count**")
    degree = {}
    for c in claims:
        sid = c.get("subject_id", "")
        oid = c.get("object_id", "")
        degree[sid] = degree.get(sid, 0) + 1
        degree[oid] = degree.get(oid, 0) + 1

    ent_map = {e["id"]: e for e in entities}
    top = sorted(degree.items(), key=lambda x: -x[1])[:15]
    for eid, deg in top:
        e = ent_map.get(eid, {})
        st.markdown(
            f'{type_badge(e.get("type","?"))} **{html.escape(e.get("name","?"))}** '
            f'— {deg} connections',
            unsafe_allow_html=True,
        )

    # Evidence coverage
    total_ev = sum(len(c.get("evidence", [])) for c in claims)
    grounded = sum(1 for c in claims if c.get("evidence"))
    st.divider()
    cols2 = st.columns(3)
    cols2[0].metric("Total evidence records", total_ev)
    cols2[1].metric("Grounded claims", f"{grounded}/{len(claims)}")
    cols2[2].metric(
        "Coverage",
        f"{grounded/max(len(claims),1)*100:.0f}%",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    # Check data exists
    if not os.path.exists(GRAPH_PATH):
        st.error(
            f"No graph found at `{GRAPH_PATH}`.\n\n"
            "Run the pipeline first:\n```\npython run_pipeline.py\n```"
        )
        return

    graph_data = load_graph()
    page = render_sidebar(graph_data)

    # Route to the selected page
    if page == "Query":
        page_query(graph_data)
    elif page == "Graph":
        page_graph(graph_data)
    elif page == "Evidence":
        page_evidence(graph_data)
    elif page == "Merges":
        page_merges(graph_data)
    elif page == "Statistics":
        page_stats(graph_data)


if __name__ == "__main__":
    main()
