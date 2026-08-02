import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from dotenv import load_dotenv

from core import session_manager
from agents.graph import run_pipeline
from agents.idea_agent import generate_research_ideas
from agents.comparison_agent import run_comparison, ASPECT_QUERIES
from evaluation.logger import RunRecord, log_run

load_dotenv()


def render_session_banner():
    """Subtle, always-visible reminder of which session's data is on screen.
    Duplicated identically across app.py / Evaluation.py / Critique.py —
    every page in this app is already independent (own sys.path setup, no
    shared layout module), so a 4-line function isn't worth a new shared
    module just for this."""
    active_id = st.session_state.get("active_session_id")
    session = session_manager.get_session(active_id) if active_id else None
    if not session:
        st.caption("📁 No session selected — please create or select one from the main page sidebar.")
        return
    st.caption(f"📁 Active session: **{session['name']}**")


st.markdown("""
<style>
.stButton>button {
    border-radius: 10px;
    background: linear-gradient(90deg, #7C3AED, #06B6D4);
    color: white;
    border: none;
}

.stTextInput>div>div>input {
    border-radius: 10px;
}

.block-container {
    padding-top: 2rem;
}

div[data-testid="stMetric"] {
    background-color: #111827;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)




# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Research Copilot", page_icon="🔬", layout="wide")

# ── Session State ──────────────────────────────────────────────────────────
# chat_histories / idea_results_by_session / compare_results_by_session are
# keyed by session_id (dict[session_id, value]) so switching sessions shows
# each session's own history instead of leaking into another's. vector_store
# and indexed_papers stay flat — they're already fully refreshed on every
# session switch below, so namespacing them would be redundant.
for key, default in [
    ("active_session_id", None),
    ("vector_store", None),
    ("indexed_papers", []),
    ("chat_histories", {}),
    ("idea_results_by_session", {}),
    ("compare_results_by_session", {}),
    ("uploader_key_version", 0),
    ("just_indexed_msg", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


@st.dialog("Delete session?")
def _confirm_delete_session(session_id: str, session_name: str):
    st.warning(
        f"Delete session **'{session_name}'** and all its papers, indexes, "
        f"and critique reports? This cannot be undone."
    )
    col_cancel, col_confirm = st.columns(2)

    if col_cancel.button("Cancel", use_container_width=True):
        st.rerun()

    if col_confirm.button("🗑️ Delete", type="primary", use_container_width=True):
        session_manager.delete_session(session_id)

        # Clean up every session-scoped state dict so a deleted session's
        # data doesn't linger as a memory leak. critique_reports lives on
        # the Critique page and may not exist yet if that page was never
        # visited this browser session — .get(..., {}) handles that safely.
        for state_key in (
            "chat_histories", "idea_results_by_session",
            "compare_results_by_session", "critique_reports",
        ):
            st.session_state.get(state_key, {}).pop(session_id, None)

        st.session_state.active_session_id = None
        st.session_state.vector_store = None
        st.session_state.indexed_papers = []
        st.rerun()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Copilot")
    st.caption("Multi-agent RAG for research papers")
    st.divider()

    st.subheader("🗂️ Session")
    sessions = session_manager.list_sessions()
    session_names = {s["id"]: s["name"] for s in sessions}

    selected_id = None
    if sessions:
        ids = list(session_names.keys())
        current = st.session_state.active_session_id
        default_index = ids.index(current) if current in ids else 0
        selected_id = st.selectbox(
            "Active session",
            ids,
            index=default_index,
            format_func=lambda sid: session_names[sid],
        )
    else:
        st.info("No sessions yet — create one below.")

    # Only reload the store when the active session actually changes, so we
    # don't re-hit disk on every widget interaction/rerun.
    if selected_id != st.session_state.active_session_id:
        st.session_state.active_session_id = selected_id
        st.session_state.vector_store = (
            session_manager.load_literature_store(selected_id) if selected_id else None
        )
        st.session_state.indexed_papers = (
            session_manager.get_indexed_literature(selected_id) if selected_id else []
        )

    active_id = st.session_state.active_session_id

    if active_id and st.button("🗑️ Delete session"):
        _confirm_delete_session(active_id, session_names.get(active_id, active_id))

    with st.expander("➕ New session"):
        new_name = st.text_input("Session name", key="new_session_name")
        if st.button("Create session", disabled=not new_name.strip()):
            record = session_manager.create_session(new_name)
            st.session_state.active_session_id = record["id"]
            st.session_state.vector_store = None
            st.session_state.indexed_papers = []
            st.rerun()

    if active_id:
        st.divider()
        st.caption(f"Active session: **{session_names.get(active_id, active_id)}**")

        if st.session_state.just_indexed_msg:
            st.success(st.session_state.just_indexed_msg)
            st.session_state.just_indexed_msg = None

        uploaded = st.file_uploader(
            "📄 Upload literature PDFs", type="pdf", accept_multiple_files=True,
            key=f"literature_uploader_{st.session_state.uploader_key_version}",
        )

        if uploaded and st.button("⚡ Index Papers", type="primary"):
            with st.spinner("Indexing papers..."):
                files = [(f.name, f.read()) for f in uploaded]
                store = session_manager.add_literature(active_id, files)
                st.session_state.vector_store = store
                st.session_state.indexed_papers = session_manager.get_indexed_literature(active_id)

            # Bump the uploader's key so it remounts empty, and rerun so the
            # cleared uploader (and the "Papers in this session" update) show
            # immediately instead of leaving the just-indexed files sitting
            # in the upload area.
            st.session_state.just_indexed_msg = f"✅ Indexed {len(uploaded)} paper(s)!"
            st.session_state.uploader_key_version += 1
            st.rerun()

        if st.session_state.indexed_papers:
            st.divider()
            st.subheader("📚 Papers in this session")
            for p in st.session_state.indexed_papers:
                st.markdown(f"• `{p}`")
    else:
        st.warning("Create or select a session to begin.")

    st.divider()
    top_k = st.slider("Chunks (k)", 3, 10, 5)
    paper_filter = st.selectbox(
        "Search within paper",
        ["All papers"] + st.session_state.indexed_papers
    )
    paper_filter = None if paper_filter == "All papers" else paper_filter

# ── Main ──────────────────────────────────────────────────────────────────
st.title("🔬 AI Research Copilot")
st.caption("Multi-agent RAG + Self-correction + Research reasoning")
render_session_banner()

if st.session_state.active_session_id is None:
    st.warning("⬅️ Create or select a session to start")
    st.stop()

if st.session_state.vector_store is None:
    st.warning("⬅️ Upload literature papers into this session to start")
    st.stop()

active_id = st.session_state.active_session_id

tab_qa, tab_compare, tab_ideas = st.tabs([
    "💬 Ask",
    "⚖️ Compare",
    "💡 Ideas"
])

# ════════════════════════════════════════════════════════════════════════════
# 💬 TAB 1 — Q&A
# ════════════════════════════════════════════════════════════════════════════
with tab_qa:
    st.subheader("Ask anything")

    # Quick prompts
    cols = st.columns(4)
    prompts = [
        "What is the main contribution?",
        "What datasets were used?",
        "What are the limitations?",
        "How does this compare to prior work?",
    ]
    for col, p in zip(cols, prompts):
        if col.button(p):
            st.session_state["qa_query_input"] = p

    # A stable, explicit key — not value=st.session_state.pop(...) — is what
    # makes the quick-prompt buttons actually work. Without a key, Streamlit
    # re-applies whatever `value=` is passed on EVERY rerun; since "prefill"
    # was a one-shot pop(), the very next rerun (clicking Ask) saw value=""
    # again and silently reset the query to empty — the exact query became
    # "" right before run_pipeline was called, hence the blank "You:" line
    # and the unrelated-looking answer (an empty-string similarity search).
    if "qa_query_input" not in st.session_state:
        st.session_state["qa_query_input"] = ""

    query = st.text_area("Your question:", key="qa_query_input")

    if st.button("🔍 Ask", type="primary", disabled=not query.strip()):
        with st.spinner("Running multi-agent pipeline..."):
            state = run_pipeline(
                query=query,
                vector_store=st.session_state.vector_store,
                k=top_k,
                paper_filter=paper_filter,
                max_retries=2,
                session_id=active_id,
            )

        st.session_state.chat_histories.setdefault(active_id, []).append(state)

    for resp in reversed(st.session_state.chat_histories.get(active_id, [])):
        st.markdown(f"**🧑 You:** {resp['query']}")
        st.divider()

        score = resp["critic_score"]
        if score >= 8:
            st.success(f"Score: {score}/10")
        elif score >= 6:
            st.warning(f"Score: {score}/10")
        else:
            st.error(f"Score: {score}/10")

        st.info(f"Verdict: {resp['verdict']}")

        if resp["retry_count"] > 0:
            st.warning(f"Retries: {resp['retry_count']}")

        st.markdown(resp["answer"])

        if resp.get("hallucination_flags"):
            with st.expander(f"⚠️ Hallucination flags ({len(resp['hallucination_flags'])})"):
                for flag in resp["hallucination_flags"]:
                    st.caption(f"• {flag}")

        if resp.get("vagueness_flags"):
            with st.expander(f"💤 Vagueness flags ({len(resp['vagueness_flags'])})"):
                for flag in resp["vagueness_flags"]:
                    st.caption(f"• {flag}")

        with st.expander("📚 Sources"):
            for c in resp["chunks"]:
                st.caption(f"{c.metadata['paper_title']} p.{c.metadata['page']}")

        st.divider()

# ════════════════════════════════════════════════════════════════════════════
# ⚖️ TAB 2 — COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Compare Papers")

    papers = st.session_state.indexed_papers

    if len(papers) < 2:
        st.info("Upload at least 2 papers.")
    else:
        col1, col2 = st.columns(2)
        a = col1.selectbox("Paper A", papers)
        b = col2.selectbox("Paper B", [p for p in papers if p != a])

        aspect = st.selectbox("Aspect", list(ASPECT_QUERIES.keys()))

        if st.button("⚖️ Compare"):
            with st.spinner("Running comparison..."):
                result = run_comparison(
                    paper_a=a,
                    paper_b=b,
                    aspect=aspect,
                    vector_store=st.session_state.vector_store,
                )
                st.session_state.compare_results_by_session[active_id] = result

                # run_comparison doesn't self-log (unlike run_pipeline) — this
                # mirrors backend/api/main.py's /compare route, which builds
                # its own RunRecord the same way. Streamlit's Compare tab
                # previously logged nothing at all for this pipeline type.
                log_run(RunRecord(
                    pipeline_type = "comparison",
                    session_id    = active_id,
                    query         = f"Compare {a} vs {b}: {aspect}",
                    num_chunks    = len(result.get("chunks_a", [])) + len(result.get("chunks_b", [])),
                ))

    res = st.session_state.compare_results_by_session.get(active_id)
    if res and res.get("structured"):
        s = res["structured"]

        st.success(s.get("verdict", ""))
        st.markdown("### 🧠 Synthesis")
        st.write(s.get("synthesis", ""))

        st.markdown("### 🔍 Differences")
        for d in s.get("differences", []):
            st.write(f"**{d['aspect']}**")
            st.write(f"A: {d['paper_a']}")
            st.write(f"B: {d['paper_b']}")
            st.divider()

# ════════════════════════════════════════════════════════════════════════════
# 💡 TAB 3 — IDEAS
# ════════════════════════════════════════════════════════════════════════════
with tab_ideas:
    st.subheader("Research Ideas")

    papers = st.session_state.indexed_papers

    if not papers:
        st.info("Upload a paper first.")
    else:
        paper = st.selectbox("Paper", papers)
        focus = st.text_input("Focus area (optional)")

        if st.button("💡 Generate Ideas"):
            with st.spinner("Generating ideas..."):
                ideas = generate_research_ideas(
                    st.session_state.vector_store,
                    paper_filter=paper,
                    focus_area=focus,
                )
                st.session_state.idea_results_by_session[active_id] = ideas

                # generate_research_ideas doesn't self-log (unlike run_pipeline)
                # — this mirrors backend/api/main.py's /ideas route. Streamlit's
                # Ideas tab previously logged nothing at all for this type.
                log_run(RunRecord(
                    pipeline_type = "ideas",
                    session_id    = active_id,
                    query         = f"Ideas for {paper}: {focus}",
                    paper_filter  = paper,
                    num_chunks    = len(ideas.get("chunks_used", [])),
                ))

    ideas = st.session_state.idea_results_by_session.get(active_id)
    if ideas:
        st.info(ideas["summary"])

        st.markdown("### ⚠️ Limitations")
        for l in ideas["explicit_limitations"]:
            st.write("-", l["finding"])

        st.markdown("### ❓ Questions")
        for q in ideas["open_questions"]:
            st.write("-", q["question"])

        st.markdown("### 🚀 Ideas")
        for e in ideas["experiment_ideas"]:
            st.write(f"**{e['title']}**")
            st.write(e["description"])
            st.caption(f"{e['difficulty']} difficulty")
            st.divider()