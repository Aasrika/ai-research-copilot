import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from core.document_processor import build_or_update_index, load_index, get_indexed_papers
from agents.graph import run_pipeline
from agents.idea_agent import generate_research_ideas
from agents.comparison_agent import run_comparison, ASPECT_QUERIES

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Research Copilot", page_icon="🔬", layout="wide")

# ── Session State ──────────────────────────────────────────────────────────
for key, default in [
    ("vector_store", None),
    ("chat_history", []),
    ("indexed_papers", get_indexed_papers()),
    ("idea_results", None),
    ("compare_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Copilot")
    st.caption("Multi-agent RAG for research papers")
    st.divider()

    uploaded = st.file_uploader("📄 Upload PDFs", type="pdf", accept_multiple_files=True)

    if uploaded and st.button("⚡ Index Papers", type="primary"):
        papers_dir = Path("data/papers")
        papers_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for f in uploaded:
            p = papers_dir / f.name
            p.write_bytes(f.read())
            paths.append(str(p))

        with st.spinner("Indexing papers..."):
            store = build_or_update_index(paths)
            st.session_state.vector_store = store
            st.session_state.indexed_papers = get_indexed_papers()

        st.success("✅ Papers indexed!")

    if st.session_state.vector_store is None:
        if st.button("📂 Load Existing Index"):
            st.session_state.vector_store = load_index()
            st.session_state.indexed_papers = get_indexed_papers()
            st.success("Index loaded!")

    if st.session_state.indexed_papers:
        st.divider()
        st.subheader("📚 Papers")
        for p in st.session_state.indexed_papers:
            st.markdown(f"• `{p}`")

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

if st.session_state.vector_store is None:
    st.warning("⬅️ Upload papers to start")
    st.stop()

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
            st.session_state["prefill"] = p

    query = st.text_area("Your question:", value=st.session_state.pop("prefill", ""))

    if st.button("🔍 Ask", type="primary", disabled=not query.strip()):
        with st.spinner("Running multi-agent pipeline..."):
            state = run_pipeline(
                query=query,
                vector_store=st.session_state.vector_store,
                k=top_k,
                paper_filter=paper_filter,
                max_retries=2,
            )

        st.session_state.chat_history.append(state)

    for resp in reversed(st.session_state.chat_history):
        st.markdown(f"**❓ {resp['query']}**")

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
                st.session_state.compare_result = result

    res = st.session_state.compare_result
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
                st.session_state.idea_results = ideas

    ideas = st.session_state.idea_results
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