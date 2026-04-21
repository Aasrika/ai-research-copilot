import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from core.document_processor import build_or_update_index, load_index, get_indexed_papers
from core.retriever import retrieve, format_context
from agents.graph import run_pipeline


# ── Helper ────────────────────────────────────────────────────────────────
def _sources_from_chunks(chunks) -> list[dict]:
    seen, sources = set(), []
    for c in chunks:
        key = (c.metadata["paper_title"], c.metadata["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "paper":   c.metadata["paper_title"],
                "page":    c.metadata["page"],
                "section": c.metadata["section"],
            })
    return sources


load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Research Copilot", page_icon="🔬", layout="wide")

# ── Session State ──────────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_papers" not in st.session_state:
    st.session_state.indexed_papers = get_indexed_papers()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Copilot")
    st.caption("Multi-agent RAG for research papers")
    st.divider()

    st.subheader("📄 Upload Papers")
    uploaded = st.file_uploader("Drop your PDFs here", type="pdf", accept_multiple_files=True)

    if uploaded and st.button("⚡ Index Papers", type="primary"):
        papers_dir = Path("data/papers")
        papers_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for f in uploaded:
            save_path = papers_dir / f.name
            with open(save_path, "wb") as out:
                out.write(f.read())
            saved_paths.append(str(save_path))

        with st.spinner(f"Indexing {len(saved_paths)} paper(s)..."):
            store = build_or_update_index(saved_paths)
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
        st.subheader("📚 Indexed Papers")
        for paper in st.session_state.indexed_papers:
            st.markdown(f"• `{paper}`")

    st.divider()
    st.subheader("⚙️ Settings")
    top_k = st.slider("Chunks retrieved (k)", 3, 10, 5)
    paper_filter = st.selectbox("Search within paper", ["All papers"] + st.session_state.indexed_papers)
    paper_filter = None if paper_filter == "All papers" else paper_filter

# ── Main ──────────────────────────────────────────────────────────────────
st.title("🔬 AI Research Copilot")
st.caption("Multi-agent RAG with self-correction and hallucination detection")

if st.session_state.vector_store is None:
    st.warning("⬅️ Upload and index papers to begin.")
    st.stop()

tab_qa, tab_compare, tab_ideas = st.tabs(["💬 Ask", "⚖️ Compare", "💡 Ideas"])

# ───────────────────────────────────────────────────────────────────────────
with tab_qa:
    st.subheader("Ask anything about your papers")

    # ── Quick prompts (VERY IMPORTANT for UX) ─────────────────────────
    st.caption("💡 Try one of these:")
    cols = st.columns(4)

    quick_prompts = [
        "What is the main contribution?",
        "What datasets were used?",
        "What are the limitations?",
        "How does this compare to prior work?",
    ]

    for col, prompt in zip(cols, quick_prompts):
        if col.button(prompt, key=f"qp_{prompt}"):
            st.session_state["prefill"] = prompt

    # ── Prefill logic ────────────────────────────────────────────────
    default_query = st.session_state.pop("prefill", "")

    query = st.text_area(
        "Your question:",
        value=default_query,
        height=80,
        placeholder="e.g. What are the key limitations of the proposed method?"
    )

    # ── Ask button (your Phase 3 pipeline) ───────────────────────────
    if st.button("🔍 Ask", type="primary", disabled=not query.strip()):
        with st.spinner("Running multi-agent pipeline..."):

            progress = st.empty()
            progress.markdown("🔎 **Retriever Agent** — fetching relevant chunks...")

            state = run_pipeline(
                query=query,
                vector_store=st.session_state.vector_store,
                k=top_k,
                paper_filter=paper_filter,
                max_retries=2,
            )

            progress.empty()

        response = {
            "query": query,
            "answer": state["answer"],
            "reasoning": state["reasoning"],
            "chunks_used": state["chunks"],
            "num_chunks": len(state["chunks"]),
            "critic_score": state["critic_score"],
            "critic_feedback": state["critic_feedback"],
            "hallucination_flags": state["hallucination_flags"],
            "retry_count": state["retry_count"],
            "verdict": state["verdict"],
        }

        st.session_state.chat_history.append(response)
    # ── Render responses ───────────────────────────────────────────────────
    for resp in reversed(st.session_state.chat_history):
        st.markdown(f"**❓ {resp['query']}**")

        score = resp["critic_score"]
        verdict = resp["verdict"]
        retries = resp["retry_count"]

        col1, col2, col3 = st.columns(3)

        if score >= 8:
            col1.success(f"Score: {score}/10")
        elif score >= 6:
            col1.warning(f"Score: {score}/10")
        else:
            col1.error(f"Score: {score}/10")

        col2.info(f"Verdict: {verdict}")

        if retries > 0:
            col3.warning(f"Retries: {retries}")

        if resp["hallucination_flags"]:
            with st.expander("⚠️ Hallucinations detected"):
                for f in resp["hallucination_flags"]:
                    st.markdown(f"- {f}")

        st.markdown(resp["answer"])

        if resp["reasoning"]:
            with st.expander("🧠 Reasoning"):
                st.markdown(resp["reasoning"])

        if resp["critic_feedback"]:
            with st.expander("🔍 Critic Feedback"):
                st.info(resp["critic_feedback"])

        with st.expander(f"📚 Sources ({resp['num_chunks']})"):
            for i, chunk in enumerate(resp["chunks_used"], 1):
                m = chunk.metadata
                st.markdown(f"**Source {i}** — {m['paper_title']} (Page {m['page']})")
                st.text_area(
                    f"chunk_{id(resp)}_{i}",
                    chunk.page_content,
                    height=120,
                    disabled=True,
                )

        st.divider()

# ───────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.subheader("Compare Papers")
    st.info("Same as Phase 2 (unchanged)")

# ───────────────────────────────────────────────────────────────────────────
with tab_ideas:
    st.subheader("💡 Coming Next")