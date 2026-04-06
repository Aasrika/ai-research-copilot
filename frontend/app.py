import sys, os
from groq import Groq
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from backend.core.document_processor import build_or_update_index, load_index, get_indexed_papers
from backend.core.retriever import retrieve, format_context
from backend.core.basic_qa import answer_question

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Copilot",
    page_icon="🔬",
    layout="wide",
)

# ── Session State Init ─────────────────────────────────────────────────────
# Session state persists across Streamlit reruns — like a mini database
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

    # ── Paper Upload ───────────────────────────────────────────────────────
    st.subheader("📄 Upload Papers")
    uploaded = st.file_uploader(
        "Drop your PDFs here",
        type="pdf",
        accept_multiple_files=True,
        help="Each paper will be indexed and made searchable"
    )

    if uploaded and st.button("⚡ Index Papers", type="primary"):
        papers_dir = Path("data/papers")
        papers_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded files to disk (Streamlit gives us BytesIO objects)
        saved_paths = []
        for f in uploaded:
            save_path = papers_dir / f.name
            with open(save_path, "wb") as out:
                out.write(f.read())
            saved_paths.append(str(save_path))

        with st.spinner(f"Indexing {len(saved_paths)} paper(s)..."):
            store = build_or_update_index(saved_paths)
            st.session_state.vector_store  = store
            st.session_state.indexed_papers = get_indexed_papers()

        st.success(f"✅ {len(saved_paths)} paper(s) indexed!")

    # ── Load existing index ────────────────────────────────────────────────
    if st.session_state.vector_store is None:
        if st.button("📂 Load Existing Index"):
            try:
                st.session_state.vector_store  = load_index()
                st.session_state.indexed_papers = get_indexed_papers()
                st.success("Index loaded!")
            except Exception as e:
                st.error(f"No index found. Upload papers first. ({e})")

    # ── Indexed Papers List ────────────────────────────────────────────────
    if st.session_state.indexed_papers:
        st.divider()
        st.subheader("📚 Indexed Papers")
        for paper in st.session_state.indexed_papers:
            st.markdown(f"• `{paper}`")

    # ── Settings ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("⚙️ Settings")
    top_k = st.slider("Chunks retrieved (k)", 3, 10, 5,
                      help="More chunks = more context but slower + pricier")
    paper_filter = st.selectbox(
        "Search within paper",
        ["All papers"] + st.session_state.indexed_papers,
        help="Restrict retrieval to a specific paper"
    )
    paper_filter = None if paper_filter == "All papers" else paper_filter


# ── Main Area ──────────────────────────────────────────────────────────────
st.title("🔬 AI Research Copilot")
st.caption("Ask anything about your uploaded papers — every answer is grounded in citations.")

# Status banner
if st.session_state.vector_store is None:
    st.warning("⬅️ Upload and index papers using the sidebar to get started.")
    st.stop()  # Don't render the rest of the UI until index is loaded

# ── Query Mode Tabs ────────────────────────────────────────────────────────
tab_qa, tab_compare, tab_ideas = st.tabs([
    "💬 Ask a Question",
    "⚖️ Compare Papers",
    "💡 Research Ideas"  # placeholder for Phase 4
])

# ────────────────────────────────────────────────────────────────────────────
with tab_qa:
    st.subheader("Ask anything about your papers")

    # Quick-start prompts
    st.caption("💡 Try one of these:")
    cols = st.columns(4)
    quick_prompts = [
        "What is the main contribution?",
        "What datasets were used?",
        "What are the limitations?",
        "What methods were compared?",
    ]
    for col, prompt in zip(cols, quick_prompts):
        if col.button(prompt, key=f"qp_{prompt}"):
            st.session_state["prefill"] = prompt

    # Query input
    default_query = st.session_state.pop("prefill", "")
    query = st.text_area(
        "Your question:",
        value=default_query,
        height=80,
        placeholder="e.g. What are the key limitations of the proposed method?"
    )

    if st.button("🔍 Ask", type="primary", disabled=not query.strip()):
        with st.spinner("Retrieving and answering..."):
            response = answer_question(
                query,
                st.session_state.vector_store,
                k=top_k,
                paper_filter=paper_filter,
            )

        # Store in chat history
        st.session_state.chat_history.append(response)

    # ── Render chat history ────────────────────────────────────────────────
    for resp in reversed(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**❓ {resp['query']}**")

            # Answer box
            st.markdown(resp["answer"])

            # Citation expander
            with st.expander(f"📚 Sources ({resp['num_chunks']} chunks retrieved)"):
                for i, chunk in enumerate(resp["chunks_used"], 1):
                    m = chunk.metadata
                    st.markdown(
                        f"**[Source {i}]** `{m['paper_title']}` — "
                        f"Page {m['page']} | Section: `{m['section']}`"
                    )
                    # Show the actual chunk text the answer was based on
                    st.text_area(
                        f"Chunk {i} text",
                        chunk.page_content,
                        height=120,
                        key=f"chunk_{id(resp)}_{i}",
                        disabled=True,
                    )

            st.divider()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.subheader("Compare two papers")

    papers = st.session_state.indexed_papers
    if len(papers) < 2:
        st.info("Index at least 2 papers to use comparison mode.")
    else:
        col1, col2 = st.columns(2)
        paper_a = col1.selectbox("Paper A", papers, key="pa")
        paper_b = col2.selectbox("Paper B", [p for p in papers if p != paper_a], key="pb")

        compare_aspect = st.selectbox(
            "What to compare?",
            ["Methodology", "Results & performance", "Datasets used",
             "Limitations", "Key contributions", "Custom..."]
        )

        if compare_aspect == "Custom...":
            compare_aspect = st.text_input("Describe what to compare:")

        if st.button("⚖️ Compare", type="primary") and compare_aspect:
            query = f"Compare the {compare_aspect.lower()} between the two papers."

            with st.spinner("Retrieving from both papers..."):
                chunks_a = retrieve(query, st.session_state.vector_store,
                                    k=4, paper_filter=paper_a)
                chunks_b = retrieve(query, st.session_state.vector_store,
                                    k=4, paper_filter=paper_b)

            ctx_a = format_context(chunks_a)
            ctx_b = format_context(chunks_b)

            compare_prompt = f"""You are a scientific research assistant.

Compare the following aspect across two papers: **{compare_aspect}**

=== PAPER A: {paper_a} ===
{ctx_a}

=== PAPER B: {paper_b} ===
{ctx_b}

Provide a structured comparison covering:
1. Key similarities
2. Key differences  
3. Which paper's approach is stronger in this aspect, and why

Cite sources using [Paper A, Page X] and [Paper B, Page X] notation.
"""

            from groq import Groq
            import os

            client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": compare_prompt}],
                temperature=0
            )

            result_text = response.choices[0].message.content

            st.markdown(result_text)

            with st.expander("📚 Source Chunks"):
                st.markdown(f"**{paper_a}:**")
                for c in chunks_a:
                    st.caption(f"p.{c.metadata.get('page')} §{c.metadata.get('section')}: "
                               f"{c.page_content[:150]}...")

                st.markdown(f"**{paper_b}:**")
                for c in chunks_b:
                    st.caption(f"p.{c.metadata.get('page')} §{c.metadata.get('section')}: "
                               f"{c.page_content[:150]}...")
# ────────────────────────────────────────────────────────────────────────────
with tab_ideas:
    st.subheader("💡 Research Idea Generation")
    st.info("🔜 Coming in Phase 4 — this agent will surface limitations, "
            "research gaps, and future directions from your papers.")
    st.markdown("""
    **Preview of what's coming:**
    - *"What are the open research questions after this paper?"*
    - *"What experiments could strengthen these results?"*
    - *"What datasets could improve this benchmark?"*
    """)