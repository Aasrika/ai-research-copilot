import json, re, os
from functools import partial
from typing import TypedDict, Optional

from groq import Groq
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from core.retriever import retrieve, format_context


# ─────────────────────────────────────────────────────────────────────────────
# GROQ CLIENT
# ─────────────────────────────────────────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────
class ComparisonState(TypedDict):
    paper_a: str
    paper_b: str
    aspect: str
    custom_query: Optional[str]

    chunks_a: list[Document]
    chunks_b: list[Document]

    raw_comparison: str
    structured: dict
    verdict: str


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────
ASPECT_QUERIES = {
    "Methodology": "methodology approach algorithm architecture",
    "Results & Performance": "results accuracy benchmark metrics",
    "Datasets Used": "dataset training data benchmark",
    "Limitations": "limitations weaknesses future work",
    "Key Contributions": "main contribution novelty",
    "Problem Framing": "problem motivation research question",
}

def _build_query(aspect, custom):
    return custom if custom else ASPECT_QUERIES.get(aspect, aspect)


def fetch_a(state: ComparisonState, vector_store: FAISS) -> dict:
    q = _build_query(state["aspect"], state.get("custom_query", ""))

    chunks = retrieve(q, vector_store, k=7, paper_filter=state["paper_a"])

    # 🔥 Fallback logic
    if len(chunks) < 3:
        print("⚠️ Low retrieval for Paper A — broadening query...")
        chunks = retrieve(
            state["aspect"],
            vector_store,
            k=7,
            paper_filter=state["paper_a"]
        )

    print(f"  📄 Paper A ({state['paper_a']}): {len(chunks)} chunks")
    return {"chunks_a": chunks}


def fetch_b(state: ComparisonState, vector_store: FAISS) -> dict:
    q = _build_query(state["aspect"], state.get("custom_query", ""))

    # First retrieval attempt
    chunks = retrieve(q, vector_store, k=7, paper_filter=state["paper_b"])

    # 🔥 Fallback logic (ADD THIS)
    if len(chunks) < 3:
        print("⚠️ Low retrieval for Paper B — broadening query...")
        chunks = retrieve(
            state["aspect"],   # broader query
            vector_store,
            k=7,
            paper_filter=state["paper_b"]
        )

    print(f"  📄 Paper B ({state['paper_b']}): {len(chunks)} chunks")
    return {"chunks_b": chunks}


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────
PROMPT = """You are a senior researcher comparing two academic papers.

Aspect: {aspect}

{custom_line}

=== PAPER A: {paper_a} ===
{context_a}

=== PAPER B: {paper_b} ===
{context_b}

Return ONLY JSON:

{{
  "paper_a_summary": "string",
  "paper_b_summary": "string",
  "similarities": ["string"],
  "differences": [
    {{"aspect": "string", "paper_a": "string", "paper_b": "string"}}
  ],
  "paper_a_strengths": ["string"],
  "paper_b_strengths": ["string"],
  "verdict": "string",
  "synthesis": "string"
}}
"""


def synthesize(state: ComparisonState):

    ctx_a = format_context(state["chunks_a"])
    ctx_b = format_context(state["chunks_b"])

    custom_line = f"Question: {state['custom_query']}" if state.get("custom_query") else ""

    prompt = PROMPT.format(
        aspect=state["aspect"],
        custom_line=custom_line,
        paper_a=state["paper_a"],
        paper_b=state["paper_b"],
        context_a=ctx_a,
        context_b=ctx_b
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return {"raw_comparison": response.choices[0].message.content}


# ─────────────────────────────────────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────────────────────────────────────
def structure(state: ComparisonState):

    raw = state["raw_comparison"]

    try:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        structured = json.loads(cleaned)

    except Exception:
        print("⚠️ Parse failed — fallback")
        structured = {"raw": raw, "parse_error": True}

    verdict = structured.get("verdict", "")

    print(f"✅ Verdict: {verdict[:80]}")

    return {
        "structured": structured,
        "verdict": verdict
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────────────────────────────────────
def build_graph(vector_store: FAISS):

    g = StateGraph(ComparisonState)

    g.add_node("fetch_a", partial(fetch_a, vector_store=vector_store))
    g.add_node("fetch_b", partial(fetch_b, vector_store=vector_store))
    g.add_node("synthesize", synthesize)
    g.add_node("structure", structure)

    g.add_edge("fetch_a", "fetch_b")
    g.add_edge("fetch_b", "synthesize")
    g.add_edge("synthesize", "structure")
    g.add_edge("structure", END)

    g.set_entry_point("fetch_a")

    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_comparison(paper_a, paper_b, aspect, vector_store, custom_query=""):

    print("\n⚖️ Running comparison...")

    graph = build_graph(vector_store)

    state = {
        "paper_a": paper_a,
        "paper_b": paper_b,
        "aspect": aspect,
        "custom_query": custom_query,
        "chunks_a": [],
        "chunks_b": [],
        "raw_comparison": "",
        "structured": {},
        "verdict": "",
    }

    return graph.invoke(state)