import json, re, os
from functools import partial
from typing import TypedDict, Optional

from groq import Groq
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from core.retriever import retrieve, format_context
from core.config import COMPARISON_MODEL
from core.token_tracking import extract_usage, estimate_cost


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
    full_comparison: bool  # True = compare across all dimensions; False = aspect focuses the output, not just retrieval

    chunks_a: list[Document]
    chunks_b: list[Document]

    raw_comparison: str
    structured: dict
    verdict: str

    token_usage: dict
    estimated_cost_usd: float


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

# Used instead of an aspect-specific query when full_comparison=True, so
# retrieval isn't skewed toward any single dimension.
FULL_COMPARISON_QUERY = (
    "methodology approach results performance datasets contributions "
    "limitations problem motivation"
)


def _build_query(aspect, custom, full_comparison=False):
    if full_comparison:
        return FULL_COMPARISON_QUERY
    return custom if custom else ASPECT_QUERIES.get(aspect, aspect)


def fetch_a(state: ComparisonState, vector_store: FAISS) -> dict:
    q = _build_query(state["aspect"], state.get("custom_query", ""), state.get("full_comparison", False))

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
    q = _build_query(state["aspect"], state.get("custom_query", ""), state.get("full_comparison", False))

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

# Full comparison: no restriction on scope — cover whatever dimensions the
# evidence supports.
FULL_COMPARISON_INSTRUCTION = """Compare these two papers across ALL major dimensions relevant to a thorough research comparison: methodology, results & performance, datasets used, limitations, key contributions, and problem framing.

For each dimension where you find meaningful, source-grounded evidence in BOTH papers, add one object to the "differences" array. Skip a dimension entirely if you don't have clear evidence for it rather than inventing content."""

# Focused (a specific preset or custom aspect): the output must be
# constrained to exactly this one aspect, not a full comparison. The
# "EXACTLY ONE" + explicit anti-padding line exist because a 70B model will
# happily volunteer extra dimensions if only told what to focus on without
# also being told what NOT to do.
FOCUSED_INSTRUCTION = """You must compare these two papers ONLY on the following aspect. Do NOT mention, discuss, or compare any other aspect, dimension, or topic, even if the source material touches on other things.

Aspect to compare: {aspect}

The "differences" array in your JSON output MUST contain EXACTLY ONE object, covering only this aspect. Do not add a second or third entry for any other topic — one entry only, no matter how much other material is available."""

PROMPT = """You are a senior researcher comparing two academic papers.

{instruction}

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

    if state.get("full_comparison"):
        instruction = FULL_COMPARISON_INSTRUCTION
    else:
        aspect_label = state.get("custom_query") or state["aspect"]
        instruction = FOCUSED_INSTRUCTION.format(aspect=aspect_label)

    prompt = PROMPT.format(
        instruction=instruction,
        paper_a=state["paper_a"],
        paper_b=state["paper_b"],
        context_a=ctx_a,
        context_b=ctx_b
    )

    response = client.chat.completions.create(
        model=COMPARISON_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw_comparison = response.choices[0].message.content
    usage = extract_usage(response, prompt, raw_comparison or "")

    return {
        "raw_comparison": raw_comparison,
        "token_usage": usage,
        "estimated_cost_usd": estimate_cost(usage, COMPARISON_MODEL),
    }


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

    # Programmatic safety net: the FOCUSED_INSTRUCTION prompt tells the model
    # to return exactly one "differences" entry, but prompt compliance alone
    # isn't guaranteed — truncate here so the UI's one-row-per-aspect
    # guarantee holds regardless of what the model actually returned.
    differences = structured.get("differences")
    if not state.get("full_comparison") and isinstance(differences, list) and len(differences) > 1:
        print(f"⚠️ Focused comparison returned {len(differences)} differences — truncating to 1")
        structured["differences"] = differences[:1]

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
def run_comparison(paper_a, paper_b, aspect, vector_store, custom_query="", full_comparison=False):

    print("\n⚖️ Running comparison...")

    graph = build_graph(vector_store)

    state = {
        "paper_a": paper_a,
        "paper_b": paper_b,
        "aspect": aspect,
        "custom_query": custom_query,
        "full_comparison": full_comparison,
        "chunks_a": [],
        "chunks_b": [],
        "raw_comparison": "",
        "structured": {},
        "verdict": "",
        "token_usage": {},
        "estimated_cost_usd": 0.0,
    }

    return graph.invoke(state)