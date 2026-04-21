"""
state.py
--------
The shared state object that flows through every node in the LangGraph.

WHY A TYPED STATE?
In LangGraph, every node receives the current state dict and returns
a PARTIAL update to it. TypedDict gives you:
  1. Autocomplete in your editor
  2. Runtime clarity on what each node is expected to read/write
  3. A single place to see the full "memory" of one pipeline run

Think of this as the "working memory" of the multi-agent system.
Every agent reads what it needs and writes what it produces.
"""

from typing import TypedDict, Optional, List
from langchain_core.documents import Document


class AgentState(TypedDict):

    # ── Input ──────────────────────────────────────────────────────────────
    query: str
    paper_filter: Optional[str]
    k: int

    # ── Retriever output ───────────────────────────────────────────────────
    chunks: List[Document]
    retrieval_query: str

    # ── Answering Agent output ─────────────────────────────────────────────
    answer: str
    reasoning: str

    # ── Critic Agent output ────────────────────────────────────────────────
    critic_score: int
    critic_feedback: str
    hallucination_flags: List[str]
    refined_query: str
    verdict: str  # "PASS" | "RETRY" | "FAIL"

    # ── Loop control ──────────────────────────────────────────────────────
    retry_count: int
    max_retries: int