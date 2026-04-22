"""
graph.py (Phase 5 - FINAL)
--------------------------
Upgrades:
  - Real per-stage timing using RunTimer
  - Automatic logging to JSONL
  - run_id returned for UI feedback
  - Error-safe logging (logs failures too)
"""

from functools import partial
import time

from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS

from agents.state import AgentState
from agents.retriever_agent import retriever_node
from agents.answering_agent import answering_node
from agents.critic_agent import critic_node, route_after_critic

from evaluation.logger import RunRecord, RunTimer, log_run


# ─────────────────────────────────────────────────────────────────────────────
# TIMING WRAPPER (KEY FIX)
# ─────────────────────────────────────────────────────────────────────────────

def timed_node(fn, name: str, timer: RunTimer):
    """Wrap a node to measure execution time."""
    def wrapper(state):
        with timer.section(name):
            return fn(state)
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(vector_store: FAISS, timer: RunTimer) -> StateGraph:
    graph = StateGraph(AgentState)

    # Nodes with timing
    graph.add_node(
        "retriever",
        timed_node(partial(retriever_node, vector_store=vector_store), "retrieval", timer)
    )

    graph.add_node(
        "answering",
        timed_node(answering_node, "generation", timer)
    )

    graph.add_node(
        "critic",
        timed_node(critic_node, "critic", timer)
    )

    # Retry handler
    def retry_handler(state: AgentState) -> dict:
        return {
            "retry_count":     state.get("retry_count", 0) + 1,
            "retrieval_query": state.get("refined_query") or state["query"],
            "answer":          "",
            "reasoning":       "",
        }

    graph.add_node("retry_handler", retry_handler)

    # Edges
    graph.add_edge("retriever",     "answering")
    graph.add_edge("answering",     "critic")
    graph.add_edge("retry_handler", "retriever")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "end": END,
            "retry": "retry_handler",
        }
    )

    graph.set_entry_point("retriever")

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    query:        str,
    vector_store: FAISS,
    k:            int  = 5,
    paper_filter: str  = None,
    max_retries:  int  = 2,
) -> AgentState:

    timer = RunTimer()
    compiled = build_graph(vector_store, timer)

    initial_state: AgentState = {
        "query":           query,
        "paper_filter":    paper_filter,
        "k":               k,

        "chunks":          [],
        "retrieval_query": query,

        "answer":          "",
        "reasoning":       "",

        "critic_score":        0,
        "critic_feedback":     "",
        "hallucination_flags": [],

        "refined_query": "",
        "verdict":       "",

        "retry_count":  0,
        "max_retries":  max_retries,
    }

    print(f"\n{'='*60}")
    print(f"🚀 Pipeline | {query[:70]}...")
    print(f"{'='*60}")

    error_msg = None

    try:
        final_state = compiled.invoke(initial_state)

    except Exception as e:
        # Capture failure for logging
        final_state = initial_state
        error_msg = str(e)

    # ── Extract retrieval info ─────────────────────────────────────────────
    chunks   = final_state.get("chunks", [])
    sections = list({c.metadata.get("section", "body") for c in chunks})
    papers   = list({c.metadata.get("paper_title", "") for c in chunks})

    # ── Build log record ───────────────────────────────────────────────────
    record = RunRecord(
        pipeline_type        = "qa",
        query                = query,
        paper_filter         = paper_filter,
        k                    = k,

        critic_score         = final_state.get("critic_score", 0),
        verdict              = final_state.get("verdict", ""),
        hallucination_flags  = final_state.get("hallucination_flags", []),

        retry_count          = final_state.get("retry_count", 0),
        answer_length        = len(final_state.get("answer", "")),

        latency_total        = timer.total(),
        latency_retrieval    = timer["retrieval"],
        latency_generation   = timer["generation"],
        latency_critic       = timer["critic"],

        num_chunks           = len(chunks),
        sections_retrieved   = sections,
        papers_retrieved     = papers,
    )

    # Add error if occurred
    if error_msg:
        record.verdict = "ERROR"
        record.hallucination_flags.append("pipeline_failure")

    log_run(record)

    # Attach run_id for UI feedback
    final_state["run_id"] = record.run_id

    print(f"\n{'='*60}")
    print(f"✅ Done")
    print(f"   Score:   {record.critic_score}/10")
    print(f"   Retries: {record.retry_count}")
    print(f"   Time:    {record.latency_total}s")
    print(f"{'='*60}\n")

    return final_state