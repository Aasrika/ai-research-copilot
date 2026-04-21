"""
graph.py
--------
Assembles the multi-agent LangGraph pipeline.

LANGGRAPH CONCEPTS USED HERE:

  StateGraph:      A graph whose nodes share a typed state dict.

  add_node():      Register a function as a named node.
                   The function signature must be: fn(state) -> partial_state

  add_edge():      Unconditional: A always goes to B.

  add_conditional_edges():
                   After node A, call router_fn(state) → returns a string
                   → that string maps to the next node name.
                   This is how the critic's verdict routes to END or RETRY.

  compile():       Freezes the graph into a runnable object.
                   After compile, you call graph.invoke(initial_state).

  functools.partial():
                   Some nodes need access to vector_store which isn't in state.
                   We use partial() to "bake in" the vector_store at graph
                   creation time, so the node still has signature fn(state).
"""

from functools import partial
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS

from agents.state import AgentState
from agents.retriever_agent import retriever_node
from agents.answering_agent import answering_node
from agents.critic_agent    import critic_node, route_after_critic


def build_graph(vector_store: FAISS) -> StateGraph:
    """
    Build and compile the multi-agent graph.

    Graph structure:
      START → retriever → answering → critic → (conditional) → END
                              ↑                      |
                              └──── retry loop ───────┘
                                  (critic updates retry_count
                                   and refined_query before looping)
    """

    # ── 1. Create the graph with our state schema ──────────────────────────
    graph = StateGraph(AgentState)

    # ── 2. Register nodes ──────────────────────────────────────────────────
    # retriever_node needs vector_store → bake it in with partial()
    graph.add_node("retriever",  partial(retriever_node, vector_store=vector_store))
    graph.add_node("answering",  answering_node)
    graph.add_node("critic",     critic_node)

    # ── 3. Add the retry handler node ─────────────────────────────────────
    # This node runs BEFORE looping back to retriever.
    # Its only job: increment retry_count and update retrieval_query.
    def retry_handler(state: AgentState) -> dict:
        return {
            "retry_count":       state.get("retry_count", 0) + 1,
            "retrieval_query":   state.get("refined_query") or state["query"],
            # Clear previous answer so answering agent starts fresh
            "answer":            "",
            "reasoning":         "",
        }

    graph.add_node("retry_handler", retry_handler)

    # ── 4. Add unconditional edges ─────────────────────────────────────────
    graph.add_edge("retriever",    "answering")
    graph.add_edge("answering",    "critic")
    graph.add_edge("retry_handler","retriever")   # retry loops back to retriever

    # ── 5. Add the conditional edge after critic ───────────────────────────
    graph.add_conditional_edges(
        "critic",                  # source node
        route_after_critic,        # router function: returns "end" or "retry"
        {
            "end":   END,             # LangGraph's built-in terminal node
            "retry": "retry_handler", # loop back through retry_handler first
        }
    )

    # ── 6. Set entry point ────────────────────────────────────────────────
    graph.set_entry_point("retriever")

    # ── 7. Compile ────────────────────────────────────────────────────────
    return graph.compile()


def run_pipeline(
    query:        str,
    vector_store: FAISS,
    k:            int  = 5,
    paper_filter: str  = None,
    max_retries:  int  = 2,
) -> AgentState:
    """
    Entry point for the full multi-agent pipeline.
    Called by the Streamlit UI and FastAPI.

    Returns the final AgentState with everything populated:
      state["answer"]            → final verified answer
      state["critic_score"]      → quality score 1–10
      state["hallucination_flags"] → any unverified claims
      state["retry_count"]       → how many retries were needed
      state["chunks"]            → chunks the final answer is based on
    """
    compiled = build_graph(vector_store)

    initial_state: AgentState = {
        # Input
        "query":           query,
        "paper_filter":    paper_filter,
        "k":               k,
        # Retrieval
        "chunks":          [],
        "retrieval_query": query,   # starts as original query
        # Answering
        "answer":          "",
        "reasoning":       "",
        # Critic
        "critic_score":        0,
        "critic_feedback":     "",
        "hallucination_flags": [],
        "refined_query":       "",
        "verdict":             "",
        # Loop control
        "retry_count":  0,
        "max_retries":  max_retries,
    }

    print(f"\n{'='*60}")
    print(f"🚀 Multi-agent pipeline starting")
    print(f"   Query: {query[:70]}...")
    print(f"{'='*60}")

    final_state = compiled.invoke(initial_state)

    print(f"\n{'='*60}")
    print(f"✅ Pipeline complete")
    print(f"   Score:   {final_state['critic_score']}/10")
    print(f"   Retries: {final_state['retry_count']}")
    print(f"   Flags:   {final_state['hallucination_flags'] or 'None'}")
    print(f"{'='*60}\n")

    return final_state