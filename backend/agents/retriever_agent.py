"""
retriever_agent.py
------------------
Node 1 in the LangGraph.

WHAT IT DOES:
  Reads the retrieval_query from state → fetches relevant chunks →
  writes chunks back to state.

WHY IT'S A SEPARATE AGENT (not just a function call):
  On a RETRY loop, the Critic Agent rewrites the query before sending
  control back here. This node doesn't care WHY it was called — it just
  retrieves for whatever query is in state. That clean separation is
  the whole point of the agent architecture.

On first call:  retrieval_query == original user query
On retry:       retrieval_query == critic's refined query (more specific)
"""

from langchain_community.vectorstores import FAISS
from agents.state import AgentState
from core.retriever import retrieve


def retriever_node(state: AgentState, vector_store: FAISS) -> dict:
    """
    LangGraph node: fetch chunks and update state.
    """

    # Get query (refined if exists, else original)
    query = state.get("retrieval_query") or state["query"]

    # Other params
    k = state.get("k", 5)
    paper_filter = state.get("paper_filter")
    retry_count = state.get("retry_count", 0)

    # Debug logs (VERY useful for Phase 3)
    if retry_count > 0:
        print(f"\n🔄 RETRY {retry_count}: Re-retrieving with refined query")
        print(f"   Original Query : {state['query']}")
        print(f"   Refined Query : {query}")

    # Retrieve chunks
    chunks = retrieve(query, vector_store, k=k, paper_filter=paper_filter)

    print(f"📦 Retrieved {len(chunks)} chunks")

    return {
        "chunks": chunks,
        "retrieval_query": query,
    }