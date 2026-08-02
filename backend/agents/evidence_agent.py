"""
evidence_agent.py
-----------------
Node 3 in the critique LangGraph — deterministic FAISS retrieval, no LLM.
Kept separate from claim_agent.py on purpose: retrieval is deterministic
and cheap, claim extraction is a reasoning-heavy LLM call — same split
as retriever_agent.py / answering_agent.py in the Q&A pipeline.

For each extracted claim, retrieves the top-k most relevant passages
from the session's literature store. The draft itself is never in this
store (session_manager never indexes the draft — see its docstring), so
there's no risk of a claim "supporting itself" against its own text.
"""

from langchain_community.vectorstores import FAISS

from core.retriever import retrieve
from agents.critique_state import CritiqueState

EVIDENCE_K = 4


def retrieve_evidence_node(state: CritiqueState, literature_store: FAISS) -> dict:
    claims = state.get("claims", [])

    claim_evidence: dict[str, list] = {}
    for claim in claims:
        chunks = retrieve(
            query=claim["text"],
            vector_store=literature_store,
            k=EVIDENCE_K,
            section_hint=False,  # claims are declarative, not natural-language questions
        )
        claim_evidence[claim["id"]] = chunks

    print(f"\n🔎 Retrieved evidence for {len(claim_evidence)} claims")

    return {"claim_evidence": claim_evidence}
