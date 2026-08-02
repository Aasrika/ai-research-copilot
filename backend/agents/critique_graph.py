"""
critique_graph.py
------------------
LangGraph for Mode 2 (Draft Critique).

Linear chain, no retry loop:
  section_check -> extract_claims -> retrieve_evidence -> classify_claims -> assemble_report -> END

Unlike the Q&A pipeline, there's nothing to retry against here: critique
is a single-pass analytical report, not an answer being quality-graded,
so reusing the Q&A graph's critic-retry pattern would be over-engineering.
section_check runs first and unconditionally since it's deterministic and
can't "fail" the way an LLM step can.
"""

import time
from functools import partial
from datetime import datetime, timezone
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS

from agents.critique_state import CritiqueState
from agents.section_check_agent import section_check_node
from agents.claim_agent import extract_claims_node
from agents.evidence_agent import retrieve_evidence_node
from agents.classification_agent import classify_claims_node
from core.critique_models import CritiqueReport
from evaluation.logger import RunRecord, log_run


def _assemble_report_node(state: CritiqueState) -> dict:
    claims = state.get("claim_alignments", [])
    section_checks = state.get("section_checks", [])

    verdict_counts = {"SUPPORTED": 0, "CONTRADICTED": 0, "SILENT": 0}
    citation_suggestions = 0
    for c in claims:
        verdict_counts[c["verdict"]] = verdict_counts.get(c["verdict"], 0) + 1
        if c.get("suggested_citation"):
            citation_suggestions += 1

    missing_sections = sum(1 for s in section_checks if not s["present"])

    summary = {
        "missing_sections": missing_sections,
        "supported": verdict_counts["SUPPORTED"],
        "contradicted": verdict_counts["CONTRADICTED"],
        "silent": verdict_counts["SILENT"],
        "citation_suggestions": citation_suggestions,
    }

    report = CritiqueReport(
        session_id=state["session_id"],
        draft_filename=Path(state["draft_path"]).name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        section_checks=section_checks,
        claims=claims,
        summary=summary,
    )

    print(f"\n📋 Critique report assembled — {len(claims)} claims, {missing_sections} missing sections")

    return {"report": report.model_dump()}


def build_critique_graph(literature_store: FAISS) -> StateGraph:
    graph = StateGraph(CritiqueState)

    graph.add_node("section_check", section_check_node)
    graph.add_node("extract_claims", extract_claims_node)
    graph.add_node(
        "retrieve_evidence",
        partial(retrieve_evidence_node, literature_store=literature_store),
    )
    graph.add_node("classify_claims", classify_claims_node)
    graph.add_node("assemble_report", _assemble_report_node)

    graph.add_edge("section_check", "extract_claims")
    graph.add_edge("extract_claims", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "classify_claims")
    graph.add_edge("classify_claims", "assemble_report")
    graph.add_edge("assemble_report", END)

    graph.set_entry_point("section_check")

    return graph.compile()


def run_critique(
    session_id: str,
    draft_path: str,
    literature_store: FAISS,
    max_claims: int = 12,
) -> CritiqueReport:
    compiled = build_critique_graph(literature_store)

    initial_state: CritiqueState = {
        "session_id": session_id,
        "draft_path": draft_path,
        "max_claims": max_claims,

        "draft_pages": [],
        "draft_text": "",
        "section_checks": [],

        "claims": [],
        "claim_evidence": {},
        "claim_alignments": [],

        "report": {},
        "run_id": "",
    }

    print(f"\n{'='*60}")
    print(f"📝 Critique Pipeline | {Path(draft_path).name}")
    print(f"{'='*60}")

    start = time.perf_counter()
    final_state = compiled.invoke(initial_state)
    elapsed = round(time.perf_counter() - start, 3)

    report = CritiqueReport.model_validate(final_state["report"])

    # Self-logs like run_pipeline does — critic_score/verdict/hallucination_flags
    # stay at their RunRecord defaults since they don't map cleanly onto a
    # critique run (same convention already used for comparison/ideas runs).
    papers_cited = sorted({
        ev["paper_title"]
        for claim in final_state.get("claim_alignments", [])
        for ev in claim.get("evidence", [])
    })
    record = RunRecord(
        pipeline_type    = "critique",
        session_id       = session_id,
        query            = f"Critique: {report.draft_filename}",
        num_chunks       = len(report.claims),
        papers_retrieved = papers_cited,
        latency_total    = elapsed,
    )
    log_run(record)

    print(f"{'='*60}")
    print("✅ Critique done")
    print(f"{'='*60}\n")

    return report
