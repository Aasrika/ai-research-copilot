"""
critique_state.py
------------------
Shared state for the Mode 2 (Draft Critique) LangGraph. Mirrors the
existing AgentState convention (flat TypedDict, nodes read/write partial
updates) rather than inventing a new pattern.

Note: unlike AgentState, there's no retry_count/max_retries here — the
critique graph is a linear, single-pass pipeline with no retry loop
(see critique_graph.py docstring for why).
"""

from typing import TypedDict, List, Dict


class CritiqueState(TypedDict):

    # ── Input ──────────────────────────────────────────────────────────────
    session_id: str
    draft_path: str
    max_claims: int

    # ── Section Check node output (rule-based, no LLM) ─────────────────────
    draft_pages: List[dict]
    draft_text: str
    section_checks: List[dict]

    # ── Claim Extraction node output ───────────────────────────────────────
    claims: List[dict]

    # ── Evidence Retrieval node output ─────────────────────────────────────
    claim_evidence: Dict[str, list]

    # ── Classification node output ──────────────────────────────────────────
    claim_alignments: List[dict]

    # ── Final assembled report ───────────────────────────────────────────────
    report: dict
    run_id: str

    # ── Token/cost tracking (accumulated across the extraction call and
    #    every per-claim classification call — see core/token_tracking.py) ──
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    tokens_estimated: bool
