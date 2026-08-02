"""
critique_models.py
-------------------
Pydantic schema for the Mode 2 (Draft Critique) report — the first
proper Pydantic domain model in the app (Pydantic was previously used
only for FastAPI request/response models in backend/api/main.py).
Scoped to this feature only; existing agents keep their prompt-JSON +
plain-dict conventions unchanged.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel


class SectionCheckItem(BaseModel):
    section: str
    present: bool
    word_count: int
    is_suspiciously_short: bool
    reason: str = ""


class ClaimEvidence(BaseModel):
    paper_title: str
    page: int
    section: str
    snippet: str


class ClaimAlignment(BaseModel):
    claim_id: str
    claim_text: str
    claim_type: Literal["contribution", "result", "discussion"]
    source_section: str
    verdict: Literal["SUPPORTED", "CONTRADICTED", "SILENT"]
    confidence: Literal["low", "medium", "high"]
    evidence: List[ClaimEvidence] = []
    notes: str = ""
    already_cited: bool = False
    suggested_citation: Optional[str] = None


class CritiqueReport(BaseModel):
    session_id: str
    draft_filename: str
    generated_at: str
    section_checks: List[SectionCheckItem]
    claims: List[ClaimAlignment]
    summary: dict
