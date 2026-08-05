"""
classification_agent.py
------------------------
Node 4 in the critique LangGraph — classifies each claim as SUPPORTED,
CONTRADICTED, or SILENT against its retrieved literature evidence, using
CRITIQUE_MODEL. One Groq call per claim (bounded by max_claims) rather
than one batched call: a bad parse only affects that single claim, and
the "quote must appear in evidence" safety net below can be checked
against exactly that claim's own retrieved context.
"""

import json
import re
import os
from groq import Groq

from core.config import CRITIQUE_MODEL
from core.retriever import format_context
from agents.critique_state import CritiqueState
from core.token_tracking import accumulate, read_totals

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


CLASSIFICATION_PROMPT = """You are auditing whether a draft paper's claim is supported, contradicted, or unaddressed by a set of literature passages.

CLAIM: {claim_text}

LITERATURE PASSAGES:
{context}

Classify the claim:
- SUPPORTED: a passage directly and clearly supports this claim
- CONTRADICTED: a passage directly and explicitly states something incompatible with this claim — you MUST quote the exact contradicting sentence from the passages above, word for word
- SILENT: the passages don't directly address this claim, are only tangentially related (different task/dataset/domain), or you are not confident

Default to SILENT unless a passage directly addresses the claim. Do NOT classify as CONTRADICTED just because a passage describes a different approach, a different dataset, or different numbers in a different setting — that is SILENT, not CONTRADICTED.

Also judge whether the claim appears to already be cited to one of these papers (you cannot see the draft's citation markers directly here, so base this only on whether the claim text itself already contains a citation marker like [12] or (Smith et al., 2021); otherwise assume not cited).

IMPORTANT: Your entire response must be ONLY a single JSON object — no preamble, no explanations, no markdown fences. Start your response with {{ and end with }}, matching exactly this schema:

{{
  "verdict": "<SUPPORTED|CONTRADICTED|SILENT>",
  "confidence": "<low|medium|high>",
  "contradicting_quote": "<exact quote from the passages if CONTRADICTED, else empty string>",
  "supporting_paper": "<paper title from the passages that best aligns with this claim, or empty string>",
  "notes": "<brief nuance, e.g. mixed evidence across papers, or empty string>",
  "already_cited": <true|false>
}}
"""


def classify_claims_node(state: CritiqueState) -> dict:
    claims = state.get("claims", [])
    claim_evidence = state.get("claim_evidence", {})

    alignments = []
    token_totals = read_totals(state)

    for claim in claims:
        evidence_chunks = claim_evidence.get(claim["id"], [])
        context = format_context(evidence_chunks) if evidence_chunks else "(no literature passages retrieved)"

        prompt = CLASSIFICATION_PROMPT.format(claim_text=claim["text"], context=context)

        response = None
        try:
            response = client.chat.completions.create(
                model=CRITIQUE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw_output = response.choices[0].message.content or ""
        except Exception as e:
            print(f"❌ Groq API error during classification: {e}")
            raw_output = ""

        if response is not None:
            token_totals = accumulate(token_totals, response, CRITIQUE_MODEL, prompt, raw_output)

        parsed = _parse_classification(raw_output)
        verdict = parsed["verdict"]

        # 🛡️ Programmatic safety net: a CONTRADICTED verdict must quote a
        # sentence that actually appears in this claim's retrieved evidence.
        # If it doesn't, we don't trust the model's contradiction — downgrade
        # to SILENT rather than risk a false-positive contradiction reaching
        # the report (this was the #2 risk flagged in the design review).
        if verdict == "CONTRADICTED":
            evidence_text = " ".join(c.page_content for c in evidence_chunks)
            quote = parsed.get("contradicting_quote", "")
            if not quote.strip() or quote.strip() not in evidence_text:
                verdict = "SILENT"
                parsed["notes"] = (
                    parsed.get("notes", "")
                    + " [downgraded from CONTRADICTED: quote not found verbatim in evidence]"
                ).strip()

        evidence = [
            {
                "paper_title": c.metadata.get("paper_title", "Unknown"),
                "page": c.metadata.get("page", 0),
                "section": c.metadata.get("section", "body"),
                "snippet": c.page_content[:300],
            }
            for c in evidence_chunks
        ]

        suggested_citation = None
        if verdict == "SUPPORTED" and not parsed.get("already_cited") and parsed.get("supporting_paper"):
            suggested_citation = parsed["supporting_paper"]

        alignments.append({
            "claim_id": claim["id"],
            "claim_text": claim["text"],
            "claim_type": claim.get("claim_type", "discussion"),
            "source_section": claim.get("source_section", "body"),
            "verdict": verdict,
            "confidence": parsed.get("confidence", "low"),
            "evidence": evidence,
            "notes": parsed.get("notes", ""),
            "already_cited": bool(parsed.get("already_cited", False)),
            "suggested_citation": suggested_citation,
        })

    print(f"\n⚖️ Classified {len(alignments)} claims")

    return {"claim_alignments": alignments, **token_totals}


def _parse_classification(raw: str) -> dict:
    if not raw:
        return _fallback()

    try:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        parsed = json.loads(cleaned)
        if parsed.get("verdict") not in ("SUPPORTED", "CONTRADICTED", "SILENT"):
            parsed["verdict"] = "SILENT"
        return parsed
    except Exception:
        print("⚠️ Invalid JSON from classification — falling back to SILENT")
        return _fallback()


def _fallback() -> dict:
    return {
        "verdict": "SILENT",
        "confidence": "low",
        "contradicting_quote": "",
        "supporting_paper": "",
        "notes": "Parsing failed — defaulted to SILENT",
        "already_cited": False,
    }
