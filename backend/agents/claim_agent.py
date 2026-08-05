"""
claim_agent.py
--------------
Node 2 in the critique LangGraph — extracts major, checkable claims
from the draft. Uses CRITIQUE_MODEL (llama-3.3-70b-versatile), a larger
model than the Q&A pipeline's, since claim extraction is a reasoning-
heavy operation (identifying atomic, checkable assertions and their
type) rather than grounded lookup.

Same JSON-prompt + fence-strip/regex-extract parsing style as
critic_agent.py, for consistency with the rest of the codebase.
"""

import json
import re
import os
import uuid
from groq import Groq

from core.config import CRITIQUE_MODEL
from agents.critique_state import CritiqueState
from core.token_tracking import accumulate, read_totals

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


CLAIM_EXTRACTION_PROMPT = """You are analyzing a draft research paper to extract its major, checkable claims. You must respond with ONLY a JSON object — never continue, complete, or paraphrase the draft text itself, even if it ends mid-sentence or mid-table.

Extract at most {max_claims} major claims — single-sentence, checkable, factual assertions the paper makes. Do NOT extract every sentence; focus on what matters for a literature-alignment review.

For each claim, classify its claim_type:
- "contribution": headline/novelty claims about what the paper contributes (mainly from Abstract/Introduction)
- "result": empirical findings (mainly from Results/Experiments)
- "discussion": interpretive claims, limitations, or reflections (mainly from Discussion/Conclusion)

DRAFT TEXT (section-tagged, for reference only — do not repeat, continue, or complete this text in your response):
{draft_text}

Remember: your entire response must be ONLY a single JSON object — no preamble, no numbered list, no headers, no explanations, no markdown fences, and do NOT repeat or continue the draft text above. Start your response with {{ and end with }}, matching exactly this schema:

{{
  "claims": [
    {{
      "text": "<single-sentence claim, as close to the paper's own wording as possible>",
      "claim_type": "<contribution|result|discussion>",
      "source_section": "<section it most likely comes from>"
    }}
  ]
}}
"""


def extract_claims_node(state: CritiqueState) -> dict:
    draft_text = state.get("draft_text", "")
    max_claims = state.get("max_claims", 12)

    if not draft_text.strip():
        print("⚠️ No draft text available — skipping claim extraction")
        return {"claims": []}

    prompt = CLAIM_EXTRACTION_PROMPT.format(draft_text=draft_text, max_claims=max_claims)

    response = None
    try:
        response = client.chat.completions.create(
            model=CRITIQUE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw_output = response.choices[0].message.content or ""
    except Exception as e:
        print(f"❌ Groq API error during claim extraction: {e}")
        raw_output = ""

    claims = _parse_claims(raw_output, max_claims)

    print(f"\n🧾 Extracted {len(claims)} claims")

    token_totals = read_totals(state)
    if response is not None:
        token_totals = accumulate(token_totals, response, CRITIQUE_MODEL, prompt, raw_output)

    return {"claims": claims, **token_totals}


def _parse_claims(raw: str, max_claims: int) -> list[dict]:
    if not raw:
        return []

    try:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        parsed = json.loads(cleaned)
        raw_claims = parsed.get("claims", [])
    except Exception:
        print("⚠️ Invalid JSON from claim extraction — returning no claims")
        return []

    claims = []
    for item in raw_claims[:max_claims]:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        claims.append({
            "id": uuid.uuid4().hex[:8],
            "text": text,
            "claim_type": item.get("claim_type") or "discussion",
            "source_section": item.get("source_section") or "body",
        })

    return claims
