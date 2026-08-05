"""
critic_agent.py
---------------
Node 3 in the LangGraph.

Strict fact-checker that verifies whether the generated answer
is grounded in retrieved source chunks.
"""

import json
import re
import os
from groq import Groq
from core.config import CRITIC_MODEL
from agents.state import AgentState
from core.retriever import format_context
from core.token_tracking import accumulate, read_totals


CRITIC_PROMPT = """You are a strict scientific fact-checker auditing an AI-generated answer about research papers.

Your task:
1. Read the SOURCE CHUNKS (ground truth)
2. Read the ANSWER (what the AI claimed)
3. Read the REASONING (how the AI arrived at the answer)
4. For each factual claim in the answer, verify it is directly supported by a source chunk
5. Judge whether the answer is SPECIFIC and USEFUL, not just technically correct

EVALUATION RUBRIC:
- PASS (8-10): Claims are grounded AND the answer is specific — names methods, cites numbers, contrasts papers, references exact findings
- RETRY (5-7): Claims are grounded but answer is vague, generic, or category-level when specific details were available in the sources. A better retrieval or a more specific query would help.
- FAIL (1-4): Claims are not grounded, hallucinated, or the answer contradicts the sources

DO NOT AWARD PASS for answers that are technically grounded but vague (e.g. "the paper discusses limitations" when the sources contain named specific limitations that were not surfaced).

SOURCE CHUNKS:
{context}

ORIGINAL QUESTION: {query}

AI'S REASONING:
{reasoning}

AI'S ANSWER:
{answer}

Respond ONLY with valid JSON.
Do NOT include explanations, markdown, or text outside JSON.

Schema:
{{
  "score": <integer 1-10>,
  "verdict": "<PASS|RETRY|FAIL>",
  "hallucination_flags": ["<claim not in sources>"],
  "vagueness_flags": ["<vague claim where sources had specifics>"],
  "feedback": "<actionable improvement>",
  "refined_query": "<better query for retry, or empty string>"
}}
"""


def critic_node(state: AgentState) -> dict:
    """
    LangGraph node: fact-check the generated answer.
    """

    chunks = state["chunks"]
    answer = state["answer"]
    reasoning = state.get("reasoning", "")
    query = state["query"]

    context = format_context(chunks)

    prompt = CRITIC_PROMPT.format(
        context=context,
        query=query,
        reasoning=reasoning,
        answer=answer,
    )

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = None
    try:
        response = client.chat.completions.create(
        model=CRITIC_MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

        raw_output = response.choices[0].message.content

    except Exception as e:
        print(f"❌ Groq API error: {e}")
        raw_output = ""

    print("\n🧠 RAW CRITIC OUTPUT:\n", raw_output)

    # response is None if the call itself raised — nothing to extract usage
    # from, so the running totals just pass through unchanged.
    token_totals = read_totals(state)
    if response is not None:
        token_totals = accumulate(token_totals, response, CRITIC_MODEL, prompt, raw_output or "")

    parsed = _parse_critic_response(raw_output)

    score = parsed.get("score", 5)
    verdict = parsed.get("verdict", "RETRY")

    print(f"\n🔍 Critic verdict: {verdict} (score: {score}/10)")

    if parsed.get("hallucination_flags"):
        print(f"⚠️ Flags: {parsed['hallucination_flags']}")

    if parsed.get("vagueness_flags"):
        print(f"💤 Vagueness: {parsed['vagueness_flags']}")

    if parsed.get("refined_query"):
        print(f"🔎 Refined query: {parsed['refined_query']}")

    return {
        "critic_score": score,
        "critic_feedback": parsed.get("feedback", ""),
        "hallucination_flags": parsed.get("hallucination_flags", []),
        "vagueness_flags": parsed.get("vagueness_flags", []),
        "refined_query": parsed.get("refined_query", ""),
        "verdict": verdict,
        **token_totals,
    }


def _parse_critic_response(raw: str) -> dict:
    """
    Parse JSON output from LLM safely.
    """

    if not raw:
        return _fallback()

    try:
        # Remove markdown formatting
        cleaned = re.sub(r"```json|```", "", raw).strip()

        # Extract JSON block if extra text exists
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        return json.loads(cleaned)

    except Exception:
        print("⚠️ Invalid JSON from critic — fallback triggered")
        return _fallback()


def _fallback() -> dict:
    return {
        "score": 5,
        "verdict": "RETRY",
        "hallucination_flags": [],
        "vagueness_flags": [],
        "feedback": "Parsing failed — retrying",
        "refined_query": "",
    }


# 🔀 Routing logic

def route_after_critic(state: AgentState) -> str:
    """
    Decide whether to accept answer or retry.
    """

    verdict = state.get("verdict", "PASS")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if retry_count >= max_retries:
        print("\n⛔ Max retries reached")
        return "end"

    if verdict == "PASS":
        print("\n✅ Answer accepted")
        return "end"

    if verdict in ["RETRY", "FAIL"]:
        print(f"\n🔄 Retrying... ({retry_count + 1})")
        return "retry"

    return "end"