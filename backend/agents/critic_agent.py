"""
critic_agent.py
---------------
Node 3 in the LangGraph. This is what makes your project research-level.

THE CRITIC'S JOB:
  Given the answer + the source chunks it was generated from,
  verify that every factual claim in the answer is actually supported
  by the sources.

  Claims not found in sources = hallucinations.

WHY THIS IS HARD (and why it matters):
  LLMs have parametric knowledge (training data) + retrieved context.
  Without a critic, the model happily blends both — citing a number
  from training that the retrieved text never stated. In a research
  context, this is dangerous. A misattributed accuracy metric or
  a wrong dataset name could mislead your entire analysis.

  This is literally what RAGFail (your own research!) is about.
  You're building the solution to the problem you studied.

CRITIC OUTPUT:
  - critic_score (1–10):   overall answer quality
  - hallucination_flags:   specific unverified claims
  - critic_feedback:       actionable instructions for improvement
  - refined_query:         a better query for re-retrieval (if needed)
  - verdict:               PASS / RETRY / FAIL
"""

import json, re
from langchain_groq import ChatGroq
from agents.state import AgentState
from core.retriever import format_context
from core.config import CRITIC_MODEL


CRITIC_PROMPT = """You are a strict scientific fact-checker auditing an AI-generated answer about a research paper.

Your task:
1. Read the SOURCE CHUNKS (ground truth)
2. Read the ANSWER (what the AI claimed)
3. Read the REASONING (how the AI arrived at the answer)
4. For each factual claim in the answer, verify it is directly supported by a source chunk

SOURCE CHUNKS:
{context}

ORIGINAL QUESTION: {query}

AI'S REASONING:
{reasoning}

AI'S ANSWER:
{answer}

Respond ONLY with valid JSON in this exact schema:
{{
  "score": <integer 1-10>,
  "verdict": "<PASS|RETRY|FAIL>",
  "hallucination_flags": ["<claim 1 not in sources>", "<claim 2>"],
  "strengths": ["<what the answer did well>"],
  "issues": ["<specific problem 1>", "<specific problem 2>"],
  "feedback": "<One paragraph of actionable feedback for improving the answer>",
  "refined_query": "<A more specific version of the original question that would retrieve better chunks, or empty string if query was fine>"
}}
"""


def critic_node(state: AgentState) -> dict:

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

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    response = llm.invoke(prompt)

    parsed = _parse_critic_response(response.content)

    score = parsed.get("score", 5)
    verdict = parsed.get("verdict", "RETRY")

    print(f"\n🔍 Critic verdict: {verdict} (score: {score}/10)")

    if parsed.get("hallucination_flags"):
        print(f"⚠️ Flags: {parsed['hallucination_flags']}")

    if parsed.get("refined_query"):
        print(f"🔎 Refined query: {parsed['refined_query']}")

    return {
        "critic_score": score,
        "critic_feedback": parsed.get("feedback", ""),
        "hallucination_flags": parsed.get("hallucination_flags", []),
        "refined_query": parsed.get("refined_query", ""),
        "verdict": verdict,
    }


def _parse_critic_response(raw: str) -> dict:
    try:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        return json.loads(cleaned)
    except Exception:
        print("⚠️ Invalid JSON from critic — fallback triggered")
        return {
            "score": 5,
            "verdict": "RETRY",
            "hallucination_flags": [],
            "feedback": "Parsing failed — retrying",
            "refined_query": "",
        }


# 🔀 Routing logic (VERY IMPORTANT)

def route_after_critic(state: AgentState) -> str:

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