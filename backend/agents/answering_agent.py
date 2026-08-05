"""
answering_agent.py
------------------
Node 2 in the LangGraph.

Robust grounded answering using Groq API.
"""

import os
from groq import Groq

from agents.state import AgentState
from core.retriever import format_context
from core.config import ANSWERING_MODEL, TEMPERATURE
from core.token_tracking import accumulate, read_totals


ANSWERING_PROMPT = """You are a rigorous scientific research assistant answering questions about research papers.

STRICT RULES:
- Use ONLY the provided source chunks. Do not use outside knowledge.
- If information is missing, explicitly say "Not found in sources"
- Every substantive claim MUST have a citation [Source X, Page Y]

QUALITY REQUIREMENTS:
- Prefer specific findings, named methods, exact numbers, and direct paraphrases over abstract summaries
- When multiple papers address the question, contrast them explicitly (e.g., "Paper 1 argues X while Paper 2 finds Y")
- Do not use vague category-level phrasing like "lack of robustness" when the source contains specific details you could name
- If the retrieved sources address only part of the question, answer that part precisely and state what remains unaddressed
- Prefer 2-4 focused sentences over long generic paragraphs

SOURCE CHUNKS:
{context}

QUESTION: {query}

{retry_instruction}

Respond EXACTLY in this format:

REASONING:
Step-by-step reasoning that references specific source chunks by number

ANSWER:
Final answer with inline citations. Be specific and concrete.

COVERAGE:
What the sources directly answered vs what's missing
"""


def answering_node(state: AgentState) -> dict:
    """
    LangGraph node: generate grounded answer.
    """

    chunks = state.get("chunks", [])
    query = state.get("query", "")
    retry = state.get("retry_count", 0)
    feedback = state.get("critic_feedback", "")

    # 🚨 If no chunks → fail early
    if not chunks:
        print("⚠️ No chunks retrieved — skipping generation")
        return {
            "answer": "No relevant information found in the documents.",
            "reasoning": "",
            "coverage": "No sources retrieved",
        }

    # Retry instruction
    retry_instruction = ""
    if retry > 0 and feedback:
        retry_instruction = (
            "NOTE — Previous answer had issues:\n"
            f"{feedback}\n"
            "Fix these issues.\n"
        )

    context = format_context(chunks)

    prompt = ANSWERING_PROMPT.format(
        context=context,
        query=query,
        retry_instruction=retry_instruction,
    )

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        response = client.chat.completions.create(
            model=ANSWERING_MODEL,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_output = response.choices[0].message.content or ""

    except Exception as e:
        print(f"❌ Groq API error: {e}")
        return {
            "answer": "Model failed to generate response.",
            "reasoning": "",
            "coverage": "",
        }

    # Only reached on a successful call, so `response` is always defined here.
    token_totals = accumulate(read_totals(state), response, ANSWERING_MODEL, prompt, raw_output)

    reasoning, answer, coverage = _parse_response(raw_output)

    # 🚨 Hard fallback if parsing fails
    if not answer.strip():
        print("⚠️ Empty answer — fallback triggered")
        return {
            "answer": raw_output.strip() or "Failed to generate answer.",
            "reasoning": "",
            "coverage": "",
            **token_totals,
        }

    print(f"\n✍️ Answer generated ({len(answer)} chars)")
    if retry > 0:
        print(f"   🔁 Retry #{retry}")

    return {
        "answer": answer,
        "reasoning": reasoning,
        **token_totals,
        "coverage": coverage,
    }


def _parse_response(raw: str) -> tuple[str, str, str]:
    """
    Robust parser for model output
    """

    raw = raw.strip()

    def extract(text, start_tag, end_tag=None):
        start = text.find(start_tag)
        if start == -1:
            return ""

        start += len(start_tag)

        if end_tag:
            end = text.find(end_tag, start)
            if end == -1:
                return text[start:].strip()
            return text[start:end].strip()

        return text[start:].strip()

    reasoning = extract(raw, "REASONING:", "ANSWER:")
    answer = extract(raw, "ANSWER:", "COVERAGE:")
    coverage = extract(raw, "COVERAGE:")

    # 🛠 fallback if format is broken
    if not answer:
        return "", raw, ""

    return reasoning, answer, coverage