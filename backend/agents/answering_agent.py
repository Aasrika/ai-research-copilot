"""
answering_agent.py
------------------
Node 2 in the LangGraph.

WHAT'S NEW vs Phase 2's basic_qa:
  Chain-of-thought (CoT) reasoning.

WHY CHAIN-OF-THOUGHT?
  Asking the model to first REASON through the sources before writing
  the final answer significantly reduces hallucination. The model is
  forced to confront "is this claim actually in the text?" before
  committing to it.

  Without CoT:  model may blend training knowledge + retrieved text
  With CoT:     model explicitly maps claims to sources first

This also produces a "reasoning" field in state that the Critic Agent
reads — the Critic can catch reasoning steps that don't hold up.
"""
from langchain_groq import ChatGroq
from agents.state import AgentState
from core.retriever import format_context
from core.config import ANSWERING_MODEL, TEMPERATURE


ANSWERING_PROMPT = """You are a rigorous scientific research assistant.
Your job is to answer questions about research papers using ONLY the provided source chunks.

First, reason through the sources step by step.
Then write your final answer with inline citations.

SOURCE CHUNKS:
{context}

QUESTION: {query}

{retry_instruction}

Respond in this exact format:

REASONING:
<Step-by-step: which sources are relevant, what each says, what gaps exist>

ANSWER:
<Final answer with [Source N, Page P] citations for every claim>

COVERAGE:
<One sentence: what the sources covered well vs what was missing>
"""


def answering_node(state: AgentState) -> dict:
    """
    LangGraph node: generate a cited, reasoned answer from retrieved chunks.
    """

    chunks = state["chunks"]
    query = state["query"]
    retry = state.get("retry_count", 0)
    feedback = state.get("critic_feedback", "")

    # 🔁 Inject critic feedback on retry
    retry_instruction = ""
    if retry > 0 and feedback:
        retry_instruction = (
            "NOTE — Previous answer had issues:\n"
            f"{feedback}\n"
            "Fix these issues and improve factual grounding.\n"
        )

    # Format retrieved chunks into context
    context = format_context(chunks)

    prompt = ANSWERING_PROMPT.format(
        context=context,
        query=query,
        retry_instruction=retry_instruction,
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=TEMPERATURE)

    response = llm.invoke(prompt)
    raw_output = response.content

    # Parse structured response
    reasoning, answer, coverage = _parse_response(raw_output)

    print(f"\n✍️ Answer generated ({len(answer)} chars)")
    if retry > 0:
        print(f"   🔁 Retry #{retry} with critic feedback")

    return {
        "answer": answer,
        "reasoning": reasoning,
    }


def _parse_response(raw: str) -> tuple[str, str, str]:
    """Extract REASONING, ANSWER, and COVERAGE sections from the LLM output."""

    def extract(text, start_tag, end_tag=None):
        start = text.find(start_tag)
        if start == -1:
            return ""

        start += len(start_tag)

        if end_tag:
            end = text.find(end_tag, start)
            return text[start:end].strip() if end != -1 else text[start:].strip()

        return text[start:].strip()

    reasoning = extract(raw, "REASONING:\n", "ANSWER:")
    answer = extract(raw, "ANSWER:\n", "COVERAGE:")
    coverage = extract(raw, "COVERAGE:\n")

    # Fallback safety
    if not answer:
        return "", raw, ""

    return reasoning, answer, coverage