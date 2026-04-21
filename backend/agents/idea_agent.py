import json, re, os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from core.retriever import retrieve, format_context


# ─────────────────────────────────────────────────────────────────────────────
# GROQ CLIENT
# ─────────────────────────────────────────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ─────────────────────────────────────────────────────────────────────────────
# IDEA PROMPT
# ─────────────────────────────────────────────────────────────────────────────
IDEA_PROMPT = """You are an experienced research scientist reviewing an academic paper.

Your goal is to generate high-value research insights that go BEYOND what the paper states.

Label clearly:
[FROM PAPER] — directly stated
[INFERRED] — your reasoning

PAPER EXCERPTS:
{context}

PAPER TITLE: {paper_title}
FOCUS AREA: {focus_area}

Respond ONLY in valid JSON:

{{
  "explicit_limitations": [
    {{"finding": "string", "source": "[FROM PAPER]", "page_hint": "string"}}
  ],
  "implicit_limitations": [
    {{"finding": "string", "source": "[INFERRED]", "reasoning": "string"}}
  ],
  "open_questions": [
    {{"question": "string", "why_important": "string"}}
  ],
  "experiment_ideas": [
    {{
      "title": "string",
      "description": "string",
      "expected_impact": "string",
      "difficulty": "Low|Medium|High"
    }}
  ],
  "dataset_improvements": [
    {{"suggestion": "string", "rationale": "string"}}
  ],
  "methodological_alternatives": [
    {{"alternative": "string", "potential_advantage": "string"}}
  ],
  "summary": "string"
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────
def _retrieve_for_ideas(vector_store: FAISS, paper_filter: str, focus_area: str):
    
    weakness_query = (
        f"limitations weaknesses future work {focus_area}"
        if focus_area else
        "limitations weaknesses future work conclusion"
    )

    method_query = (
        f"methodology contribution {focus_area}"
        if focus_area else
        "methodology main contribution experiment"
    )

    chunks_1 = retrieve(weakness_query, vector_store, k=5, paper_filter=None)
    chunks_2 = retrieve(method_query, vector_store, k=4, paper_filter=None)

    seen, combined = set(), []
    for c in chunks_1 + chunks_2:
        key = (c.metadata["page"], c.metadata.get("chunk_index", 0))
        if key not in seen:
            seen.add(key)
            combined.append(c)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def generate_research_ideas(vector_store: FAISS, paper_filter=None, focus_area=""):

    print("\n💡 Idea Agent running...")

    chunks = _retrieve_for_ideas(vector_store, paper_filter, focus_area)
    print(f"   Retrieved {len(chunks)} chunks")

    context = format_context(chunks)

    prompt = IDEA_PROMPT.format(
        context=context,
        paper_title=paper_filter or "the paper",
        focus_area=focus_area or "general"
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    raw = response.choices[0].message.content

    ideas = _parse_ideas(raw)

    ideas["chunks_used"] = chunks
    ideas["paper_title"] = paper_filter
    ideas["focus_area"]  = focus_area

    return ideas


# ─────────────────────────────────────────────────────────────────────────────
# SAFE PARSER
# ─────────────────────────────────────────────────────────────────────────────
def _parse_ideas(raw: str):

    try:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        return json.loads(cleaned)

    except Exception:
        print("⚠️ JSON parse failed — returning fallback")

        return {
            "summary": raw[:500],
            "explicit_limitations": [],
            "implicit_limitations": [],
            "open_questions": [],
            "experiment_ideas": [],
            "dataset_improvements": [],
            "methodological_alternatives": [],
        }