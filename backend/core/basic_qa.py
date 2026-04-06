"""
basic_qa.py
-----------
Phase 1 Final Piece: Take retrieved chunks → generate a cited answer.
"""

from groq import Groq
import os
from dotenv import load_dotenv
from backend.core.config import TOP_K, TEMPERATURE
from langchain_core.documents import Document
from backend.core.document_processor import load_index
from backend.core.retriever import retrieve, format_context

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(query: str, chunks: list[Document]) -> str:
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        src = f"[Source {i}: {chunk.metadata['source']}, Page {chunk.metadata['page']}]"
        context_parts.append(f"{src}\n{chunk.page_content}")

    context_block = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a scientific research assistant analyzing academic papers.

Answer the following question using ONLY the provided context from the paper(s).
For every claim you make, cite the source using [Source N] notation.
If the provided context does not contain enough information, say so clearly.

CONTEXT:
{context_block}

QUESTION: {query}

ANSWER (with citations):"""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RAG FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def answer_question(query: str, vector_store, k: int = 5, paper_filter=None) -> dict:
    
    # Step 1: Retrieve chunks
    chunks = retrieve(query, vector_store, k=k, paper_filter=paper_filter)

    # ✅ Handle empty retrieval (VERY IMPORTANT)
    if not chunks:
        return {
            "query": query,
            "answer": "No relevant information found in the indexed papers.",
            "sources": [],
            "chunks_used": [],
            "num_chunks": 0
        }

    # Step 2: Format context
    context = format_context(chunks)

    # ✅ Stronger prompt (reduces hallucination)
    prompt = f"""You are a scientific research assistant.

STRICT RULES:
- Use ONLY the provided context
- Do NOT use prior knowledge
- Do NOT hallucinate
- Every claim MUST have a citation [Source N]
- If information is missing, say "Not enough information"

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    # Step 3: Call Groq LLM
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE
    )

    answer_text = response.choices[0].message.content

    # ✅ Step 4: Safe metadata extraction (prevents crashes)
    sources = []
    for chunk in chunks:
        m = chunk.metadata

        file    = m.get("source", "unknown")
        page    = m.get("page", "?")
        paper   = m.get("paper_title", "unknown")
        section = m.get("section", "unknown")

        sources.append({
            "file": file,
            "paper": paper,
            "page": page,
            "section": section
        })

    # ✅ Deduplicate sources
    seen = set()
    unique_sources = []
    for s in sources:
        key = (s["file"], s["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return {
        "query": query,
        "answer": answer_text,
        "sources": unique_sources,
        "chunks_used": chunks,
        "num_chunks": len(chunks)
    }

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def pretty_print_response(response: dict):
    print("\n" + "="*60)
    print(f"❓ QUESTION: {response['query']}")
    print("="*60)
    print(f"\n📝 ANSWER:\n{response['answer']}")
    print(f"\n📚 SOURCES USED ({response['num_chunks']} chunks):")
    for s in response["sources"]:
        print(f"   • {s['file']} — Page {s['page']}")
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 AI Research Copilot — Phase 1 Test")
    print("Loading index from disk...")

    # IMPORTANT: use_openai=False (you used local embeddings earlier)
    store = load_index()

    print("✅ Index loaded. Ask questions about your paper(s).")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Your question: ").strip()

        if query.lower() in ("quit", "exit", "q"):
            break

        if not query:
            continue

        response = answer_question(query, store)
        pretty_print_response(response)