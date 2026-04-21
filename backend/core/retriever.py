from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from core.config import TOP_K, MMR_LAMBDA, SECTION_KEYWORDS


# ─────────────────────────────────────────────────────────────────────────────
# QUERY INTENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

QUERY_SECTION_HINTS = {
    "limitations":        ["discussion"],
    "future work":        ["discussion"],
    "contribution":       ["abstract", "introduction"],
    "dataset":            ["methods"],
    "baseline":           ["results", "methods"],
    "accuracy":           ["results"],
    "results":            ["results"],
    "how does":           ["methods"],
    "architecture":       ["methods"],
    "related":            ["related_work", "introduction"],
    "compared to":        ["results", "discussion"],
    "improvement":        ["results", "discussion"],
    "why":                ["introduction", "discussion"],
    "motivation":         ["introduction"],
}

def infer_section_filter(query: str) -> list[str] | None:
    """
    Look at the query and guess which section is most relevant.

    WHY THIS HELPS:
    "What are the limitations?" should search the Discussion section,
    not the Methods section. Without this, FAISS might return methods
    chunks that happen to contain the word "limit" (e.g., "we limit
    our analysis to..."), which is semantically wrong.

    Returns None if no strong signal → no filtering → search everything.
    """
    q = query.lower()
    for hint, sections in QUERY_SECTION_HINTS.items():
        if hint in q:
            return sections
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MMR RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query:        str,
    vector_store: FAISS,
    k:            int  = TOP_K,
    paper_filter: str  = None,  # restrict to a specific paper title
    section_hint: bool = True,  # auto-detect section from query
) -> list[Document]:
    """
    Retrieve the top-k most relevant AND diverse chunks for a query.

    ── TECHNIQUE 1: MMR (Maximal Marginal Relevance) ──────────────────────
    Problem with plain similarity search:
      Your top 5 chunks might all say the same thing — the same paragraph
      repeated in slightly different words. You waste context window space
      and the LLM gets a skewed view.

    MMR solution:
      Iteratively select chunks that are:
        - Similar to the query (relevance)  ← weighted by MMR_LAMBDA
        - Dissimilar to already-selected chunks (diversity) ← 1 - MMR_LAMBDA

    Formula: MMR(d) = λ·sim(d, query) - (1-λ)·max_sim(d, selected)

    With MMR_LAMBDA=0.7: 70% weight on relevance, 30% on diversity.
    This is the sweet spot for research papers — you want relevant chunks
    but don't want the same paragraph 3 times.

    ── TECHNIQUE 2: Section Filtering ─────────────────────────────────────
    After MMR retrieval, if we detected a section hint, we RERANK results
    to push matching-section chunks to the top.

    We don't hard-filter (discard non-matching chunks) because:
      - Section detection isn't perfect
      - Sometimes the answer spans sections
    Reranking is a softer, safer approach.
    """
    # Step 1: Build metadata filter dict for LangChain
    filter_dict = {}
    if paper_filter:
        filter_dict["paper_title"] = paper_filter

    # Step 2: MMR retrieval (fetch 2x k, then rerank down to k)
    fetch_k = min(k * 2, 20)  # fetch more candidates for MMR to work well

    try:
        if filter_dict:
            results = vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k,
                lambda_mult=MMR_LAMBDA,
                filter=filter_dict,
            )
        else:
            results = vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k,
                lambda_mult=MMR_LAMBDA,
            )
    except Exception as e:
        # Fallback to plain similarity search if MMR fails
        print(f"⚠️  MMR failed ({e}), falling back to similarity search")
        results = vector_store.similarity_search(query, k=k)

    # Step 3: Section-based reranking
    if section_hint:
        preferred_sections = infer_section_filter(query)
        if preferred_sections:
            # Push preferred-section chunks to front, preserve order within groups
            priority   = [d for d in results if d.metadata.get("section") in preferred_sections]
            secondary  = [d for d in results if d.metadata.get("section") not in preferred_sections]
            results    = (priority + secondary)[:k]
            if priority:
                print(f"  📌 Section hint → prioritizing: {preferred_sections}")

    # Step 4: Log what we retrieved (useful for debugging)
    print(f"\n🔍 Retrieved {len(results)} chunks for: '{query[:55]}...'")
    for doc in results:
        m = doc.metadata
        print(f"   [{m['paper_title']} | p.{m['page']} | §{m['section']}]  "
              f"{doc.page_content[:60].strip()}...")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT CHUNKS FOR LLM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def format_context(chunks: list[Document]) -> str:
    """
    Format retrieved chunks into a clean context block for the LLM prompt.
    Each source is numbered so the LLM can cite [Source N].
    """
    parts = []
    for i, doc in enumerate(chunks, 1):
        m = doc.metadata
        header = (f"[Source {i} | Paper: {m['paper_title']} | "
                  f"Page: {m['page']} | Section: {m['section']}]")
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)