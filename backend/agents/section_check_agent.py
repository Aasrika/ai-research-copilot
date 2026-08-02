"""
section_check_agent.py
-----------------------
Node 1 in the critique LangGraph — rule-based, no LLM.

Runs first and unconditionally: it can't "fail" the way an LLM critic
can, so nothing gates on it — its output just feeds into the final
report. It also builds the section-tagged, References-truncated draft
text that the claim-extraction node consumes, reusing the same
heading scan section_checker.py already did (find_references_page)
rather than re-deriving section boundaries a second way.
"""

from pathlib import Path

from core.document_processor import parse_pdf, detect_section
from core.section_checker import check_sections, find_references_page
from agents.critique_state import CritiqueState

# Keep claim extraction within a safe budget. This isn't primarily a
# context-window limit (the model supports far more) — testing showed
# that beyond ~12K characters, the model can start "continuing" the
# draft's own content instead of following the JSON instructions,
# especially when truncation lands mid-table. 8K keeps a comfortable
# margin below where that behavior appeared.
MAX_DRAFT_CHARS = 8000

# Fallback if no References heading (or no headings at all) are detected —
# use at most this many pages of raw text instead of the whole draft.
FALLBACK_MAX_PAGES = 6


def section_check_node(state: CritiqueState) -> dict:
    pages = parse_pdf(state["draft_path"])

    section_checks = [item.model_dump() for item in check_sections(pages)]

    references_page = find_references_page(pages)

    tagged_pages = []
    for page in pages:
        if references_page is not None and page["page"] >= references_page:
            break
        section = detect_section(page["text"])
        tagged_pages.append(f"[Section: {section}]\n{page['text']}")

    # If nothing survived the References cut (e.g. no headings detected at
    # all in an unusually formatted draft), fall back to the first few pages.
    if not tagged_pages:
        tagged_pages = [
            f"[Section: {detect_section(p['text'])}]\n{p['text']}"
            for p in pages[:FALLBACK_MAX_PAGES]
        ]

    draft_text = "\n\n".join(tagged_pages)[:MAX_DRAFT_CHARS]

    found = sum(1 for s in section_checks if s["present"])
    print(f"\n📝 Section check: {found}/{len(section_checks)} required sections found "
          f"({Path(state['draft_path']).name})")

    return {
        "draft_pages": pages,
        "draft_text": draft_text,
        "section_checks": section_checks,
    }
