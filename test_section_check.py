"""
test_section_check.py
----------------------
Standalone CLI smoke test for section_checker.py, mirroring the
existing test_graph.py convention.

Covers:
  1. Full-paper detection against the three real sample PDFs.
  2. A deliberately-truncated synthetic paper (Methods section entirely
     removed) to verify check_sections() correctly flags it as missing
     while everything else is still detected.
"""

import sys
sys.path.insert(0, "backend")

from dotenv import load_dotenv
load_dotenv()

from core.document_processor import parse_pdf
from core.section_checker import check_sections

PDFS = [
    "data/papers/test_paper.pdf",
    "data/papers/paper2.pdf",
    "data/papers/EdgeBlockAI paper .pdf",
]

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — full-paper detection against real sample PDFs
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("TEST 1 — full-paper detection")
print("=" * 60)

for path in PDFS:
    pages = parse_pdf(path)
    results = check_sections(pages)
    found = sum(1 for r in results if r.present)

    print(f"\n{path}: {found}/{len(results)} sections found")
    for r in results:
        status = "PRESENT" if r.present else "MISSING"
        flag = " [SUSPICIOUSLY SHORT]" if r.is_suspiciously_short else ""
        print(f"  {r.section:22s} {status}{flag}")

    assert found >= 7, f"{path}: expected at least 7/8 sections, got {found}"

print("\n✅ All three papers detected at least 7/8 required sections.")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — deliberately-truncated paper (Methods section removed)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 2 — deliberately-truncated paper (Methods section removed)")
print("=" * 60)

_intro_filler = (
    "This is the introduction section with enough words to clear the "
    "minimum floor so it is not flagged as suspiciously short. "
) * 6
_related_filler = (
    "This is the related work section discussing prior literature at "
    "reasonable length so it clears its own minimum floor. "
) * 6
_results_filler = (
    "This is the results and discussion section describing findings at "
    "reasonable length so it clears its own minimum floor. "
) * 6
_conclusion_filler = "This is the conclusion section wrapping up the paper. " * 4

truncated_pages = [
    {"page": 1, "text": (
        "Abstract—This paper presents a synthetic example used only to "
        "test the section checker's missing-section detection.\n"
        "I. INTRODUCTION\n"
        f"{_intro_filler}"
    )},
    {"page": 2, "text": (
        "II. RELATED WORK\n"
        f"{_related_filler}\n"
        "IV. RESULTS AND DISCUSSION\n"
        f"{_results_filler}"
    )},
    {"page": 3, "text": (
        "V. CONCLUSION AND FUTURE WORK\n"
        f"{_conclusion_filler}\n"
        "REFERENCES\n"
        "[1] Example citation one.\n[2] Example citation two.\n"
    )},
]

# No Methods/Methodology/Approach/etc. heading anywhere above — deliberate.

results = check_sections(truncated_pages)
by_section = {r.section: r for r in results}

methods = by_section["Methods"]
print(f"\nMethods: present={methods.present}, reason={methods.reason!r}")
assert not methods.present, "Expected Methods to be MISSING in the truncated paper"

other_present = sorted(s for s, r in by_section.items() if s != "Methods" and r.present)
print(f"Other sections found: {other_present}")
assert len(other_present) == 7, f"Expected all 7 non-Methods sections present, got {other_present}"

print("\n✅ Truncated paper correctly flags Methods as missing while every other section is still detected.")
