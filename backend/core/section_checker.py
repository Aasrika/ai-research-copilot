"""
section_checker.py
-------------------
Rule-based (no LLM) section-presence check for Mode 2 (Draft Critique).

This is a document-structure scan over the WHOLE draft, distinct from
document_processor.detect_section() which is a per-chunk (800-char)
heuristic used for retrieval reranking. The two intentionally use
different keyword tables: detect_section()'s SECTION_KEYWORDS folds
"conclusion" into "discussion" for reranking purposes, but the
user-facing section-presence check needs Conclusion and
Discussion/Limitations reported as two distinct required sections.
"""

import re
from typing import Optional

from core.critique_models import SectionCheckItem

REQUIRED_SECTIONS = {
    "Abstract":               ["abstract"],
    "Introduction":           ["introduction"],
    "Related Work":           ["related work", "related works", "background",
                                "literature review", "prior work"],
    "Methods":                ["method", "methodology", "approach", "proposed method",
                                "system design", "system architecture", "framework design"],
    "Experiments/Results":    ["experiment", "results", "evaluation", "empirical"],
    "Discussion/Limitations": ["discussion", "limitations", "threats to validity",
                                "analysis", "findings discussion"],
    "Conclusion":              ["conclusion", "concluding remarks", "summary",
                                "future scope", "future work"],
    "References":             ["references", "bibliography"],
}

# Minimum reasonable word count per section type. References is excluded
# from the "suspiciously short" judgment entirely (see check_sections) —
# a short reference list isn't a quality signal the way a short Methods
# section is.
MIN_WORD_FLOOR = {
    "Abstract": 50,
    "Introduction": 150,
    "Related Work": 100,
    "Methods": 150,
    "Experiments/Results": 100,
    "Discussion/Limitations": 80,
    "Conclusion": 40,
    "References": 0,
}

# ─────────────────────────────────────────────────────────────────────────────
# HEADING DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_ROMAN_NUMERALS = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
]
# Longest-first so the alternation doesn't stop at a shorter partial match
# (e.g. matching "X" inside "XIX").
_ROMAN_ALTERNATION = "|".join(sorted(_ROMAN_NUMERALS, key=len, reverse=True))

# A heading prefix is EITHER an Arabic numeral ("1.", "3.2"), a Roman
# numeral + period ("I.", "IV."), or a single uppercase letter + period
# ("A.", "C.") — all common section/subsection numbering conventions
# found across the sample papers.
_PREFIX_RE = rf"(?:\d+\.?\d*\.?|(?:{_ROMAN_ALTERNATION})\.|[A-Z]\.)\s*"

# A heading-like line: short, no trailing period, an optional prefix
# above, then a short run of words. ":", "&", "—" (em-dash), and "-"
# (hyphen) are allowed within the body since real headings in these
# papers use them ("6 CONCLUSION & DISCUSSION", "D. Proposed Method:
# Perceptual Loss Integration").
HEADING_RE = re.compile(rf"^(?:{_PREFIX_RE})?([A-Za-z][A-Za-z\s\-/:&—]{{1,55}})$")

# "Abstract—"/"Abstract:"/"Abstract " immediately followed by body text
# on the SAME line — common in IEEE two-column PDFs, where the whole
# line (heading + first sentence) is far longer than a normal heading.
_ABSTRACT_INLINE_RE = re.compile(r"^abstract\b[\s:—-]*", re.IGNORECASE)

_PREFIX_ONLY_RE = re.compile(rf"^(?:{_PREFIX_RE})")


def _is_heading_candidate(line: str) -> Optional[str]:
    line = line.strip()
    if not line or len(line) > 60 or line.endswith("."):
        return None
    match = HEADING_RE.match(line)
    if not match:
        return None
    return match.group(1).strip().lower()


def _looks_like_real_heading(line: str) -> bool:
    """
    Every genuine heading across the sample papers is EITHER numbered
    (Arabic/Roman/lettered prefix) OR written in ALL CAPS ("ABSTRACT",
    "REFERENCES"). Ordinary table-cell text or prose that happens to
    satisfy the loose HEADING_RE — e.g. "Ledger-only evaluation under
    idealized conditions", a table cell describing a COMPETING paper's
    evaluation, not this paper's own Results section — has neither
    trait. This filters those out without needing real document-
    outline/font-size information.
    """
    stripped = line.strip()
    if _PREFIX_ONLY_RE.match(stripped):
        return True
    letters = [c for c in stripped if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _is_abstract_inline_line(line: str) -> bool:
    """
    Bypasses the normal length/isolation checks entirely, since the
    "Abstract" marker itself is short even though the whole line isn't.
    """
    return bool(_ABSTRACT_INLINE_RE.match(line.strip()))


def _candidate_mask(lines: list[str]) -> list[bool]:
    return [_is_heading_candidate(line) is not None for line in lines]


def _is_isolated(mask: list[bool], idx: int) -> bool:
    """
    Accept a heading-candidate only if it's the FIRST line of its run of
    consecutive candidate-like lines.

    NOTE: the original design called for requiring a literal blank line
    before/after each heading. PyMuPDF's plain-text extraction inserts
    NO blank lines around headings in any of the three sample PDFs (0
    blank lines found across all of them), so that literal check would
    reject every heading, including genuine ones.

    A simple max-run-length guard was tried instead, but real headings
    can also sit in short runs: two-column PDF extraction sometimes
    places a heading immediately next to its own subsection heading with
    no body-text line breaking them up (e.g. "IV. RESULTS AND
    DISCUSSION" / "A. Experimental Setup" / body — a run of 3,
    indistinguishable by length from an author-info block or a short
    table header). The reliable signal is POSITION within the run, not
    length: a table's column-header row or an author block always has
    its distinguishing word as the 2nd+ item ("Dataset" then "Method"
    then "Flip Success"...), while a genuine heading is always the FIRST
    item a real paragraph break introduces.
    """
    return idx == 0 or not mask[idx - 1]


def _all_lines(pages: list[dict]) -> list[str]:
    lines: list[str] = []
    for page in pages:
        lines.extend(page["text"].split("\n"))
    return lines


def _scan_headings(lines: list[str]) -> dict[str, int]:
    """First line index (in document order) where each canonical section's
    heading is detected, if any."""
    found: dict[str, int] = {}

    # Abstract special-case runs first and unconditionally so it isn't
    # blocked by the generic length/isolation checks below.
    for idx, line in enumerate(lines):
        if _is_abstract_inline_line(line):
            found["Abstract"] = idx
            break

    mask = _candidate_mask(lines)
    for idx, line in enumerate(lines):
        if not mask[idx] or not _is_isolated(mask, idx):
            continue
        if not _looks_like_real_heading(line):
            continue
        heading = _is_heading_candidate(line)
        for section, keywords in REQUIRED_SECTIONS.items():
            if section in found:
                continue
            if any(kw in heading for kw in keywords):
                found[section] = idx
                # Deliberately no `break` — a single heading can satisfy
                # multiple canonical sections (e.g. "Results and
                # Discussion" matches both Experiments/Results and
                # Discussion/Limitations); they'll share one span below.
    return found


def find_references_page(pages: list[dict]) -> Optional[int]:
    """
    Page number (matching page["page"], 1-indexed) where a References/
    Bibliography heading is first found — used by section_check_agent to
    truncate the draft text before claim extraction. Page-level (not
    line-level) since that's all the caller needs.
    """
    for page in pages:
        for line in page["text"].split("\n"):
            heading = _is_heading_candidate(line)
            if heading and any(kw in heading for kw in REQUIRED_SECTIONS["References"]):
                return page["page"]
    return None


def check_sections(pages: list[dict]) -> list[SectionCheckItem]:
    lines = _all_lines(pages)
    found = _scan_headings(lines)

    # Sort matches by document position so we can compute each section's
    # span as "from its heading to the next DISTINCT detected heading".
    # Multiple sections can share the same start_idx (a combined heading
    # like "Results and Discussion") — they share the same span rather
    # than being sliced against each other.
    ordered = sorted(found.items(), key=lambda kv: kv[1])
    distinct_idxs = sorted(set(idx for _, idx in ordered))

    spans: dict[str, str] = {}
    for section, start_idx in ordered:
        later = [i for i in distinct_idxs if i > start_idx]
        end_idx = later[0] if later else len(lines)
        # Abstract's heading line usually has real body text fused into
        # it (see _is_abstract_inline_line), so it's included in the
        # span; other headings are standalone lines with no body content
        # of their own.
        span_start = start_idx if section == "Abstract" else start_idx + 1
        spans[section] = "\n".join(lines[span_start:end_idx])

    word_counts = {section: len(text.split()) for section, text in spans.items()}

    results = []
    for section in REQUIRED_SECTIONS:
        if section not in found:
            results.append(SectionCheckItem(
                section=section, present=False, word_count=0,
                is_suspiciously_short=False, reason="Section heading not found",
            ))
            continue

        wc = word_counts[section]

        if section == "References":
            # A short reference list isn't a quality problem the way a
            # short Methods section is — presence is what matters here.
            results.append(SectionCheckItem(
                section=section, present=True, word_count=wc,
                is_suspiciously_short=False, reason="",
            ))
            continue

        floor = MIN_WORD_FLOOR.get(section, 0)
        others = [w for s, w in word_counts.items() if s != section and s != "References"]
        avg_others = sum(others) / len(others) if others else 0

        below_floor = wc < floor
        below_relative = avg_others > 0 and wc < 0.15 * avg_others
        is_short = below_floor or below_relative

        reason_parts = []
        if below_floor:
            reason_parts.append(f"{wc} words (below floor of {floor})")
        if below_relative:
            reason_parts.append(f"{wc} words vs ~{avg_others:.0f} avg for other sections")

        results.append(SectionCheckItem(
            section=section, present=True, word_count=wc,
            is_suspiciously_short=is_short, reason="; ".join(reason_parts),
        ))

    return results
