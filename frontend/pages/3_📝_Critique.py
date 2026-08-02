"""
3_📝_Critique.py
-----------------
Streamlit multi-page app: this file auto-appears as a sidebar page.

Mode 2 (Draft Critique). Operates on the SAME active session as the
main page (st.session_state.active_session_id, set there — session
switching lives only in the main page's sidebar, so this page just
reads the current value rather than re-rendering a picker).

Runs the Phase 2 critique pipeline (backend/agents/critique_graph.py)
against the session's literature store and displays the resulting
CritiqueReport. No evaluation logging yet — that's Phase 6.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

from core import session_manager
from agents.critique_graph import run_critique


def render_session_banner():
    """Subtle, always-visible reminder of which session's data is on screen.
    Duplicated identically across app.py / Evaluation.py / Critique.py —
    every page in this app is already independent (own sys.path setup, no
    shared layout module), so a 4-line function isn't worth a new shared
    module just for this."""
    active_id = st.session_state.get("active_session_id")
    session = session_manager.get_session(active_id) if active_id else None
    if not session:
        st.caption("📁 No session selected — please create or select one from the main page sidebar.")
        return
    st.caption(f"📁 Active session: **{session['name']}**")


st.set_page_config(page_title="Draft Critique", page_icon="📝", layout="wide")

# ── Session State ──────────────────────────────────────────────────────────
for key, default in [
    ("critique_reports", {}),          # {session_id: CritiqueReport}
    ("draft_uploader_key_version", 0),
    ("just_critiqued_msg", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("📝 Draft Critique")
st.caption("Section-presence check + claim-vs-literature alignment for your draft paper.")
render_session_banner()

# ── Active session gate ─────────────────────────────────────────────────────
active_id = st.session_state.get("active_session_id")

if not active_id:
    st.warning("⬅️ No active session. Select or create one on the main page first.")
    st.stop()

literature_papers = session_manager.get_indexed_literature(active_id)

if not literature_papers:
    st.warning(
        "This session has no literature papers indexed yet. "
        "Upload literature papers via the main page's sidebar before running a critique."
    )
    st.stop()

st.caption(f"Literature in this session: {', '.join(f'`{p}`' for p in literature_papers)}")

# ── Draft upload ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("📄 Draft Paper")

existing_draft = session_manager.get_draft_path(active_id)
if existing_draft:
    st.caption(f"Current draft on file: `{existing_draft.name}`")

if st.session_state.just_critiqued_msg:
    st.success(st.session_state.just_critiqued_msg)
    st.session_state.just_critiqued_msg = None

uploaded_draft = st.file_uploader(
    "Upload your draft paper (PDF)", type="pdf",
    key=f"draft_uploader_{st.session_state.draft_uploader_key_version}",
)

run_clicked = st.button("📝 Run Critique", type="primary", disabled=not uploaded_draft)

if run_clicked:
    try:
        with st.spinner("Running critique pipeline — this can take 30-60 seconds (multiple LLM calls)..."):
            draft_path = session_manager.set_draft(active_id, uploaded_draft.name, uploaded_draft.read())

            # A new draft invalidates any previously cached report for this session.
            st.session_state.critique_reports.pop(active_id, None)

            literature_store = session_manager.load_literature_store(active_id)
            report = run_critique(
                session_id=active_id,
                draft_path=str(draft_path),
                literature_store=literature_store,
            )
            st.session_state.critique_reports[active_id] = report

    except Exception as e:
        # Node-level Groq failures are already handled inside the critique
        # graph (fallback to empty claims / SILENT verdicts) — this catches
        # anything else (rate limits, an unreadable/corrupt PDF, etc.) so the
        # user sees a plain message instead of a raw traceback.
        print(f"❌ Critique pipeline error: {e}")
        st.error(
            "Something went wrong while running the critique. This can happen with a "
            "Groq API rate limit, a temporary connection issue, or a PDF that couldn't "
            "be parsed. Please try again in a moment."
        )

    else:
        # Bump the uploader's key so it remounts empty, same pattern as the
        # literature uploader on the main page. Only on success — on failure
        # the uploaded file stays in place so the user can just retry.
        st.session_state.just_critiqued_msg = f"✅ Critique complete for `{uploaded_draft.name}`!"
        st.session_state.draft_uploader_key_version += 1
        st.rerun()

# ── Report display ──────────────────────────────────────────────────────────
report = st.session_state.critique_reports.get(active_id)

if not report:
    st.info("Upload a draft above and click **Run Critique** to generate a report.")
    st.stop()

st.divider()
st.subheader("📋 Critique Report")
st.caption(f"Draft: `{report.draft_filename}` · Generated: {report.generated_at}")

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Missing Sections", report.summary.get("missing_sections", 0))
k2.metric("Supported Claims", report.summary.get("supported", 0))
k3.metric("Contradicted Claims", report.summary.get("contradicted", 0))
k4.metric("Citation Suggestions", report.summary.get("citation_suggestions", 0))

# ── Section presence check ──────────────────────────────────────────────────
st.markdown("### 📑 Section Presence Check")

section_rows = []
for s in report.section_checks:
    if not s.present:
        icon = "❌"
    elif s.is_suspiciously_short:
        icon = "⚠️"
    else:
        icon = "✅"
    section_rows.append({
        "Status": icon,
        "Section": s.section,
        "Words": s.word_count,
        "Notes": s.reason,
    })

st.dataframe(section_rows, use_container_width=True, hide_index=True)

# ── Claim alignments ─────────────────────────────────────────────────────────
st.markdown("### 🔍 Claim–Literature Alignment")

if not report.claims:
    st.info("No claims were extracted from this draft.")
else:
    verdict_order = {"CONTRADICTED": 0, "SUPPORTED": 1, "SILENT": 2}
    verdict_icon = {"CONTRADICTED": "🔴", "SUPPORTED": "🟢", "SILENT": "⚪"}
    sorted_claims = sorted(report.claims, key=lambda c: verdict_order.get(c.verdict, 3))

    for claim in sorted_claims:
        icon = verdict_icon.get(claim.verdict, "⚪")
        title = f"{icon} {claim.claim_text[:90]}{'...' if len(claim.claim_text) > 90 else ''}"

        with st.expander(title):
            if claim.verdict == "SUPPORTED":
                st.success(f"Verdict: SUPPORTED")
            elif claim.verdict == "CONTRADICTED":
                st.error(f"Verdict: CONTRADICTED")
            else:
                st.warning(f"Verdict: SILENT")

            st.caption(f"Confidence: **{claim.confidence}** · Type: {claim.claim_type} · Source: {claim.source_section}")

            st.markdown(f"**Claim:** {claim.claim_text}")

            if claim.notes:
                st.caption(f"Note: {claim.notes}")

            if claim.evidence:
                st.markdown("**Evidence:**")
                for ev in claim.evidence:
                    st.caption(f"📄 {ev.paper_title} · p.{ev.page} · §{ev.section}")
                    st.markdown(f"> {ev.snippet}")
            else:
                st.caption("No literature evidence retrieved for this claim.")

            if claim.suggested_citation:
                st.info(f"💡 Consider citing: **{claim.suggested_citation}**")

# ── Export ────────────────────────────────────────────────────────────────────
st.divider()
st.download_button(
    "⬇️ Export Report (JSON)",
    data=report.model_dump_json(indent=2),
    file_name=f"critique_{report.draft_filename}.json",
    mime="application/json",
)
