import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from core import session_manager
from agents.graph import run_pipeline
from agents.idea_agent import generate_research_ideas
from agents.comparison_agent import run_comparison, ASPECT_QUERIES
from evaluation.logger import RunRecord, log_run

load_dotenv()


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


def render_continue_callout():
    """Shown above the hero on the landing page when the user has a "last
    active" session from earlier in this browser session (see
    _activate_session, defined below). In-memory only — not persisted to
    disk, since it's meant to reflect this browser session, not a durable
    preference. If that session has since been deleted, this silently
    shows nothing rather than pointing at a dead session."""
    last_id = st.session_state.last_active_session_id
    if not last_id:
        return
    session = session_manager.get_session(last_id)
    if not session:
        return  # deleted since it was last active — skip gracefully

    st.markdown(
        f"""
        <div style="background-color:{_CALLOUT_BG}; border-radius:16px; text-align:center;
                    padding:1.75rem 2rem; margin-bottom:1.5rem;">
            <div style="font-size:1.2rem; font-weight:600;">
                👋 Welcome back! Continue with your last session:
                <span style="color:{_CALLOUT_ACCENT};">{session['name']}</span> →
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, col_btn, _ = st.columns([2, 1, 2])
    with col_btn:
        if st.button("→ Continue", use_container_width=True, key="continue_last_session_btn"):
            _activate_session(last_id)
            st.rerun()


# Shared violet tints for the landing page's "chapter" bands/cards — derived
# from config.toml's primaryColor (#8B5CF6) at low opacity so they read as
# subtle background variation rather than competing UI chrome. Kept as
# module constants so the hero band, card fill, and CTA band can't drift
# out of sync with each other.
_ACCENT_RGB = "139,92,246"           # 8B5CF6 as an rgb() triple, for rgba() use
_BAND_BG = f"rgba({_ACCENT_RGB},0.08)"   # hero + CTA bands
_CARD_BG = f"rgba({_ACCENT_RGB},0.14)"   # feature/step/strength cards — a clearly-visible purple wash
_ACCENT = "#8B5CF6"
_CARD_BORDER = f"rgba({_ACCENT_RGB},0.4)"     # purple-tinted border (was a neutral gray) so cards read as purple, not just dark boxes
_CARD_SHADOW = f"0 4px 24px rgba({_ACCENT_RGB},0.15)"  # soft glow for depth against the near-black page background
_MUTED_TEXT = "#9CA3AF"        # description/subtitle text — explicit color instead of opacity, for readability on dark
_CARD_STYLE = (
    f"background-color:{_CARD_BG}; border:1px solid {_CARD_BORDER}; "
    f"border-radius:14px; box-shadow:{_CARD_SHADOW};"
)

# Dedicated teal accent for the "continue where you left off" callout only —
# deliberately distinct from the site-wide purple above, so it reads as a
# special notice rather than blending into the rest of the landing page.
_CALLOUT_RGB = "20,184,166"        # 14B8A6
_CALLOUT_BG = f"rgba({_CALLOUT_RGB},0.10)"
_CALLOUT_ACCENT = "#14B8A6"


def _feature_card_html(icon: str, title: str, desc: str) -> str:
    return f"""
    <div style="{_CARD_STYLE} padding:2rem; margin-bottom:1.5rem; min-height:250px;">
        <span style="font-size:3rem; display:block; text-align:center;">{icon}</span>
        <div style="font-size:1.3rem; font-weight:700; text-align:center; margin-top:0.75rem;">
            {title}
        </div>
        <div style="color:{_MUTED_TEXT}; text-align:center; margin-top:0.75rem; line-height:1.55;">
            {desc}
        </div>
    </div>
    """


def _step_card_html(number: int, title: str, desc: str) -> str:
    return f"""
    <div style="{_CARD_STYLE} text-align:center; padding:1.5rem; margin-bottom:1rem;">
        <div style="font-size:3rem; font-weight:700; color:{_ACCENT};">{number}</div>
        <div style="font-size:1.2rem; font-weight:700; margin-top:0.5rem;">{title}</div>
        <div style="color:{_MUTED_TEXT}; margin-top:0.5rem; line-height:1.5;">{desc}</div>
    </div>
    """


def _strength_card_html(icon: str, title: str, desc: str) -> str:
    return f"""
    <div style="{_CARD_STYLE} text-align:center; padding:1.5rem; margin-bottom:1rem;">
        <span style="font-size:2.5rem; display:block; text-align:center;">{icon}</span>
        <div style="font-size:1.1rem; font-weight:700; margin-top:0.75rem;">{title}</div>
        <div style="color:{_MUTED_TEXT}; margin-top:0.5rem; line-height:1.5;">{desc}</div>
    </div>
    """


def _section_heading_html(title: str, subtitle: str = "") -> str:
    subtitle_html = f'<div style="color:{_MUTED_TEXT}; margin-top:0.25rem;">{subtitle}</div>' if subtitle else ""
    return f"""
    <div style="text-align:center; margin-top:4rem; margin-bottom:2rem;">
        <div style="font-size:2rem; font-weight:700;">{title}</div>
        {subtitle_html}
    </div>
    """


def render_landing_page():
    """Shown only when no session is active — replaces the tabs entirely
    (see the call site in the Main section). Colors/spacing here are
    additive page content, not theme overrides, so this doesn't fight
    .streamlit/config.toml the way the old hardcoded CSS block used to:
    headings rely on the theme's own text color, description/subtitle text
    uses the explicit _MUTED_TEXT color, and the deliberate violet accents
    (_ACCENT/_BAND_BG/_CARD_BG/_CARD_BORDER above) are all derived from
    config.toml's primaryColor #8B5CF6."""

    # ── Section 1: Hero ──────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="background-color:{_BAND_BG}; border-radius:16px; text-align:center;
                    padding:5rem 2rem 4rem 2rem;">
            <div style="font-size:4rem; font-weight:700; line-height:1.15; margin-bottom:1rem;">
                AI Research Copilot
            </div>
            <div style="font-size:1.5rem; color:{_MUTED_TEXT}; margin-bottom:1.5rem;">
                Your AI copilot for reading, comparing, and reviewing research papers.
            </div>
            <div style="font-size:1.05rem; color:{_MUTED_TEXT}; max-width:700px; margin:0 auto; line-height:1.7;">
                Whether you're doing a literature review, comparing methodologies,
                brainstorming follow-up research, or double-checking your own draft
                against the field — this tool helps you engage with research papers
                faster and more rigorously.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section 2: What you can do (2x2 feature grid) ───────────────────
    st.markdown(
        _section_heading_html("What you can do", "Four ways to work with your literature"),
        unsafe_allow_html=True,
    )

    features = [
        ("💬", "Ask",
         "Query your literature with grounded, cited answers. The system retrieves "
         "relevant passages, generates responses with inline citations, and passes "
         "each answer through a critic loop that catches hallucinations and vague "
         "statements before showing you the result."),
        ("🎨", "Compare",
         "Side-by-side comparison of two papers. Choose from preset dimensions like "
         "datasets, methods, and results — or type a custom comparison focus. Get a "
         "structured, tabular output that makes differences easy to scan."),
        ("💡", "Follow-up Ideas",
         "Surface research gaps, open questions, and concrete follow-up experiment "
         "ideas from any paper. Useful when planning your own research that builds "
         "on existing work."),
        ("📝", "Draft Critique",
         "Section-presence check plus claim-vs-literature alignment for your draft "
         "paper. See which of your claims are supported, contradicted, or "
         "unaddressed by the literature you uploaded. Get citation suggestions for "
         "gaps you didn't cite."),
    ]

    grid_cols = st.columns(2) + st.columns(2)
    for col, (icon, title, desc) in zip(grid_cols, features):
        with col:
            st.markdown(_feature_card_html(icon, title, desc), unsafe_allow_html=True)

    # Soft divider — a hard st.divider() line reads too harshly against the
    # accent-tinted bands above/below it.
    st.markdown(
        f'<hr style="border:none; border-top:1px solid rgba({_ACCENT_RGB},0.2); '
        f'margin:3.5rem 0 1rem 0;">',
        unsafe_allow_html=True,
    )

    # ── Section 3: How it works (3-step workflow) ───────────────────────
    st.markdown(
        _section_heading_html("How it works", "Three steps to get started"),
        unsafe_allow_html=True,
    )

    steps = [
        ("Create a session",
         "Start a new workspace for your research topic. Sessions are isolated — "
         "different topics don't mix."),
        ("Upload your literature",
         "Add PDF research papers to your session. They're indexed automatically "
         "for search and citation."),
        ("Ask, compare, ideate, or critique",
         "Use any of the four tools with your indexed papers. Upload your own "
         "draft when you're ready for critique."),
    ]
    step_cols = st.columns(3)
    for i, (col, (title, desc)) in enumerate(zip(step_cols, steps), start=1):
        with col:
            st.markdown(_step_card_html(i, title, desc), unsafe_allow_html=True)

    # ── Section 4: Why this tool ─────────────────────────────────────────
    st.markdown(_section_heading_html("What makes it different"), unsafe_allow_html=True)

    strengths = [
        ("🔒", "Grounded, not hallucinated",
         "Every answer cites specific passages. A critic model verifies responses "
         "against sources before you see them."),
        ("📊", "Transparent evaluation",
         "Built-in dashboard shows pass rates, retry patterns, and cost per query. "
         "See how the system is performing, not just guess."),
        ("🎯", "Session-isolated",
         "Each research topic gets its own workspace with its own papers. No "
         "cross-contamination between different projects."),
    ]
    strength_cols = st.columns(3)
    for col, (icon, title, desc) in zip(strength_cols, strengths):
        with col:
            st.markdown(_strength_card_html(icon, title, desc), unsafe_allow_html=True)

    # ── Section 5: Get-started CTA ───────────────────────────────────────
    cta_text = (
        "← Select a session from the sidebar to continue"
        if session_manager.list_sessions()
        else "← Create your first session in the sidebar to begin"
    )
    st.markdown(
        f"""
        <div style="background-color:{_BAND_BG}; border-radius:16px; text-align:center;
                    padding:3rem 2rem; margin-top:4rem;">
            <div style="font-size:2rem; font-weight:700; margin-bottom:1rem;">
                Ready to start?
            </div>
            <div style="font-size:1.3rem; font-weight:600; color:{_ACCENT};">
                {cta_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Colors/border-radius/metric styling now come entirely from
# .streamlit/config.toml (the old dark/purple theme this block used to
# hardcode — #7C3AED buttons, #111827 metric tiles — has been replaced by
# the warm light theme there; keeping this would fight with it). Only the
# layout-only tweak (not a theme/color concern) stays.
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)




# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Research Copilot", page_icon="🔬", layout="wide")

# ── Session State ──────────────────────────────────────────────────────────
# chat_histories / idea_results_by_session / compare_results_by_session are
# keyed by session_id (dict[session_id, value]) so switching sessions shows
# each session's own history instead of leaking into another's. vector_store
# and indexed_papers stay flat — they're already fully refreshed on every
# session switch below, so namespacing them would be redundant.
for key, default in [
    ("active_session_id", None),
    ("vector_store", None),
    ("indexed_papers", []),
    ("chat_histories", {}),
    ("idea_results_by_session", {}),
    ("compare_results_by_session", {}),
    ("uploader_key_version", 0),
    ("just_indexed_msg", None),
    ("last_active_session_id", None),   # in-memory only — powers the landing "Continue" callout
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _activate_session(session_id: Optional[str]) -> None:
    """Sets the active session and (re)loads its literature store. Passing
    None deactivates (returns to the landing page). Updates
    last_active_session_id only when activating a real session — Home and
    delete-session both pass None here and deliberately leave the tracker
    alone, since that's what lets the landing page's "Continue where you
    left off" callout still find it afterward."""
    st.session_state.active_session_id = session_id
    st.session_state.vector_store = (
        session_manager.load_literature_store(session_id) if session_id else None
    )
    st.session_state.indexed_papers = (
        session_manager.get_indexed_literature(session_id) if session_id else []
    )
    if session_id:
        st.session_state.last_active_session_id = session_id


@st.dialog("Delete session?")
def _confirm_delete_session(session_id: str, session_name: str):
    st.warning(
        f"Delete session **'{session_name}'** and all its papers, indexes, "
        f"and critique reports? This cannot be undone."
    )
    col_cancel, col_confirm = st.columns(2)

    if col_cancel.button("Cancel", use_container_width=True):
        st.rerun()

    if col_confirm.button("🗑️ Delete", type="primary", use_container_width=True):
        session_manager.delete_session(session_id)

        # Clean up every session-scoped state dict so a deleted session's
        # data doesn't linger as a memory leak. critique_reports lives on
        # the Critique page and may not exist yet if that page was never
        # visited this browser session — .get(..., {}) handles that safely.
        for state_key in (
            "chat_histories", "idea_results_by_session",
            "compare_results_by_session", "critique_reports",
        ):
            st.session_state.get(state_key, {}).pop(session_id, None)

        # Per-paper checkbox selections live as individual widget keys (see
        # the paper-selection design), not a single dict — clean those up
        # with a prefix scan instead of a dict.pop().
        prefix = f"paper_cb_{session_id}_"
        for key in [k for k in st.session_state.keys() if k.startswith(prefix)]:
            del st.session_state[key]
        st.session_state.pop(f"select_all_cb_{session_id}", None)

        _activate_session(None)
        st.rerun()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Copilot")
    st.caption("Multi-agent RAG for research papers")
    st.divider()

    st.subheader("🗂️ Session")
    sessions = session_manager.list_sessions()
    session_names = {s["id"]: s["name"] for s in sessions}

    # Distinct label from Streamlit's own built-in page navigator (which is
    # left as-is, above this sidebar, for reaching Evaluation/Critique) —
    # this button is an in-app action, not a page link. Only shown when
    # there's actually a session active to go "back" from — with none
    # active (already on the landing page), it has nothing to do.
    if (
        sessions
        and st.session_state.active_session_id is not None
        and st.button("🏠 Back to Home", use_container_width=True)
    ):
        _activate_session(None)
        st.rerun()

    selected_id = None
    if sessions:
        ids = list(session_names.keys())
        # None is a real option here (not just a fallback) so the picker can
        # start genuinely unselected on fresh startup / after Home, instead
        # of always resolving to some session (which is what caused the old
        # auto-select-first-session behavior — st.selectbox can't return
        # "nothing" on its own; it needs an actual option for that).
        options = [None] + ids
        current = st.session_state.active_session_id
        default_index = options.index(current) if current in options else 0
        selected_id = st.selectbox(
            "Active session",
            options,
            index=default_index,
            format_func=lambda sid: "— Select a session —" if sid is None else session_names[sid],
        )
    else:
        st.info("No sessions yet — create one below.")

    # Only reload the store when the active session actually changes, so we
    # don't re-hit disk on every widget interaction/rerun.
    if selected_id != st.session_state.active_session_id:
        _activate_session(selected_id)

    active_id = st.session_state.active_session_id

    if active_id and st.button("🗑️ Delete session"):
        _confirm_delete_session(active_id, session_names.get(active_id, active_id))

    with st.expander("➕ New session"):
        new_name = st.text_input("Session name", key="new_session_name")
        if st.button("Create session", disabled=not new_name.strip()):
            record = session_manager.create_session(new_name)
            _activate_session(record["id"])
            st.rerun()

    if active_id:
        st.divider()
        st.caption(f"Active session: **{session_names.get(active_id, active_id)}**")

        if st.session_state.just_indexed_msg:
            st.success(st.session_state.just_indexed_msg)
            st.session_state.just_indexed_msg = None

        uploaded = st.file_uploader(
            "📄 Upload literature PDFs", type="pdf", accept_multiple_files=True,
            key=f"literature_uploader_{st.session_state.uploader_key_version}",
        )

        if uploaded and st.button("⚡ Index Papers", type="primary"):
            with st.spinner("Indexing papers..."):
                files = [(f.name, f.read()) for f in uploaded]
                store = session_manager.add_literature(active_id, files)
                st.session_state.vector_store = store
                st.session_state.indexed_papers = session_manager.get_indexed_literature(active_id)

            # Bump the uploader's key so it remounts empty, and rerun so the
            # cleared uploader (and the "Papers in this session" update) show
            # immediately instead of leaving the just-indexed files sitting
            # in the upload area.
            st.session_state.just_indexed_msg = f"✅ Indexed {len(uploaded)} paper(s)!"
            st.session_state.uploader_key_version += 1
            st.rerun()

        if st.session_state.indexed_papers:
            st.divider()
            st.subheader("📚 Papers in this session")
            st.caption("Select which ones to include in your questions")

            known_papers = st.session_state.indexed_papers
            select_all_key = f"select_all_cb_{active_id}"

            if select_all_key not in st.session_state:
                st.session_state[select_all_key] = True

            def _toggle_all_papers(select_all_key=select_all_key,
                                    known_papers=known_papers, active_id=active_id):
                # on_change callback — the correct way to cascade a change to
                # OTHER widgets' values in Streamlit. Setting session_state
                # via a dynamic `value=` param (like the old prefill pattern)
                # is what caused the Issue 2 bug; this avoids that entirely.
                new_val = st.session_state[select_all_key]
                for p in known_papers:
                    st.session_state[f"paper_cb_{active_id}_{p}"] = new_val

            st.checkbox("Select all", key=select_all_key, on_change=_toggle_all_papers)

            for p in known_papers:
                cb_key = f"paper_cb_{active_id}_{p}"
                if cb_key not in st.session_state:
                    st.session_state[cb_key] = True  # default: all papers selected
                st.checkbox(p, key=cb_key)
    else:
        st.warning("Create or select a session to begin.")

    st.divider()
    with st.expander("⚙️ Advanced retrieval settings"):
        top_k = st.slider(
            "Retrieval depth", 3, 10, 5,
            help=(
                "How many text chunks the system retrieves per question. "
                "Higher = more context but may include less-relevant passages. "
                "Default 5 works well for most papers."
            ),
        )

# ── Main ──────────────────────────────────────────────────────────────────
if st.session_state.active_session_id is None:
    render_continue_callout()
    render_landing_page()
    st.stop()

st.title("🔬 AI Research Copilot")
st.caption("Multi-agent RAG + Self-correction + Research reasoning")
render_session_banner()

if st.session_state.vector_store is None:
    st.warning("⬅️ Upload literature papers into this session to start")
    st.stop()

active_id = st.session_state.active_session_id

# Papers currently checked in the sidebar (widget keys are the source of
# truth — see the sidebar's paper-checkbox section). Defaults to True for
# safety, though the sidebar always sets these before this point is reached.
selected_papers = [
    p for p in st.session_state.indexed_papers
    if st.session_state.get(f"paper_cb_{active_id}_{p}", True)
]

if not selected_papers:
    st.warning("⚠️ Select at least one paper in the sidebar to continue.")
    st.stop()

# None (no filter) when everything is checked, matching the original
# unfiltered "All papers" behavior exactly; a list only when the user has
# narrowed the selection.
paper_filter = None if len(selected_papers) == len(st.session_state.indexed_papers) else selected_papers

tab_qa, tab_compare, tab_ideas = st.tabs([
    "💬 Ask",
    "⚖️ Compare",
    "💡 Follow-up Ideas"
])

# ════════════════════════════════════════════════════════════════════════════
# 💬 TAB 1 — Q&A
# ════════════════════════════════════════════════════════════════════════════
with tab_qa:
    st.subheader("Ask anything")

    # Quick prompts
    cols = st.columns(4)
    prompts = [
        "What is the main contribution?",
        "What datasets were used?",
        "What are the limitations?",
        "How does this compare to prior work?",
    ]
    for col, p in zip(cols, prompts):
        if col.button(p):
            st.session_state["qa_query_input"] = p

    # A stable, explicit key — not value=st.session_state.pop(...) — is what
    # makes the quick-prompt buttons actually work. Without a key, Streamlit
    # re-applies whatever `value=` is passed on EVERY rerun; since "prefill"
    # was a one-shot pop(), the very next rerun (clicking Ask) saw value=""
    # again and silently reset the query to empty — the exact query became
    # "" right before run_pipeline was called, hence the blank "You:" line
    # and the unrelated-looking answer (an empty-string similarity search).
    #
    # Clearing this box after Ask needs one more step than the quick-prompt
    # fix: Streamlit raises StreamlitAPIException if you write to
    # session_state[key] AFTER that widget has already been instantiated in
    # the SAME script run (the Ask button is below this text_area). So a
    # "please clear" flag is set instead, and honored here — BEFORE the
    # widget is created — on the next run.
    if st.session_state.get("_clear_qa_input"):
        st.session_state["qa_query_input"] = ""
        st.session_state["_clear_qa_input"] = False

    if "qa_query_input" not in st.session_state:
        st.session_state["qa_query_input"] = ""

    query = st.text_area(
        "Your question:", key="qa_query_input",
        placeholder="Ask a question about the selected papers...",
    )

    if st.button("🔍 Ask", type="primary", disabled=not query.strip()):
        with st.spinner("Running multi-agent pipeline..."):
            state = run_pipeline(
                query=query,
                vector_store=st.session_state.vector_store,
                k=top_k,
                paper_filter=paper_filter,
                max_retries=2,
                session_id=active_id,
            )

        st.session_state.chat_histories.setdefault(active_id, []).append(state)

        # Can't clear qa_query_input directly here — the widget already
        # rendered above in this same run. Flag it instead; the check at
        # the top of this tab handles the actual clearing on the rerun.
        st.session_state["_clear_qa_input"] = True
        st.rerun()

    for resp in reversed(st.session_state.chat_histories.get(active_id, [])):
        st.markdown(f"**🧑 You:** {resp['query']}")
        st.divider()

        score = resp["critic_score"]
        if score >= 8:
            st.success(f"Score: {score}/10")
        elif score >= 6:
            st.warning(f"Score: {score}/10")
        else:
            st.error(f"Score: {score}/10")

        st.info(f"Verdict: {resp['verdict']}")

        if resp["retry_count"] > 0:
            st.warning(f"Retries: {resp['retry_count']}")

        st.markdown(resp["answer"])

        if resp.get("hallucination_flags"):
            with st.expander(f"⚠️ Hallucination flags ({len(resp['hallucination_flags'])})"):
                for flag in resp["hallucination_flags"]:
                    st.caption(f"• {flag}")

        if resp.get("vagueness_flags"):
            with st.expander(f"💤 Vagueness flags ({len(resp['vagueness_flags'])})"):
                for flag in resp["vagueness_flags"]:
                    st.caption(f"• {flag}")

        with st.expander("📚 Sources"):
            for c in resp["chunks"]:
                st.caption(f"{c.metadata['paper_title']} p.{c.metadata['page']}")

        st.divider()

# ════════════════════════════════════════════════════════════════════════════
# ⚖️ TAB 2 — COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Compare Papers")

    papers = selected_papers  # narrowed to the sidebar's checked papers

    if len(papers) < 2:
        st.info("Select at least 2 papers in the sidebar to compare.")
    else:
        col1, col2 = st.columns(2)
        a = col1.selectbox("Paper A", papers)
        b = col2.selectbox("Paper B", [p for p in papers if p != a])

        aspect_choice = st.selectbox(
            "Aspect",
            ["Full comparison"] + list(ASPECT_QUERIES.keys()) + ["Custom..."],
        )

        full_comparison = aspect_choice == "Full comparison"

        if aspect_choice == "Custom...":
            effective_aspect = st.text_input("What should be compared?")
        else:
            effective_aspect = aspect_choice

        compare_disabled = aspect_choice == "Custom..." and not effective_aspect.strip()

        if st.button("⚖️ Compare", disabled=compare_disabled):
            with st.spinner("Running comparison..."):
                result = run_comparison(
                    paper_a=a,
                    paper_b=b,
                    aspect=effective_aspect,
                    vector_store=st.session_state.vector_store,
                    full_comparison=full_comparison,
                )
                st.session_state.compare_results_by_session[active_id] = result

                # run_comparison doesn't self-log (unlike run_pipeline) — this
                # mirrors backend/api/main.py's /compare route, which builds
                # its own RunRecord the same way. Streamlit's Compare tab
                # previously logged nothing at all for this pipeline type.
                token_usage = result.get("token_usage", {})
                log_run(RunRecord(
                    pipeline_type      = "comparison",
                    session_id         = active_id,
                    query              = f"Compare {a} vs {b}: {effective_aspect}",
                    num_chunks         = len(result.get("chunks_a", [])) + len(result.get("chunks_b", [])),
                    prompt_tokens      = token_usage.get("prompt_tokens", 0),
                    completion_tokens  = token_usage.get("completion_tokens", 0),
                    total_tokens       = token_usage.get("total_tokens", 0),
                    estimated_cost_usd = result.get("estimated_cost_usd", 0.0),
                    tokens_estimated   = token_usage.get("estimated", False),
                ))

    res = st.session_state.compare_results_by_session.get(active_id)
    if res and res.get("structured"):
        s = res["structured"]

        st.success(s.get("verdict", ""))
        st.markdown("### 🧠 Synthesis")
        st.write(s.get("synthesis", ""))

        st.markdown("### 🔍 Comparison")
        differences = s.get("differences", [])
        if differences:
            rows = [
                {
                    "Dimension": d.get("aspect", ""),
                    "Paper A": d.get("paper_a", ""),
                    "Paper B": d.get("paper_b", ""),
                }
                for d in differences
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No differences returned for this comparison.")

# ════════════════════════════════════════════════════════════════════════════
# 💡 TAB 3 — FOLLOW-UP IDEAS
# ════════════════════════════════════════════════════════════════════════════
with tab_ideas:
    st.subheader("Find research gaps and follow-up opportunities")
    st.caption(
        "Select a paper from your literature. The system will surface limitations, "
        "open questions, and concrete follow-up experiment ideas — useful when "
        "planning your own research that builds on this work."
    )

    papers = selected_papers  # narrowed to the sidebar's checked papers

    if not papers:
        st.info("Select a paper in the sidebar first.")
    else:
        paper = st.selectbox("Paper", papers)

        # See the Ask tab's qa_query_input for why this is a flag-then-clear
        # rather than clearing directly after the button click.
        if st.session_state.get("_clear_ideas_focus"):
            st.session_state["ideas_focus_input"] = ""
            st.session_state["_clear_ideas_focus"] = False

        if "ideas_focus_input" not in st.session_state:
            st.session_state["ideas_focus_input"] = ""
        focus = st.text_input("Focus area (optional)", key="ideas_focus_input")

        if st.button("💡 Generate Ideas"):
            with st.spinner("Generating ideas..."):
                ideas = generate_research_ideas(
                    st.session_state.vector_store,
                    paper_filter=paper,
                    focus_area=focus,
                )
                st.session_state.idea_results_by_session[active_id] = ideas

                # generate_research_ideas doesn't self-log (unlike run_pipeline)
                # — this mirrors backend/api/main.py's /ideas route. Streamlit's
                # Ideas tab previously logged nothing at all for this type.
                token_usage = ideas.get("token_usage", {})
                log_run(RunRecord(
                    pipeline_type      = "ideas",
                    session_id         = active_id,
                    query              = f"Ideas for {paper}: {focus}",
                    paper_filter       = paper,
                    num_chunks         = len(ideas.get("chunks_used", [])),
                    prompt_tokens      = token_usage.get("prompt_tokens", 0),
                    completion_tokens  = token_usage.get("completion_tokens", 0),
                    total_tokens       = token_usage.get("total_tokens", 0),
                    estimated_cost_usd = ideas.get("estimated_cost_usd", 0.0),
                    tokens_estimated   = token_usage.get("estimated", False),
                ))

            st.session_state["_clear_ideas_focus"] = True
            st.rerun()

    ideas = st.session_state.idea_results_by_session.get(active_id)
    if ideas:
        st.info(ideas["summary"])

        st.markdown("### ⚠️ Limitations")
        for l in ideas["explicit_limitations"]:
            st.write("-", l["finding"])

        st.markdown("### ❓ Questions")
        for q in ideas["open_questions"]:
            st.write("-", q["question"])

        st.markdown("### 🚀 Ideas")
        for e in ideas["experiment_ideas"]:
            st.write(f"**{e['title']}**")
            st.write(e["description"])
            st.caption(f"{e['difficulty']} difficulty")
            st.divider()