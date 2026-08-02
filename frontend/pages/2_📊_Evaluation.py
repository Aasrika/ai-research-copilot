"""
2_📊_Evaluation.py
------------------
Streamlit multi-page app: this file auto-appears as a sidebar page.

File naming convention:  {order}_{emoji}_{Title}.py
Streamlit picks it up automatically from the pages/ folder.

WHAT THIS DASHBOARD SHOWS:
  Row 1: Headline KPIs — pass rate, avg score, hallucination rate, avg latency
  Row 2: Score trend + Verdict distribution
  Row 3: Latency breakdown + Section distribution
  Row 4: Failure analysis — weak runs + top hallucination flags
  Row 5: Raw run log (last 50)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
from core import session_manager
from evaluation.metrics import compute_all_metrics
from evaluation.logger  import load_runs, update_feedback


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


st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊", layout="wide")
st.title("📊 Evaluation Dashboard")
st.caption("Live metrics across all pipeline runs — refreshes on each page load.")
render_session_banner()

# ── Session filter ───────────────────────────────────────────────────────────
sessions = session_manager.list_sessions()

if not sessions:
    st.info("No sessions exist yet — create one on the main page first.")
    st.stop()

session_names = {s["id"]: s["name"] for s in sessions}
filter_options = ["All sessions"] + list(session_names.keys())

# Default to the active session if one is set; otherwise "All sessions".
# This only determines the widget's INITIAL value — once rendered, the
# user's own selection persists across reruns like any Streamlit widget.
active_id = st.session_state.get("active_session_id")
default_index = filter_options.index(active_id) if active_id in filter_options else 0

selected = st.selectbox(
    "Filter by session",
    filter_options,
    index=default_index,
    format_func=lambda x: "All sessions" if x == "All sessions" else session_names.get(x, x),
)
session_filter = None if selected == "All sessions" else selected

# ── Refresh button ─────────────────────────────────────────────────────────
if st.button("🔄 Refresh"):
    st.rerun()

all_metrics = compute_all_metrics(session_id=session_filter)
m           = all_metrics["qa"]       # focus on Q&A metrics by default
combined    = all_metrics["combined"]

if combined.get("total_runs", 0) == 0:
    if session_filter is None:
        st.info("No runs logged yet. Ask some questions in the main app first!")
    else:
        st.info(f"No runs logged yet for session **{session_names.get(session_filter, session_filter)}**.")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# ROW 1 — KPI Tiles
# ════════════════════════════════════════════════════════════════════════════
st.subheader("📈 Key Performance Indicators")
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total Runs",          combined["total_runs"])
k2.metric("Pass Rate",           f"{m.get('pass_rate', 0)}%",
          help="% of Q&A runs with verdict=PASS")
k3.metric("Avg Critic Score",    f"{m.get('avg_critic_score', 0)}/10")
k4.metric("Hallucination Rate",  f"{m.get('hallucination_rate', 0)}%",
          help="% of runs with ≥1 unverified claim flagged",
          delta=f"-{m.get('hallucination_rate',0)}%" if m.get('hallucination_rate',0) < 20 else None,
          delta_color="inverse")
k5.metric("Avg Latency",         f"{m.get('latency', {}).get('total', 0)}s")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# ROW 2 — Score Trend + Verdict Breakdown
# ════════════════════════════════════════════════════════════════════════════
col_trend, col_verdict = st.columns([3, 2])

with col_trend:
    st.subheader("📉 Critic Score Over Time")
    trend_data = m.get("score_trend", [])
    if trend_data:
        df_trend = pd.DataFrame(trend_data)
        # Add a rolling average line if enough data
        if len(df_trend) >= 5:
            df_trend["rolling_avg"] = df_trend["score"].rolling(3, min_periods=1).mean()
        st.line_chart(df_trend.set_index("timestamp")[
            ["score"] + (["rolling_avg"] if "rolling_avg" in df_trend.columns else [])
        ])
        st.caption("Rolling average (window=3) shows whether quality is improving.")
    else:
        st.info("Not enough data yet — run at least 5 queries.")

with col_verdict:
    st.subheader("⚖️ Verdict Distribution")
    verdicts = m.get("verdict_counts", {})
    if verdicts:
        df_v = pd.DataFrame(
            {"Verdict": list(verdicts.keys()), "Count": list(verdicts.values())}
        ).set_index("Verdict")
        st.bar_chart(df_v)
        # Interpretation note
        if verdicts.get("PASS", 0) / sum(verdicts.values()) < 0.7:
            st.warning("⚠️ Pass rate < 70% — consider increasing k or tuning chunk size.")
        else:
            st.success("✅ Pass rate looks healthy.")
    else:
        st.info("No verdict data yet.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# ROW 3 — Latency + Section Distribution
# ════════════════════════════════════════════════════════════════════════════
col_lat, col_sec = st.columns(2)

with col_lat:
    st.subheader("⏱️ Latency Breakdown")
    lat = m.get("latency", {})
    if lat.get("total", 0) > 0:
        df_lat = pd.DataFrame({
            "Stage":   ["Retrieval", "Generation", "Critic"],
            "Seconds": [lat["retrieval"], lat["generation"], lat["critic"]]
        }).set_index("Stage")
        st.bar_chart(df_lat)
        st.caption(f"Total avg: **{lat['total']}s** end-to-end")
        if lat["total"] > 15:
            st.warning("Latency > 15s — consider caching frequent queries or using gpt-4o-mini for critic too.")
    else:
        st.info("No latency data yet.")

with col_sec:
    st.subheader("📑 Retrieved Sections Distribution")
    sec_dist = m.get("section_distribution", {})
    if sec_dist:
        df_sec = pd.DataFrame({
            "Section": list(sec_dist.keys()),
            "Count":   list(sec_dist.values())
        }).set_index("Section")
        st.bar_chart(df_sec)
        st.caption("Shows which paper sections are being retrieved most. "
                   "Heavy 'body' bias = section detection needs tuning.")
    else:
        st.info("No section data yet.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# ROW 4 — Failure Analysis
# ════════════════════════════════════════════════════════════════════════════
col_weak, col_flags = st.columns(2)

with col_weak:
    st.subheader("🔴 Weakest Runs (score < 6)")
    weak = m.get("weak_runs", [])
    if weak:
        df_weak = pd.DataFrame(weak)
        st.dataframe(df_weak, use_container_width=True, hide_index=True)
        st.caption("These are your failure cases — good inputs for prompt tuning.")
    else:
        st.success("✅ No runs with score < 6!")

with col_flags:
    st.subheader("⚠️ Most Common Hallucination Flags")
    flags = m.get("top_flags", [])
    if flags:
        df_flags = pd.DataFrame(flags, columns=["Flag", "Count"]).set_index("Flag")
        st.dataframe(df_flags, use_container_width=True)
        st.caption("Recurring flags suggest systematic gaps in your retrieval or prompts.")
    else:
        st.success("✅ No recurring hallucination patterns.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# ROW 5 — All Pipeline Types Summary
# ════════════════════════════════════════════════════════════════════════════
st.subheader("📋 Metrics By Pipeline Type")
summary_rows = []
for ptype in ["qa", "comparison", "ideas", "critique"]:
    pm = all_metrics[ptype]
    if pm.get("total_runs", 0) > 0:
        summary_rows.append({
            "Pipeline":          ptype.title(),
            "Runs":              pm["total_runs"],
            "Avg Score":         pm.get("avg_critic_score", "N/A"),
            "Pass Rate":         f"{pm.get('pass_rate', 0)}%",
            "Hallucination Rate":f"{pm.get('hallucination_rate', 0)}%",
            "Avg Latency (s)":   pm.get("latency", {}).get("total", "N/A"),
        })

if summary_rows:
    st.dataframe(pd.DataFrame(summary_rows).set_index("Pipeline"),
                 use_container_width=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# ROW 6 — Raw Run Log
# ════════════════════════════════════════════════════════════════════════════
with st.expander("🗂️ Raw Run Log (last 50)"):
    runs = load_runs(session_id=session_filter)[-50:]
    if runs:
        df_runs = pd.DataFrame(runs)[[
            "run_id","timestamp","pipeline_type","query",
            "critic_score","verdict","retry_count",
            "latency_total","num_chunks"
        ]]
        df_runs["query"]     = df_runs["query"].str[:60] + "..."
        df_runs["timestamp"] = df_runs["timestamp"].str[:16]
        st.dataframe(df_runs, use_container_width=True, hide_index=True)
    else:
        st.info("No runs yet.")