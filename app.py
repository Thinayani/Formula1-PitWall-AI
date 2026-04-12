import streamlit as st
import requests
import json
import time

# Page config

st.set_page_config(
    page_title = "PitWall AI",
    page_icon  = " ",
    layout     = "wide",
)

# Constants

API_BASE        = "http://localhost:8000"
EXAMPLE_QUERIES = [
    "Who won the 2021 Abu Dhabi Grand Prix and how did the strategy unfold?",
    "Which driver took the most pole positions in the 2020 season?",
    "When was the last time a safety car directly decided the championship?",
    "Compare tyre strategies at Monaco between 2018 and 2019.",
    "What is the record for fastest pit stop in the data?",
    "Which team dominated the 2014 season and by how much?",
]

# Session state

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# Sidebar

with st.sidebar:
    st.markdown("## PitWall AI")
    st.markdown("*F1 Strategy Intelligence*")
    st.divider()

    st.markdown("### Filters")
    season_filter = st.selectbox(
        "Season (optional)",
        options=["All seasons"] + [str(y) for y in range(2023, 2009, -1)],
    )
    data_type_filter = st.selectbox(
        "Data type",
        options=["All", "Race results", "Telemetry"],
    )

    season_val    = int(season_filter)    if season_filter != "All seasons"   else None
    data_type_val = None
    if data_type_filter == "Race results":
        data_type_val = "race_result"
    elif data_type_filter == "Telemetry":
        data_type_val = "telemetry"

    st.divider()

    st.markdown("### Try asking")
    for q in EXAMPLE_QUERIES:
        if st.button(q, key=f"ex_{q[:20]}", use_container_width=True):
            st.session_state.pending_query = q

    st.divider()

    # Backend health
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=2)
        if resp.ok:
            st.success("Backend: online")
        else:
            st.error("Backend: error")
    except Exception:
        st.error("Backend: offline\nStart with: uvicorn main:app --reload")

    # Index stats
    try:
        stats = requests.get(f"{API_BASE}/stats", timeout=2).json()
        if "vectors_indexed" in stats:
            st.metric("Vectors indexed", f"{stats['vectors_indexed']:,}")
    except Exception:
        pass

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()


# Main area

col_chat, col_sources = st.columns([2, 1])

with col_chat:
    st.markdown("### Chat")

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

with col_sources:
    st.markdown("### Sources")

    if st.session_state.last_sources:
        st.caption(f"{len(st.session_state.last_sources)} chunks retrieved")
        for src in st.session_state.last_sources:
            race  = src.get("race_name", "Unknown")
            year  = src.get("season", "")
            dtype = src.get("data_type", "")
            score = src.get("score", 0)
            preview = src.get("text_preview", "")

            dtype_label = "Race result" if dtype == "race_result" else "Telemetry"
            score_pct   = min(int(score * 10 + 70), 99) if score else 0