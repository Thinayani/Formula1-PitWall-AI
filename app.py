# Run: streamlit run app.py
# Requires: main.py running in a separate terminal

import streamlit as st
import requests
import json
import time

# Page config

st.set_page_config(
    page_title = "PitWall AI",
    page_icon  = "",
    layout     = "wide",
)

# Styling
st.markdown("""
<style>
    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0f0f0f; }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* Source cards */
    .source-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
        color: #aaa;
    }
    .source-card .race { font-weight: 600; color: #e0e0e0; }
    .source-card .score { color: #e8290b; font-size: 11px; }

    /* Chat bubbles */
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

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

    # Handle example query injection
    prefill = st.session_state.pop("pending_query", None)

    user_input = st.chat_input("Ask anything about F1 strategy...") or prefill

    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Stream assistant response
        with st.chat_message("assistant"):
            placeholder   = st.empty()
            full_response = ""
            sources_raw   = []

            try:
                payload = {
                    "question":  user_input,
                    "season":    season_val,
                    "data_type": data_type_val,
                    "top_n":     5,
                    "stream":    True,
                }
                with requests.post(
                    f"{API_BASE}/query/stream",
                    json=payload,
                    stream=True,
                    timeout=60,
                ) as r:
                    r.raise_for_status()
                    buffer = ""
                    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                        buffer += chunk

                        # Check for sources footer sentinel
                        if "__SOURCES__" in buffer:
                            answer_part, sources_part = buffer.split("__SOURCES__", 1)
                            full_response = answer_part
                            try:
                                sources_raw = json.loads(sources_part)
                            except json.JSONDecodeError:
                                pass
                            placeholder.markdown(full_response)
                            break
                        else:
                            placeholder.markdown(buffer + "▌")

                    if not full_response:
                        full_response = buffer

            except requests.exceptions.ConnectionError:
                full_response = (
                    "Cannot reach the PitWall API. "
                    "Make sure `uvicorn main:app --reload` is running."
                )
                placeholder.error(full_response)

            except Exception as e:
                full_response = f"Error: {e}"
                placeholder.error(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.last_sources = sources_raw

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

            st.markdown(f"""
<div class="source-card">
  <div class="race">{race} {year}</div>
  <div>{dtype_label}</div>
  <div class="score">Relevance: {score_pct}%</div>
  <div style="margin-top:6px;font-size:12px;color:#888">{preview}</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.caption("Sources will appear here after your first query.")

        st.markdown("""
<div style="margin-top: 2rem; color: #555; font-size: 13px;">
PitWall AI retrieves relevant race data, telemetry, and strategy information
from an indexed database of F1 seasons 2010–2023, then synthesizes an answer
using Claude.
</div>
""", unsafe_allow_html=True)