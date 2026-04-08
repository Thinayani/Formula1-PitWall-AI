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

# Main are

col_chat, col_sources = st.columns([2, 1])

with col_chat:
    st.markdown("### Chat")

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])