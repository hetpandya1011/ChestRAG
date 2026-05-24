"""ChestRAG web frontend — Weekend 1 empty-stack skeleton.

Shows a (not-yet-wired) query box and pings the API's /health endpoint so we
can visually confirm the web container can reach the api container over the
docker-compose network.
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

# In the compose network the api is reachable at http://api:8000; default to
# localhost for running outside Docker.
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ChestRAG", page_icon="🩻")
st.title("🩻 ChestRAG")
st.caption("Retrieval-augmented research over the chest X-ray AI literature")

# Connectivity check — this is the visible proof that web -> api wiring works.
with st.sidebar:
    st.subheader("Service status")
    try:
        resp = httpx.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200 and resp.json().get("status") == "ok":
            st.success("API: ok")
        else:
            st.error(f"API returned {resp.status_code}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"API unreachable: {exc}")

question = st.text_input("Ask about chest X-ray AI methods, models, or datasets")
if st.button("Ask") and question:
    st.info("Answer generation isn't built yet — this is the Weekend 1 scaffold.")
