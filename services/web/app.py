"""ChestRAG web frontend — query box wired to the API's /query endpoint."""
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

# Connectivity check — proves web -> api wiring.
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
    with st.spinner("Retrieving and generating..."):
        try:
            resp = httpx.post(f"{API_URL}/query", json={"question": question}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Query failed: {exc}")
        else:
            st.markdown(data["answer"])
            citations = data.get("citations", [])
            if citations:
                st.subheader("Citations")
                for c in citations:
                    st.markdown(f"**[{c['n']}] {c['title']}** — p.{c['page']}")
                    st.caption(c["snippet"])
            else:
                st.caption("No sources were cited for this answer.")
