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

# Mode picks the endpoint: RAG = fixed retrieve->generate; Agent = LLM-routed tool use.
mode = st.radio(
    "Mode",
    ["RAG", "Agent"],
    horizontal=True,
    help=(
        "RAG: always retrieves from the corpus, then answers (fast, fixed pipeline). "
        "Agent: the model chooses tools — corpus search and/or live HuggingFace model "
        "lookup (license, downloads). Use Agent for questions that mix literature with "
        "model-ecosystem facts."
    ),
)
question = st.text_input("Ask about chest X-ray AI methods, models, or datasets")

if st.button("Ask") and question:
    endpoint = "/agent" if mode == "Agent" else "/query"
    spinner = "Agent working (choosing tools)..." if mode == "Agent" else "Retrieving..."
    with st.spinner(spinner):
        try:
            resp = httpx.post(f"{API_URL}{endpoint}", json={"question": question}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Request failed: {exc}")
        else:
            # In Agent mode, surface what the agent actually did (the agentic proof).
            steps = data.get("steps", [])
            if steps:
                tools_used = ", ".join(f"`{s['tool']}`" for s in steps)
                st.info(f"🛠️ Agent called {len(steps)} tool(s): {tools_used}")
                with st.expander("Tool-call trace"):
                    for i, s in enumerate(steps, start=1):
                        st.markdown(f"{i}. **{s['tool']}** — `{s['args']}`")

            st.markdown(data["answer"])
            citations = data.get("citations", [])
            if citations:
                st.subheader("Citations")
                for c in citations:
                    st.markdown(f"**[{c['n']}] {c['title']}** — p.{c['page']}")
                    st.caption(c["snippet"])
            elif mode == "RAG":
                st.caption("No sources were cited for this answer.")
