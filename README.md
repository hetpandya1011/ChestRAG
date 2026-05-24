# ChestRAG

Retrieval-augmented research tool over the chest X-ray AI literature. Ask comparative
questions about chest X-ray AI methods, models, datasets, and clinical validation, and
get synthesized answers with citations to source papers.

> 🚧 Work in progress. See [PROJECT.md](PROJECT.md) for the full brief, architecture, and
> 4-weekend build plan. A complete README (demo video, eval numbers, live link) lands in Weekend 4.

## Local dev

```bash
cp .env.example .env        # defaults are fine for the empty stack
docker compose up --build   # starts Postgres+pgvector, the API, and Streamlit
```

- Streamlit UI → http://localhost:8501
- API docs → http://localhost:8000/docs
