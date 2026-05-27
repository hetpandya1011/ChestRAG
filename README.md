# ChestRAG

Retrieval-augmented research tool over the chest X-ray AI literature. Ask comparative
questions about chest X-ray AI methods, models, datasets, and clinical validation, and
get synthesized answers with inline citations to source papers.

**Corpus:** 241 papers · **Retrieval:** hybrid pgvector (dense + Postgres FTS) · **Generation:** GPT-4o

---

## Demo

> Live URL and demo video coming Weekend 4 (AWS deploy).

Example query: *"Compare CheXzero and BioViL-T on zero-shot classification"*

The system retrieves the five most relevant chunks across 241 papers, generates a
synthesized answer, and returns numbered citations with page-level provenance.

---

## Architecture

Two paths share one corpus and one citation layer:

```
                         User question
                               │
              ┌────────────────┴────────────────┐
              ▼                                  ▼
     /query  (fixed RAG)              /agent  (agentic RAG)
              │                                  │
              ▼                          GPT-4o + tools — the LLM
 Hybrid retrieval (dense+FTS)           decides which to call:
   241 papers, 7 757 chunks              ├─ search_corpus  → hybrid retrieval
              │                          └─ hf_model_lookup → live HF metadata
              ▼                                  │   (license, downloads, ...)
   GPT-4o generation                            ▼  loop until it answers
   (cite only from sources)              answer + [n] citations + tool trace
              │                                  │
              └────────────────┬────────────────┘
                               ▼
                  answer + [n] inline citations  →  Streamlit UI
```

- **`/query`** — fixed pipeline: always retrieve, then generate. Fast, predictable.
- **`/agent`** — the LLM is the router: it picks `search_corpus`, `hf_model_lookup`, or
  both per question (a tool-use loop). See [Agentic RAG](#agentic-rag) below.

**Stack:** FastAPI · Streamlit · LlamaIndex · pgvector · OpenAI embeddings (`text-embedding-3-small`) · OpenAI function calling · Docker · AWS (EC2 + RDS)

---

## Agentic RAG

The `/agent` endpoint turns the fixed RAG pipeline into a tool-use loop. GPT-4o is given
two tools and **decides at runtime** which to call — there is no hardcoded routing:

| Tool | Answers | Source |
|---|---|---|
| `search_corpus` | "What does the literature say?" — methods, datasets, results | the 241-paper corpus (hybrid retrieval) |
| `hf_model_lookup` | "What does the model look like *today*?" — license, downloads, maintenance | live HuggingFace Hub API |

**Routing is the tool descriptions, not `if/else`.** The model reads each tool's
description and matches it to the question. A literature question calls only
`search_corpus`; a "what's the license?" question calls only `hf_model_lookup`; a mixed
question calls both and synthesizes. The loop runs tools, feeds results back, and repeats
until the model answers (capped at 5 steps).

**Design rationale — the agent must do what the corpus *can't*.** The tool was originally
going to be arXiv search, but that's redundant: the corpus is already arXiv papers, so the
agent would just be a slower duplicate of the RAG path. HuggingFace lookup is genuinely
complementary — it returns operational/ecosystem facts (license, current download counts,
maintenance status) that papers don't contain and that drift after publication.

**Example** (`/agent`, *"How many downloads does BiomedCLIP have, and what does it do?"*):
the agent calls `hf_model_lookup` → 975 k downloads, MIT license (live), then
`search_corpus` → cited passages on its contrastive-learning architecture. One answer,
two sources, each labelled. When HuggingFace returns no declared license, the agent
reports "license not specified" rather than inventing one.

---

## Eval results

### Retrieval (50 questions, 241-paper corpus)

| Config | Hit@5 | Recall@5 | MRR |
|---|---|---|---|
| Hybrid (dense + FTS) | **0.975** | **0.975** | **0.963** |
| Hybrid + cross-encoder rerank | 0.925 | 0.925 | 0.908 |

Plain hybrid beat reranking — the MS-MARCO cross-encoder mis-scores dense biomedical text,
so reranking is off by default (`use_rerank=False`).

### Generation faithfulness (LLM-as-judge, gpt-4o judge, 15 questions)

| Model | Faithfulness | Latency | Cost/query |
|---|---|---|---|
| gpt-4o | 1.0 | 2.6 s | $0.006 |
| claude-sonnet-4-6 | 1.0 | 6.3 s | $0.011 |

Both models score perfect faithfulness. GPT-4o is ~2.4× faster and ~1.8× cheaper, so it
is the default generation model. Claude is available via `GENERATION_MODEL=claude-sonnet-4-6`
for Anthropic-ecosystem deployments.

*Note: faithfulness judged by gpt-4o; slight self-bias toward the OpenAI row. Cost and latency are the objective comparisons.*

### Adversarial / refusal (6 unanswerable questions)

All 6 unanswerable questions correctly refused (faithfulness 1.0 — the judge rewards
on-topic refusals as grounded responses).

---

## Local dev

```bash
cp .env.example .env          # fill in OPENAI_API_KEY
docker compose up --build     # Postgres+pgvector, API, Streamlit
```

- Streamlit UI → http://localhost:8501
- API docs → http://localhost:8000/docs

### Ingest (first run only)

```bash
source .venv/bin/activate
python -m ingest.fetch        # download 241 PDFs → data/raw/
python -m ingest.parse        # extract text → data/parsed/
python -m ingest.chunk        # section-aware chunking → data/chunks.jsonl
python -m ingest.embed_and_store  # embed + load into pgvector
```

### Run eval

```bash
python -m eval.run_eval                  # retrieval metrics (50 Q)
python -m eval.run_eval --faithfulness 15  # + faithfulness judge
python -m eval.benchmark -n 15           # multi-model cost/latency/quality
```

---

## Project structure

```
services/
  api/        FastAPI app (retrieval, generation, reranking, agent loop)
    tools/    agent tools (HuggingFace model-card lookup)
  web/        Streamlit UI (RAG / Agent toggle + tool-call trace)
ingest/       PDF fetch → parse → chunk → embed pipeline
eval/         Questions, metrics, LLM judge, benchmark
infra/        AWS deploy runbook (coming Weekend 4)
```

---

## Design decisions

- **Hybrid retrieval over pure vector search** — Postgres FTS catches exact model names
  (CheXzero, BioViL-T) that dense vectors smear across synonyms. +5 pp recall@5 in practice.
- **Reranking disabled** — measured and rejected: cross-encoder recall@5 dropped from 0.975
  to 0.925 on this biomedical corpus. Toggle via `use_rerank=True` in `retrieve()`.
- **Section-aware chunking** — heuristic heading detection drops references/acknowledgments,
  keeps abstract/methods/results. Reduces noise in retrieved chunks.
- **GPT-4o for generation** — identical faithfulness to Claude Sonnet at 1.8× lower cost
  and 2.4× lower latency (benchmark on 15 questions, n=15).
- **Agent tool = HuggingFace lookup, not arXiv search** — the agent tool must do something
  the corpus *can't* (return live model-ecosystem data), not duplicate the corpus's own
  retrieval. Routing is delegated to the LLM via tool descriptions, not hardcoded.
