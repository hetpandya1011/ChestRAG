# ChestRAG — Project Brief

## What This Project Is

A retrieval-augmented research tool over the chest X-ray AI literature. Users (researchers, ML engineers, clinical AI practitioners) can ask complex, comparative questions about chest X-ray AI methods, foundation models, datasets, and clinical validation studies, and receive synthesized answers with citations to source papers.

The corpus is intentionally narrow — chest X-ray AI specifically, not all medical imaging — to enable depth, defensible completeness, and demo-able domain-specific queries (e.g., "Compare CheXagent and CheXzero on CheXpert").

## Why It's Built This Way

This is a portfolio project to demonstrate production-grade AI engineering for a job search. Every architectural decision is chosen to cover multiple skills from target job descriptions simultaneously. The point is depth-per-decision, not feature accumulation.

Decisions should optimize for:

1. **Demonstrating engineering depth**, not just wiring frameworks together
2. **Shipping a working, deployed system** on AWS — not localhost, not Vercel, not Streamlit Cloud
3. **Defensible design choices** I can explain in interviews
4. **Evaluation as a first-class concern** — not an afterthought

When in doubt, prefer the simpler, more honest design over the buzzier one. A well-engineered RAG beats a half-built agent.

## About My Code Level

I'm not a fluent Python developer. I understand AI/ML architecture well but rely on AI assistance for implementation. I need to understand what the code does at a conceptual level so I can defend the architecture and design decisions in interviews — but I don't need to be quizzed on syntax or be able to write it from scratch. Explain decisions and trade-offs as you go; skip exhaustive line-by-line walkthroughs unless I ask.

## Target Job Description Coverage

This project is designed to honestly check the following JD requirements across target Toronto AI engineering roles:

- Python, FastAPI, async patterns
- RAG / retrieval / vector databases (pgvector)
- Hybrid retrieval (dense + BM25) with re-ranking
- LLM API integration (Anthropic, OpenAI, Google)
- Multi-model selection with documented trade-offs
- Agentic workflows / tool use / function calling
- Evaluation methodology (LLM-as-judge, hallucination detection, recall@k)
- Docker / docker-compose
- AWS deployment (EC2, RDS, S3, IAM basics)
- CI/CD via GitHub Actions
- Testing (pytest)
- SQL / Postgres
- Production logging and cost tracking
- Modular service design
- Healthcare/life-sciences domain context

## Scope

**In scope:**
- Curated corpus of ~300–500 chest X-ray AI papers (arXiv, PubMed Central, MICCAI/MIDL proceedings)
- FastAPI service for retrieval and generation
- Streamlit frontend calling the FastAPI service
- pgvector in Postgres for vector storage
- Hybrid retrieval (dense + BM25) with cross-encoder re-ranking
- Citation enforcement (every claim ties to a retrieved chunk)
- One agent tool with function calling: HuggingFace model card lookup (returns license, benchmarks, downloads, code snippets for a given model). Design rationale: the agent tool must do something the corpus *can't* — return live model-ecosystem data — not something the corpus already does (retrieve papers). This is what keeps the agent path architecturally distinct rather than a slower duplicate of RAG.
- Evaluation harness with hand-built question set and LLM-as-judge
- Multi-model benchmarking (Claude vs GPT-4 vs Gemini) on eval set
- Docker + docker-compose for local orchestration
- AWS deployment (EC2 + RDS + S3)
- GitHub Actions CI/CD running tests on every push
- Cost and latency logging per query

**Out of scope:**
- Other imaging modalities
- Open web search (security/hallucination risk)
- Multi-modal (image) retrieval — text only
- User accounts, auth, persistence
- GROBID parsing (pymupdf is sufficient)
- Terraform / IaC (deploy manually; time vs. value not justified)
- Streaming responses (polish, not signal)
- Next.js frontend (Streamlit ships faster)
- Fine-tuning experiments
- Kubernetes / production-scale concurrency

## Architecture

```
┌─────────────────┐
│  Streamlit UI   │  ← user-facing
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────────────────────────────┐
│         FastAPI Service                  │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Retrieve │→ │ Re-rank  │→ │Generate│ │
│  └──────────┘  └──────────┘  └────────┘ │
│        │                          │      │
│        ▼                          ▼      │
│  ┌──────────┐              ┌──────────┐ │
│  │ pgvector │              │ HF tool  │ │  ← optional agent path
│  │ (RDS)    │              │ lookup   │ │  ← live model-ecosystem info
│  └──────────┘              └──────────┘ │     (license/benchmarks/downloads),
└─────────────────────────────────────────┘     NOT papers — that's the corpus's job
         │
         ▼
┌─────────────────┐
│  Ingest Pipeline│  ← offline / batch
│  OpenAlex → S3  │
│  PDF → chunks   │
│  → embeddings   │
└─────────────────┘

Deployment: AWS (EC2 host, RDS Postgres, S3 for raw PDFs)
CI/CD: GitHub Actions (pytest + lint on push, manual deploy)
```

## Tech Choices

- **Language:** Python 3.11+
- **Service framework:** FastAPI with async endpoints
- **Retrieval framework:** LlamaIndex (lean, RAG-focused) — drop to raw API where it gets in the way
- **LLM (generation):** Anthropic Claude via API
- **LLMs (benchmarking):** GPT-4 and Gemini for multi-model eval comparison
- **Embeddings:** OpenAI `text-embedding-3-small` to start; benchmark Voyage `voyage-3` in eval
- **Vector store:** pgvector in Postgres (local Docker for dev, RDS for prod)
- **PDF parsing:** pymupdf
- **Frontend:** Streamlit calling FastAPI
- **Containers:** Docker + docker-compose for local stack
- **Deployment:** AWS EC2 (t3.small or similar) + RDS Postgres + S3
- **CI/CD:** GitHub Actions
- **Testing:** pytest
- **Logging:** structlog with JSON output

## Project Structure

```
chestrag/
├── README.md
├── PROJECT.md
├── pyproject.toml
├── .env.example
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml              # pytest + lint on push
├── services/
│   ├── api/                    # FastAPI service
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── routes.py
│   │   ├── retrieval.py
│   │   ├── rerank.py
│   │   ├── generate.py
│   │   ├── tools.py            # the one agent tool
│   │   └── logging_config.py
│   └── web/                    # Streamlit frontend
│       ├── Dockerfile
│       └── app.py
├── ingest/                     # offline batch pipeline
│   ├── discover.py             # OpenAlex / PubMed search
│   ├── fetch.py                # download PDFs to S3 / local
│   ├── parse.py                # pymupdf extraction
│   ├── chunk.py                # section-aware chunking
│   └── embed_and_store.py      # embed → pgvector
├── eval/
│   ├── questions.jsonl         # hand-built eval set (~50 questions)
│   ├── metrics.py              # recall@k, MRR
│   ├── llm_judge.py            # LLM-as-judge for faithfulness
│   ├── run_eval.py             # full eval pipeline
│   └── results/                # gitignored; eval run outputs
├── tests/
│   ├── test_retrieval.py
│   ├── test_chunking.py
│   └── test_api.py
├── infra/
│   └── aws_setup.md            # manual deploy runbook
└── data/                       # gitignored
```

## Build Plan — 4 Weekends

Each weekend is one architectural decision that covers multiple JD requirements.

### Weekend 1 — End-to-end pipeline works, locally, dockerized

**One architectural decision:** Build it as FastAPI + Streamlit + pgvector in Docker from day one (not Chroma, not monolithic Streamlit).

**Skills covered:** Python, FastAPI, async, Docker, docker-compose, pgvector, SQL, modular services, API design.

**Deliverable:**
- 20 seed chest X-ray papers (CheXNet, CheXpert, CheXagent, CheXzero, RadFM, BiomedCLIP, etc.) ingested
- FastAPI service with `/query` endpoint
- Streamlit frontend with a query box
- pgvector running in Docker
- `docker-compose up` brings everything up
- Basic vector search → Claude generation → citations
- pytest tests for at least retrieval and chunking

**Stop when:** A query through the Streamlit UI returns an answer with citations, locally, via docker-compose.

### Weekend 2 — Real corpus, real retrieval, eval harness

**One architectural decision:** Treat evaluation as core infrastructure, not a post-hoc check.

**Skills covered:** Evaluation methodology, LLM-as-judge, hallucination detection, hybrid retrieval, re-ranking, statistical thinking.

**Deliverable:**
- Corpus expanded to 300–500 papers via OpenAlex search
- Section-aware chunking (methods/results/discussion treated differently)
- Hybrid retrieval: dense (pgvector) + BM25, weighted combination
- Cross-encoder re-ranker on top-N candidates
- Hand-built 50-question eval set with known correct source papers
- Eval metrics: retrieval recall@5, MRR
- LLM-as-judge: faithfulness score (does the answer cite real chunks and only claim what they support?)
- Hallucination detection: flag answers with claims unsupported by retrieved chunks
- `python eval/run_eval.py` runs full eval and outputs JSON results

**Stop when:** Eval numbers are recorded for at least one configuration and can be re-run.

### Weekend 3 — Multi-model benchmark, one agent tool, deploy to AWS

**One architectural decision:** Deploy to real cloud (EC2 + RDS), not a PaaS, and add the minimum agent surface area.

**Skills covered:** AWS (EC2, RDS, S3, IAM basics), cloud deployment, multi-model evaluation, agentic workflows, function calling, tool use, production architecture.

**Deliverable:**
- Run eval set against Claude, GPT-4, and Gemini; record cost / latency / quality trade-offs in a table
- Document the model selection decision in README
- Add ONE agent tool with function calling: HuggingFace model card lookup (license, current benchmarks, downloads, code snippets, maintenance status for a named model). Chosen because it returns operational/ecosystem data the papers don't contain — the agent must do something the corpus *can't*, not duplicate the corpus's own retrieval path.
- Router logic: literature questions → pure RAG path; "what does this model look like today" questions → agent path
- Deploy: EC2 t3.small running Docker, RDS Postgres for pgvector, S3 for raw PDFs
- Public URL accessible (with Nginx + Let's Encrypt for HTTPS)
- Document the AWS architecture and manual deploy steps in `infra/aws_setup.md`

**Stop when:** A stranger can hit the public URL and get an answer, and your eval set has run against three models with results documented.

### Weekend 4 — CI/CD, observability, polish, ship

**One architectural decision:** Make the project look like production engineering, not a tutorial.

**Skills covered:** CI/CD, testing in CI, observability, cost tracking, documentation, communication.

**Deliverable:**
- GitHub Actions workflow: runs pytest + ruff on every push and PR
- Test coverage at least on retrieval, chunking, API endpoints, and one full integration test
- Structured JSON logging (structlog) on every API call: query, retrieved chunks, model used, tokens in/out, cost, latency
- Cost dashboard: simple SQL view or script that summarizes daily token costs by model
- README rewritten as a sales document (see below)
- 60–90 second demo video showing 3 strongest queries
- LinkedIn post drafted (do not auto-post; review first)

**Stop when:** A hiring manager scanning the README for 60 seconds gets the full picture — what it does, the architecture, the eval numbers, the design decisions, the tech stack.

## What "Done" Looks Like

- Live AWS-hosted URL where a stranger can ask questions and get useful answers
- README explains design decisions, shows architecture diagram, lists tech stack (greppable for recruiters), and reports real eval numbers
- 60–90 second demo video at the top of the README
- All four weekends' work merged to `main`, with clean commit history
- CI passing on `main`
- I can answer "why did you choose X over Y?" for every major component
- A LinkedIn post written but not necessarily posted yet

## README Structure (for Weekend 4)

```
# ChestRAG

[Demo video] [Live demo link] [CI badge]

One-paragraph: what it does, who it's for, why I built it.

## Architecture
[diagram]
Brief explanation of each component.

## Evaluation Results
| Metric | Score | Configuration |
|---|---|---|
| Recall@5 | 0.XX | hybrid retrieval, 500-token chunks |
| Faithfulness | 0.XX | LLM-as-judge on 50 questions |
| p50 latency | X.Xs | end-to-end |
| Cost per query | $0.0X | Claude Sonnet path |

## Multi-Model Comparison
[Claude vs GPT-4 vs Gemini table with cost / quality / latency]

## Key Engineering Decisions
[5–8 bullets — chunking, retrieval, re-ranking, citation enforcement, when-to-agent routing, etc.]

## Tech Stack
[Greppable bullet list: Python, FastAPI, LlamaIndex, pgvector, Postgres, Claude, GPT-4, Gemini, Docker, docker-compose, AWS (EC2, RDS, S3), GitHub Actions, pytest, structlog]

## What I'd Do Differently at Scale
[Shows maturity — IaC, queue-based ingestion, model caching layer, etc.]

## Local Setup
[Standard `docker-compose up` instructions]
```

## Working Norms With Claude Code

- **Test small before scaling.** 20 papers before 500.
- **Commit early and often.** Small commits with clear messages I can roll back to.
- **Prefer simple, explicit code over clever abstractions.** Must be readable when I revisit it.
- **Surface trade-offs.** When making a decision, briefly explain alternatives and why you chose one.
- **Don't hide failure modes.** Comment on hacky bits so I know.
- **Always use a virtual environment.**
- **Never commit secrets.** Use `.env` and `.env.example`.
- **Don't over-engineer.** If it's not in the scope above, ask before adding it.

## Open Questions (Discuss in First Session)

1. AWS region: us-east-1 (cheapest, biggest) vs. ca-central-1 (closer to Toronto, slightly pricier).
2. RDS instance size — t4g.micro to keep costs down, or t4g.small for headroom?
3. Eval question format — start with a proposed JSONL schema in Weekend 2.

## Starting Point

Read this whole file. Ask any clarifying questions you have. Then propose the initial Weekend 1 scaffolding — folders, `pyproject.toml`, `.env.example`, `docker-compose.yml` skeleton (Postgres + pgvector + API + web), `.gitignore`. No business logic yet. First commit is just the structure.
