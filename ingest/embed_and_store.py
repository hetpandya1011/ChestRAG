"""Embed the chunks in data/chunks.jsonl and store them in pgvector.

Reads each chunk (text + citation metadata), embeds it with OpenAI
text-embedding-3-small, and writes it into a pgvector table via LlamaIndex's
PGVectorStore. The table is created with hybrid search enabled (dense vector +
Postgres full-text), which retrieval uses later.

Idempotent: drops and recreates the table, so re-running after a chunking change
starts clean.

Prereqs:
  - db container running:  docker compose up -d db
  - OPENAI_API_KEY set in .env

Usage (from repo root, inside the venv):
    python ingest/embed_and_store.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = REPO_ROOT / "data" / "chunks.jsonl"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
TABLE_NAME = "chunks"          # PGVectorStore creates the physical table "data_chunks"

# Host from env: "localhost" for the local docker-compose DB, or the RDS endpoint when
# loading the cloud database from this machine. Falls back to localhost for local use.
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")


def load_nodes() -> list[TextNode]:
    nodes = []
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            # Postgres text columns reject NUL (0x00) bytes, which PDF extraction
            # occasionally emits. Strip them before embedding/storing.
            text = c["text"].replace("\x00", "")
            nodes.append(
                TextNode(
                    id_=c["chunk_id"],
                    text=text,
                    metadata={
                        "paper_id": c["paper_id"],
                        "title": c["title"],
                        "page": c["page"],
                        "section": c.get("section"),
                    },
                )
            )
    return nodes


def reset_table(user: str, password: str, database: str, port: int) -> None:
    """Drop the existing vector table so the build is idempotent."""
    conn = psycopg2.connect(host=DB_HOST, port=port, user=user, password=password, dbname=database)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS data_{TABLE_NAME}")
    conn.close()


def main() -> int:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set in .env. Add it and re-run.")
        return 1
    if not CHUNKS_FILE.exists():
        print(f"No chunks at {CHUNKS_FILE}. Run `python ingest/chunk.py` first.")
        return 1

    user = os.getenv("POSTGRES_USER", "chestrag")
    password = os.getenv("POSTGRES_PASSWORD", "chestrag")
    database = os.getenv("POSTGRES_DB", "chestrag")
    port = int(os.getenv("POSTGRES_PORT", "5432"))

    nodes = load_nodes()
    print(f"Embedding {len(nodes)} chunks with {EMBED_MODEL} -> pgvector\n")

    try:
        print(f"Resetting table data_{TABLE_NAME} ...")
        reset_table(user, password, database, port)
    except psycopg2.OperationalError as exc:
        print(
            f"Could not connect to Postgres at {DB_HOST}:{port}. "
            f"Is the db container up?  Try: docker compose up -d db\n  ({exc})"
        )
        return 1

    vector_store = PGVectorStore.from_params(
        host=DB_HOST,
        port=str(port),
        database=database,
        user=user,
        password=password,
        table_name=TABLE_NAME,
        embed_dim=EMBED_DIM,
        hybrid_search=True,
        text_search_config="english",
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(model=EMBED_MODEL)

    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    print(f"\nStored {len(nodes)} chunks in pgvector table 'data_{TABLE_NAME}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
