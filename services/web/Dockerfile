FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml ./
COPY services ./services
# Web only talks to the api over HTTP — no retrieval/ML stack here.
# Installs just streamlit (+ shared core: httpx, structlog, dotenv).
RUN pip install --no-cache-dir ".[web]"

EXPOSE 8501
CMD ["streamlit", "run", "services/web/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
