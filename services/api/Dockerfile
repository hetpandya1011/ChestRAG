# Python 3.12 (broad wheel support) on a slim base to keep the image small.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy only what's needed to install, then install. (Source is also live-mounted
# by docker-compose for dev, so edits reload without rebuilding.)
COPY pyproject.toml ./
COPY services ./services
# Install only the api-runtime extras (FastAPI, LlamaIndex, pgvector, LLM SDKs).
# Skips sentence-transformers/pymupdf/yaml that this container never uses, which
# keeps the image small (and the EC2 disk happy at build time).
RUN pip install --no-cache-dir ".[api]"

EXPOSE 8000
# Default command; docker-compose overrides this with --reload for dev.
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
