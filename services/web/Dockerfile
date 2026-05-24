FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml ./
COPY services ./services
RUN pip install --no-cache-dir .

EXPOSE 8501
CMD ["streamlit", "run", "services/web/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
