FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Bake the (small) retrieval models into the image so the first request is fast. Disable with --build-arg PREFETCH_MODELS=false
ARG PREFETCH_MODELS=true
RUN if [ "$PREFETCH_MODELS" = "true" ]; then python scripts/prefetch_models.py; fi

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
