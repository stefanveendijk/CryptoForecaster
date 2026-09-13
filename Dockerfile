FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY . .

CMD ["sh", "-c", "python version_trigger.py; (sleep 30; echo MODEL_STATE_AFTER_START; cat /data/cloud_state.json 2>/dev/null || true) & exec python -m uvicorn cloud_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
