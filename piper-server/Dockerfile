FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md config.turkish.yaml preload_models.py start.sh /app/
COPY src /app/src

RUN pip install --upgrade pip \
    && pip install . \
    && chmod +x /app/start.sh

EXPOSE 9200

CMD ["/app/start.sh"]
