FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md config.yaml preload_models.py start.sh /app/
COPY src /app/src

RUN pip install --upgrade pip \
    && pip install . \
    && chmod +x /app/start.sh

EXPOSE 9200

CMD ["/app/start.sh"]
