FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY openoe ./openoe

RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 --user-group appuser

USER appuser

EXPOSE 8000

CMD openoe --host 0.0.0.0 --port ${PORT:-8000}
