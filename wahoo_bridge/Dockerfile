# ── Build stage: install dependencies with uv ────────────────────────────
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY main.py __init__.py ./

# Run as a non-root user; the bridge SQLite DB lives on a mounted volume.
RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app \
    && chown -R app:app /app
USER app

EXPOSE 8085

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8085/docs').status==200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8085"]
