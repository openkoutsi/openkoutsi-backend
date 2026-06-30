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

COPY backend/ backend/

# Run as a non-root user. /app/data is the default DATA_DIR; in production a
# volume is mounted over it (DATA_DIR=/data) for the sensitive SQLite databases.
RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app \
    && chmod +x backend/scripts/docker-entrypoint.sh \
    && mkdir -p /app/data \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health').status==200 else 1)"

# Entrypoint runs DB migrations (registry + per-user) before exec'ing uvicorn,
# so a freshly pulled image self-applies schema upgrades on start.
CMD ["backend/scripts/docker-entrypoint.sh"]
