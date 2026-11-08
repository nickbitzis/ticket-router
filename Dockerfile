# Build stage: resolve and install dependencies with uv into a virtualenv.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Runtime stage: slim image, non-root user, only the venv and app code.
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 router
WORKDIR /app
COPY --from=build --chown=router:router /app /app
# The feedback store writes to /app/data at runtime, so it must exist and belong to the app user.
RUN mkdir -p /app/data && chown router:router /app/data
USER router
ENV PATH="/app/.venv/bin:$PATH" HOST=0.0.0.0 PORT=8836 TICKETROUTER_DB=/app/data/feedback.db
EXPOSE 8836
# Startup trains the classical tier and runs the threshold sweep, about 40 seconds on 20,000 tickets.
HEALTHCHECK --interval=15s --timeout=3s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8836/healthz', timeout=2)"
CMD ["ticketrouter", "serve"]
