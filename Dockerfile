FROM python:3.12-slim

# Install uv for deterministic dependency sync from pyproject + uv.lock
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first to leverage Docker layer cache
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy project files
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
