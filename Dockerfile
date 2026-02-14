FROM python:3.14.2-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /usr/src/app

RUN pip install --target . awslambdaric

# Copy workspace config and lock file
COPY pyproject.toml uv.lock ./

# Copy workspace members
COPY langevals_core/ langevals_core/
COPY evaluators/ evaluators/
COPY notebooks/pyproject.toml notebooks/

# Install dependencies (frozen from lock file, no dev deps, all extras)
RUN uv sync --frozen --no-dev --all-extras

# Copy application code
COPY langevals/ langevals/

# Preload evaluators
RUN PYTHONPATH="." uv run python langevals/server.py --preload

ENV RUNNING_IN_DOCKER=true

COPY . .

CMD ["uv", "run", "python", "langevals/server.py"]
