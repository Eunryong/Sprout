FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "/app/.venv/bin/python -m server.web --host 0.0.0.0 --port ${PORT:-8080} --lang en"]
