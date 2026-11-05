FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app
COPY pyproject.toml README.md ./
COPY ai_lab ./ai_lab
RUN pip install --no-cache-dir ".[test]"

COPY config ./config
COPY tests ./tests
CMD ["python", "-m", "ai_lab.cli", "--help"]

