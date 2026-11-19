FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN useradd --create-home --uid 10001 app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY agents/ ./agents/
COPY config/ ./config/
COPY graph/ ./graph/
COPY prompts/ ./prompts/
COPY tools/ ./tools/
COPY main.py .

RUN mkdir -p data workspace_data \
    && chown -R app:app /app

USER app

CMD ["python", "main.py"]
