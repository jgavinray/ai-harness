FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir .

RUN useradd --create-home --shell /usr/sbin/nologin harness \
    && mkdir -p /config /app/logs \
    && chown -R harness:harness /config /app/logs

USER harness

EXPOSE 8484

CMD ["python", "-m", "harness", "--config", "/config/harness.toml"]
