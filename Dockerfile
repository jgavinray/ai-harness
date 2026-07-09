FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# git + node/claude are for the flywheel service (eval sentinel drives the
# real Claude Code CLI against the local harness); serving doesn't use them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git nodejs npm \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code \
    && npm cache clean --force

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir .[analytics]

COPY scripts ./scripts
COPY evals ./evals

RUN useradd --create-home --shell /usr/sbin/nologin harness \
    && mkdir -p /config /app/logs /app/traces /app/corpus /app/evals/results \
       /home/harness/.ai-harness \
    && chown -R harness:harness /config /app/logs /app/traces /app/corpus \
       /app/evals /home/harness/.ai-harness

USER harness

EXPOSE 8484

CMD ["python", "-m", "harness", "--config", "/config/harness.toml"]
