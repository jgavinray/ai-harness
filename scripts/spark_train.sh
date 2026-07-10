#!/usr/bin/env bash
# Flywheel phase 2: train a LoRA adapter on the DGX Spark from the gated
# corpus. Run from the harness host (this repo's root). Idempotent: each
# step checks its own completion. See
# docs/superpowers/runbooks/spark-training.md for the full runbook and the
# decisions behind each step.
#
# Usage:
#   scripts/spark_train.sh sync       # ship corpus + trainer + Dockerfile
#   scripts/spark_train.sh download   # start/resume bf16 base download (bg)
#   scripts/spark_train.sh image      # build the spark-trainer image
#   scripts/spark_train.sh train      # stop heretic serving, train, restart
#   scripts/spark_train.sh fetch      # copy the adapter back to ./adapters/
#   scripts/spark_train.sh status     # download/training progress at a glance
set -euo pipefail

SPARK="${SPARK_HOST:-192.168.0.33}"
WORK=/home/jgavinray/models                  # Spark-side working dir
BASE_ID="Qwen/Qwen3.6-27B"                   # from int4-AutoRound README base_model
BASE_DIR="$WORK/Qwen3.6-27B"
CORPUS_LOCAL="corpus/corpus.jsonl"
CORPUS_REMOTE="$WORK/corpus-$(date +%F).jsonl"
ADAPTER="$WORK/adapters/$(date +%F)"
IMAGE=spark-trainer:latest
HERETIC_COMPOSE=/opt/qwen36                  # sacrificial-tier vLLM compose dir
TRAIN_LOG="$WORK/train-$(date +%F).log"

run() { ssh -o BatchMode=yes "$SPARK" "$@"; }

case "${1:-}" in
sync)
    scp -q "$CORPUS_LOCAL" "$SPARK:$CORPUS_REMOTE"
    scp -q scripts/qlora_train.py "$SPARK:$WORK/qlora_train.py"
    scp -q docker/spark-trainer/Dockerfile "$SPARK:$WORK/Dockerfile.spark-trainer"
    run "du -h $CORPUS_REMOTE"
    ;;
download)
    # hf CLI lives in a venv: DGX OS host python is PEP 668 externally-managed.
    run "test -x /home/jgavinray/.hfenv/bin/hf || (python3 -m venv /home/jgavinray/.hfenv && /home/jgavinray/.hfenv/bin/pip install -q -U huggingface_hub)"
    run "nohup /home/jgavinray/.hfenv/bin/hf download $BASE_ID --local-dir $BASE_DIR > $WORK/qwen36-base-download.log 2>&1 & echo download pid \$!"
    ;;
image)
    run "docker build -q -t $IMAGE -f $WORK/Dockerfile.spark-trainer $WORK"
    ;;
train)
    # The Spark's 121 GB unified memory is pinned by the heretic vLLM
    # (sacrificial tier) — stop it for the window, always restart after.
    run "cd $HERETIC_COMPOSE && docker compose stop"
    run "nohup docker run --rm --gpus all --ipc=host \
        -v $WORK:/work $IMAGE \
        python3 /work/qlora_train.py \
            --model /work/Qwen3.6-27B \
            --data $CORPUS_REMOTE \
            --out ${ADAPTER} \
            --quant none \
        > $TRAIN_LOG 2>&1 && cd $HERETIC_COMPOSE && docker compose start \
        & echo train pid \$!"
    echo "training started; heretic restarts automatically on success."
    echo "on failure restart it manually: ssh $SPARK 'cd $HERETIC_COMPOSE && docker compose start'"
    ;;
fetch)
    mkdir -p adapters
    scp -qr "$SPARK:$ADAPTER" adapters/
    ls adapters/
    ;;
status)
    run "tail -2 $WORK/qwen36-base-download.log 2>/dev/null; du -sh $BASE_DIR 2>/dev/null; tail -3 $TRAIN_LOG 2>/dev/null; docker ps --format '{{.Names}} {{.Status}}' | head -3"
    ;;
*)
    grep '^#   scripts/' "$0" | sed 's/^# *//'
    exit 1
    ;;
esac
