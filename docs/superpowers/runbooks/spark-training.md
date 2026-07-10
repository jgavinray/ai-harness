# Runbook: LoRA training on the DGX Spark

Flywheel phase 2 (spec: `docs/superpowers/specs/2026-07-09-self-improving-flywheel-design.md`).
Everything here is driven by `scripts/spark_train.sh` from the harness host;
this document records what each step does and why, so a small-model
maintainer can re-derive or repair the pipeline.

## Topology and decisions (verified 2026-07-09)

- **Training host: DGX Spark** (`192.168.0.33`, aarch64, GB10, 121 GB
  unified memory, 2 TB free disk). User decision: the Spark only serves the
  sacrificial-tier heretic vLLM (`/opt/qwen36/`), which is cheap to stop
  for a training window. The RTX Pro 6000 on `.196` serves `main` and is
  never touched (94/97 GB pinned by vLLM).
- **Base model: `Qwen/Qwen3.6-27B`** (bf16, ~55 GB) — taken from
  `base_model:` in the int4-AutoRound README on `.196`. No unquantized copy
  existed on any host, so it downloads once to
  `/home/jgavinray/models/Qwen3.6-27B` on the Spark.
- **Plain bf16 LoRA, not QLoRA:** bitsandbytes ships no CUDA aarch64
  binary (fails to load in-container on the GB10), so
  `qlora_train.py --quant none`. The 121 GB unified memory holds the 27B
  in bf16 with gradient checkpointing.
- **Training image: `spark-trainer:latest`**, built on the Spark from
  `docker/spark-trainer/Dockerfile` — `FROM quant-tools:v3` (the Spark-local
  image with NVIDIA's ARM torch 2.11 + transformers) plus peft/trl/datasets.
- **Corpus:** the gated corpus (`corpus/corpus.jsonl`, `{"messages": ...}`
  rows, trl conversational format) is scp'd to
  `/home/jgavinray/models/corpus-<date>.jsonl`. 12,708 rows / 506 MB as of
  2026-07-09.
- **hf CLI** lives in `/home/jgavinray/.hfenv` on the Spark: DGX OS host
  python is PEP 668 externally-managed, so no `pip --user`.

## Procedure

```bash
scripts/spark_train.sh sync       # corpus + trainer + Dockerfile -> Spark
scripts/spark_train.sh download   # bf16 base (once; resumable, background)
scripts/spark_train.sh image      # build spark-trainer:latest
scripts/spark_train.sh status     # watch download; wait for ~55 GB complete
scripts/spark_train.sh train      # stops heretic, trains, restarts heretic
scripts/spark_train.sh status     # watch training loss in train-<date>.log
scripts/spark_train.sh fetch      # adapter -> ./adapters/<date>/
```

`train` restarts the heretic automatically on success. **On failure the
heretic stays down** — restart it by hand:
`ssh 192.168.0.33 'cd /opt/qwen36 && docker compose start'` (model reload
takes ~15 min).

## After the adapter exists

1. Serve it as a candidate: add `--enable-lora
   --lora-modules candidate=<adapter dir>` to the vLLM args on `.196`
   (`/storage03/vllm-docker-compose/hyper03/entrypoint.sh`), and give
   `harness.toml` a backend entry with `roles = ["candidate"]`.
   Flagged assumption to verify here: a bf16-base adapter applying cleanly
   over the int4-AutoRound deployment; if vLLM refuses, serve the candidate
   from the bf16 copy instead.
2. `scripts/shadow_eval.py --execute` — full envelope against the candidate.
3. `scripts/promote_candidate.py --results <shadow results.jsonl> ...` —
   prints the proposed roles diff and writes `harness.toml.proposed`;
   applying it is a human-reviewed commit (`--apply` exists but is not the
   default on purpose).

## Known constraints

- Do not run training while the heretic serves: vLLM pins the unified
  memory (`free` shows available 0).
- One epoch over 12.7k rows at max_seq_len 8192 on the GB10 is
  hours-scale; the job runs under nohup and survives SSH disconnects.
- `training_due` records in `logs/flywheel.jsonl` (threshold:
  `flywheel.train_threshold_rows`) tell you when enough new gated corpus
  has accrued to justify the next adapter.
