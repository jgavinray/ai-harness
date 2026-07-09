"""Compose-native scheduler for the self-improvement loops (flywheel spec).

One process (`python -m harness.flywheel --config …`) runs the deterministic
growth-loop jobs as subprocesses of the bundled scripts: nightly memory
distill, corpus rebuild, partition retention, DuckDB refresh, and skill
compilation; weekly the envelope sentinel re-runs the eval suite and flags
any family that drops below "supported". Serving is never touched by a job
failure — the only shared surface is the data-plane volumes. Every job run
appends one JSONL record to the flywheel log; /stats surfaces the tail.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from harness.config import Settings, load_settings
from harness.log import RequestLogger

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
JOB_TIMEOUT_S = 6 * 3600
OUTPUT_TAIL_CHARS = 2000


def next_nightly(now: datetime, hour: int) -> datetime:
    run = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if run <= now:
        run += timedelta(days=1)
    return run


def prune_partitions(
    requests_dir: Path | None,
    traces_dir: Path | None,
    days: int,
    now: float | None = None,
) -> list[str]:
    """Delete date-named partitions older than the cutoff. Anything not
    matching YYYY-MM-DD (legacy files, sessions.jsonl) is never touched."""
    cutoff = (
        datetime.fromtimestamp(now if now is not None else time.time())
        - timedelta(days=days)
    ).strftime("%Y-%m-%d")
    removed: list[str] = []
    if requests_dir and requests_dir.is_dir():
        for f in requests_dir.glob("*.jsonl"):
            if DATE_RE.match(f.stem) and f.stem < cutoff:
                f.unlink()
                removed.append(str(f))
    if traces_dir and traces_dir.is_dir():
        for d in traces_dir.iterdir():
            if d.is_dir() and DATE_RE.match(d.name) and d.name < cutoff:
                shutil.rmtree(d)
                removed.append(str(d))
    return removed


def run_job(name: str, argv: list[str], cwd: Path, logger: RequestLogger) -> dict:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=JOB_TIMEOUT_S
        )
        rc = proc.returncode
        output = (proc.stdout + proc.stderr)[-OUTPUT_TAIL_CHARS:]
    except Exception as exc:  # job isolation: a broken job must not kill the loop
        rc = -1
        output = str(exc)[-OUTPUT_TAIL_CHARS:]
    record = {
        "ts": time.time(),
        "job": name,
        "argv": argv,
        "rc": rc,
        "duration_s": round(time.monotonic() - start, 1),
        "output_tail": output,
    }
    logger.write(record)
    return record


def _trace_source(settings: Settings) -> str:
    if settings.traces.layout == "partitioned":
        return settings.traces.dir
    return str(Path(settings.traces.dir) / "sessions.jsonl")


def nightly_jobs(settings: Settings) -> list[tuple[str, list[str]]]:
    py = sys.executable
    fly = settings.flywheel
    traces = _trace_source(settings)
    jobs: list[tuple[str, list[str]]] = []
    if settings.memory.enabled:
        jobs.append(("memory_distill", [
            py, "scripts/memory_distill.py",
            "--traces", traces,
            "--memory-dir", settings.memory.dir,
        ]))
    jobs.append(("corpus", [
        py, "scripts/corpus.py",
        "--traces", traces,
        "--results", fly.results_path,
        "--out", fly.corpus_path,
        "--include-live",
    ]))
    jobs.append(("analytics", [
        py, "scripts/analytics.py", "refresh", "--db", fly.analytics_db,
    ]))
    if Path(settings.skills.dir).expanduser().is_dir():
        jobs.append(("compile_skills", [
            py, "scripts/compile_skills.py",
            "--skills-dir", settings.skills.dir,
            "--cache-dir", settings.skills.cache_dir,
        ]))
    return jobs


def nightly_cycle(settings: Settings, logger: RequestLogger, cwd: Path) -> None:
    fly = settings.flywheel
    Path(fly.corpus_path).parent.mkdir(parents=True, exist_ok=True)
    for name, argv in nightly_jobs(settings):
        run_job(name, argv, cwd, logger)
    start = time.monotonic()
    removed = prune_partitions(
        Path(settings.log.requests_dir) if settings.log.requests_dir else None,
        Path(settings.traces.dir) if settings.traces.enabled else None,
        fly.retention_days,
    )
    logger.write({
        "ts": time.time(),
        "job": "retention",
        "rc": 0,
        "duration_s": round(time.monotonic() - start, 1),
        "removed": len(removed),
    })


def sentinel_verdicts(results_path: Path) -> dict[str, str]:
    sys.path.insert(0, str(Path.cwd() / "evals"))
    import report as eval_report  # noqa: E402

    tasks = eval_report.aggregate_tasks(eval_report.load(results_path))
    return {key[2]: metrics["verdict"] for key, metrics in tasks.items()}


def write_sentinel_state(path: str | Path, verdicts: dict[str, str]) -> dict:
    degraded = sorted(f for f, v in verdicts.items() if v != "supported")
    state = {"ts": time.time(), "degraded": degraded, "verdicts": verdicts}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state))
    return state


def sentinel_cycle(
    settings: Settings, logger: RequestLogger, cwd: Path, trials: int | None = None
) -> None:
    fly = settings.flywheel
    backend = settings.backends[0] if settings.backends else None
    if backend is None:
        logger.write({"ts": time.time(), "job": "sentinel", "rc": -1,
                      "output_tail": "no backends configured"})
        return
    out_dir = f"evals/results/sentinel-{time.strftime('%Y-%m-%d')}"
    argv = [
        sys.executable, "evals/run.py",
        "--backend-url", backend.base_url,
        "--model", backend.model,
        "--profile", backend.profile,
        "--kind", backend.kind,
        "--configs", "full",
        "--trials", str(trials or fly.sentinel_trials),
        "--out", out_dir,
    ]
    record = run_job("sentinel", argv, cwd, logger)
    results = cwd / out_dir / "results.jsonl"
    if record["rc"] == 0 and results.exists():
        state = write_sentinel_state(fly.sentinel_state_path, sentinel_verdicts(results))
        logger.write({"ts": time.time(), "job": "sentinel_verdict", "rc": 0,
                      "degraded": state["degraded"], "verdicts": state["verdicts"]})


async def run_loop(settings: Settings, cwd: Path) -> None:
    fly = settings.flywheel
    logger = RequestLogger(fly.log_path)
    while True:
        now = datetime.now()
        nxt = next_nightly(now, fly.nightly_hour)
        await asyncio.sleep(max(1.0, (nxt - now).total_seconds()))
        await asyncio.to_thread(nightly_cycle, settings, logger, cwd)
        if fly.sentinel_weekday >= 0 and datetime.now().weekday() == fly.sentinel_weekday:
            await asyncio.to_thread(sentinel_cycle, settings, logger, cwd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/config/harness.toml")
    ap.add_argument("--once", choices=["nightly", "sentinel"],
                    help="run one cycle immediately and exit")
    ap.add_argument("--trials", type=int, default=None,
                    help="sentinel trial override (with --once sentinel)")
    args = ap.parse_args()
    settings = load_settings(args.config)
    cwd = Path.cwd()
    if args.once:
        logger = RequestLogger(settings.flywheel.log_path)
        if args.once == "nightly":
            nightly_cycle(settings, logger, cwd)
        else:
            sentinel_cycle(settings, logger, cwd, trials=args.trials)
        return
    if not settings.flywheel.enabled:
        sys.exit("[flywheel] enabled = false; nothing to do")
    asyncio.run(run_loop(settings, cwd))


if __name__ == "__main__":
    main()
