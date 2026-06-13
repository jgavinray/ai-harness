#!/usr/bin/env python3
"""Eval runner: tasks x configs x trials through the real Claude Code CLI.

Usage:
  .venv/bin/python evals/run.py --backend-url http://192.168.0.196:8001/v1 \
      --model qwen3.6-27b --profile qwen --kind vllm \
      --configs baseline,full --trials 3 --out evals/results

Each trial: copy repo_template -> tmpdir, git init, start harness on a free
port with the generated config, run `claude -p` against it, run check.sh,
append one row to results.jsonl with metrics from the request log.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from configs import config_matrix, write_configs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = Path(__file__).resolve().parent / "tasks"
PYTHON = str(ROOT / ".venv" / "bin" / "python")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(url: str, timeout_s: float = 15.0) -> bool:
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def aggregate_log(log_path: Path) -> dict:
    agg = {
        "requests": 0, "input_tokens": 0, "output_tokens": 0, "retries": 0,
        "repaired_calls": 0, "valid_calls": 0, "invalid_calls": 0,
        "degenerate_aborts": 0, "tool_surfaced": 0, "guard_fires": 0,
        "plan_drift": 0, "wall_ms": 0,
    }
    if not log_path.exists():
        return agg
    for line in log_path.read_text().splitlines():
        rec = json.loads(line)
        agg["requests"] += 1
        for key in agg:
            if key != "requests":
                value = rec.get(key) or 0
                if key == "guard_fires" and isinstance(value, dict):
                    value = sum(value.values())
                agg[key] += value
    return agg


def run_trial(task_dir: Path, cfg_path: Path, port: int, log_path: Path,
              claude_bin: str, timeout_s: int, tag: str = "") -> dict:
    workdir = Path(shutil.copytree(task_dir / "repo_template",
                                   Path(os.environ.get("TMPDIR", "/tmp")) /
                                   f"eval-{task_dir.name}-{time.time_ns()}"))
    subprocess.run(["git", "init", "-q"], cwd=workdir, check=True)
    subprocess.run(["git", "add", "-A"], cwd=workdir, check=True)
    subprocess.run(["git", "-c", "user.email=eval@local", "-c", "user.name=eval",
                    "commit", "-qm", "initial"], cwd=workdir, check=True)

    server = subprocess.Popen(
        [PYTHON, "-m", "harness", "--config", str(cfg_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT,
        env=dict(os.environ, HARNESS_TRACE_TAG=tag),
    )
    try:
        if not wait_for(f"http://127.0.0.1:{port}/stats"):
            return {"success": False, "error": "harness did not start"}

        prompt = (task_dir / "prompt.txt").read_text().strip()
        env = dict(
            os.environ,
            ANTHROPIC_BASE_URL=f"http://127.0.0.1:{port}",
            ANTHROPIC_API_KEY="local",
            CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
        )
        start = time.time()
        try:
            proc = subprocess.run(
                [claude_bin, "-p", prompt,
                 "--allowedTools", "Read,Edit,Write,Bash,Grep,Glob,WebFetch"],
                cwd=workdir, env=env, capture_output=True, text=True,
                timeout=timeout_s,
            )
            answer = proc.stdout
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            answer = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            timed_out = True
        session_wall_s = round(time.time() - start, 1)

        (workdir / "answer.txt").write_text(answer or "")
        check = workdir / "check.sh"
        shutil.copy(task_dir / "check.sh", check)
        result = subprocess.run(["bash", str(check)], capture_output=True, text=True, timeout=60)

        row = {
            "success": result.returncode == 0 and not timed_out,
            "timed_out": timed_out,
            "session_wall_s": session_wall_s,
            "check_output": (result.stdout + result.stderr)[-500:],
        }
        row.update(aggregate_log(log_path))
        return row
    finally:
        server.terminate()
        server.wait(timeout=10)
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--profile", default="qwen")
    ap.add_argument("--kind", default="openai")
    ap.add_argument("--configs", default="baseline,full")
    ap.add_argument("--tasks", default=",".join(sorted(p.name for p in TASKS_DIR.iterdir())))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--claude-bin", default="claude")
    ap.add_argument("--out", default="evals/results")
    args = ap.parse_args()

    names = args.configs.split(",")
    unknown = set(names) - set(config_matrix())
    if unknown:
        sys.exit(f"unknown configs: {unknown}; available: {sorted(config_matrix())}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    for config_name in names:
        port = free_port()
        log_path = out_dir / f"requests-{config_name}.jsonl"
        cfg_paths = write_configs(
            out_dir / "configs", [config_name],
            backend_url=args.backend_url, model=args.model, profile=args.profile,
            kind=args.kind, port=port, log_path=str(log_path),
            traces_dir=str(out_dir / "traces"),
        )
        for task_name in args.tasks.split(","):
            for trial in range(args.trials):
                log_path.unlink(missing_ok=True)
                tag = f"{args.model}-{config_name}-{task_name}-{trial}"
                row = run_trial(TASKS_DIR / task_name, cfg_paths[config_name],
                                port, log_path, args.claude_bin, args.timeout, tag)
                row.update({"task": task_name, "config": config_name, "trial": trial,
                            "model": args.model, "tag": tag})
                with results_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                status = "PASS" if row.get("success") else "FAIL"
                print(f"[{config_name}/{task_name}/{trial}] {status} "
                      f"wall={row.get('session_wall_s')}s retries={row.get('retries')}")

    print(f"\nresults: {results_path}\nreport: {PYTHON} evals/report.py {results_path}")


if __name__ == "__main__":
    main()
