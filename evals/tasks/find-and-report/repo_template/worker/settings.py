import json
import os

DEFAULTS = {"pool_size": 4, "timeout_s": 30}


def load():
    cfg = dict(DEFAULTS)
    override = os.environ.get("WORKER_POOL_SIZE")
    if override is not None:
        cfg["pool_size"] = int(override)
    path = os.environ.get("WORKER_CONFIG")
    if path and os.path.exists(path):
        cfg.update(json.loads(open(path).read()))
    return cfg
