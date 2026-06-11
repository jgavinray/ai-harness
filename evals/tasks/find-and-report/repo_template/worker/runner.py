from worker.settings import load


def start():
    cfg = load()
    print(f"starting with {cfg['pool_size']} workers")
