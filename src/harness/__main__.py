import argparse

import uvicorn

from harness.config import load_settings
from harness.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="harness", description="Claude Code proxy for small LLMs")
    parser.add_argument("--config", default="harness.toml", help="path to harness.toml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    app = create_app(settings, config_path=args.config)
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
