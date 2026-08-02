"""Run the local AI4Mars Research Console server."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the AI4Mars Research Console locally.")
    parser.add_argument("--host", default="127.0.0.1", help="Local bind host (default: 127.0.0.1).")
    parser.add_argument("--port", default=8000, type=int, help="Local port (default: 8000).")
    parser.add_argument("--runs-root", type=Path, help="Override the default outputs/runs directory.")
    parser.add_argument("--reload", action="store_true", help="Enable development auto-reload.")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    uvicorn.run(
        create_app(repo_root=repo_root, runs_root=args.runs_root),
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()