"""Non-interactive reproducible AI4Mars segmentation training entry point.

Delegates to the paper-aligned DeepLabv3+ implementation in ``src.paper_train``.
Invoke as ``python -m src.train --config <config>``.
"""
from __future__ import annotations

from src.paper_train import main as paper_main


def main() -> None:
    paper_main()


if __name__ == "__main__":
    main()
