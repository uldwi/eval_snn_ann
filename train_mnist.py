#!/usr/bin/env python3
"""Backward-compatible entry point; implementation lives in ``eval_snn_ann``."""
from eval_snn_ann import *  # noqa: F401,F403
from eval_snn_ann.cli import main

if __name__ == "__main__":
    main()
