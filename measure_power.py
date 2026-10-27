#!/usr/bin/env python3
"""Backward-compatible entry point for ESP32/Joulescope measurements."""
from eval_snn_ann.measurement.power import *  # noqa: F401,F403
from eval_snn_ann.measurement.power import main

if __name__ == "__main__":
    main()
