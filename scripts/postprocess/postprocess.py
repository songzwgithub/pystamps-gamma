#!/usr/bin/env python3
from __future__ import annotations

"""Compatibility wrapper for the canonical pySTAMPS postprocess.

The old implementation duplicated scientific logic, hard-coded phuw2.mat,
and wrote to postprocess_parity. The canonical module follows the Stage-8
phase provenance marker, so deramp/GACOS products are handled correctly.
"""

from pystamps.postprocess_core import main


if __name__ == "__main__":
    main()
