"""Shared pytest configuration and fixtures for the turbofan-rul-prediction test suite."""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src` is importable when pytest is
# invoked from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
