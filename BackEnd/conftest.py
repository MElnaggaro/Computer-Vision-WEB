"""
Shared pytest fixtures for the test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the BackEnd package root is on sys.path so that ``app.*`` imports work
# regardless of where pytest is invoked from.
_BACKEND_ROOT = Path(__file__).resolve().parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
