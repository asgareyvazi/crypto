"""Shared pytest configuration."""
import sys
from pathlib import Path

# Ensure src/ is on sys.path (editable install also does this, but keep belt-and-braces).
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
