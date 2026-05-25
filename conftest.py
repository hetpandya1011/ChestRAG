"""Pytest config: ensure repo-root packages (services, ingest) are importable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
