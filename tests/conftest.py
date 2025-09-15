from __future__ import annotations

import sys
from pathlib import Path

# Ensure project and library roots are on sys.path for tests
ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "library"
for path in (ROOT, LIB):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
