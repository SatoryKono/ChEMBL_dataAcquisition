from __future__ import annotations

import builtins
import sys
from pathlib import Path

# Ensure library directory is on sys.path for tests
ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "library"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Expose ``build_clients`` for legacy tests expecting a global symbol.
from pipeline_targets_main import build_clients as _build_clients

builtins.build_clients = _build_clients
