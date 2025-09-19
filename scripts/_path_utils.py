"""Helpers for making CLI scripts importable without installation."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root(*, package_dir: str = "library") -> Path:
    """Inserts the project root and package directory into `sys.path`.

    This function is idempotent, meaning that repeated calls will not create
    duplicate `sys.path` entries. The package directory is placed ahead of the
    project root to allow packages such as `chembl2uniprot` (stored inside the
    `library` folder) to be imported as top-level modules when running the
    scripts directly.

    Args:
        package_dir: The name of the directory within the project root that
            contains importable modules. The directory is added to `sys.path`
            if it exists.

    Returns:
        The absolute path to the project root directory.
    """

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    package_path = project_root / package_dir
    package_str = str(package_path)
    if package_path.exists() and package_str not in sys.path:
        sys.path.insert(0, package_str)

    return project_root
