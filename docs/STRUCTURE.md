# Project Structure

This repository is organised into dedicated directories for clarity and reuse.

- `library/` – Python packages and reusable modules.
- `schemas/` – JSON schema definitions.
- `scripts/` – Executable scripts and CLI entry points.
- `docs/` – Project documentation, including this file and the README.
- `tests/` – Unit tests and sample data.

The project root contains the consolidated configuration and meta files:

- `pyproject.toml` – build system and dependency declarations.
- `requirements.txt` – list of runtime and development dependencies.
- `config.yaml` – unified application configuration.
- `.gitignore` – git ignore rules.

Place new code, schemas and scripts in the respective directories to keep the
repository organised.
