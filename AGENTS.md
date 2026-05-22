# Speech Segmentation Agent Guidelines

- This repo uses `uv` for Python dependency management and virtual environment setup.
- Always use `uv run <command>` instead of `python <script>` when executing code (e.g., `uv run python main.py`).
- Use `uv add <package>` to install dependencies rather than pip or poetry.
- Dependencies are managed via `pyproject.toml` and locked in `uv.lock`.
- Activate the virtual environment with `source .venv/bin/activate` when needed, but prefer `uv run` for one-off commands.
