---
name: use_uv_for_deps
description: User requires uv for all Python dependency management, not pip or conda
type: feedback
---

Always use `uv` for dependency management in this project (and likely all projects for this user).

**Why:** User explicitly corrected when pip was attempted - "her şeyi uv ile yapmalısın" (you must do everything with uv).

**How to apply:** Use `uv sync`, `uv run`, `uv add` etc. Never use `pip install` or `conda`. Use `pyproject.toml` not `requirements.txt`/`setup.py`.
