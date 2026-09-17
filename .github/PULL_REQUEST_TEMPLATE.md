## What changed and why

## How it was verified

- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy`
- [ ] `uv run pytest` (coverage gate 90 %)
- [ ] `uv run pytest -m models` if `inference.py`, `pipeline.py` or `render.py` changed
- [ ] `uv run pytest -m e2e` if a template, the form or a route changed
- [ ] `uv run python scripts/check_text.py` and `uv run python scripts/gen_algorithms_doc.py --check`
- [ ] CHANGELOG.md updated under Unreleased
