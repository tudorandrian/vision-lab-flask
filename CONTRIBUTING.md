# Contributing

Thanks for considering a contribution to vision-lab-flask.

## Setup

```bash
git clone https://github.com/tudorandrian/vision-lab-flask.git
cd vision-lab-flask
uv sync --extra yolo
uv run playwright install chromium
```

`--extra yolo` also lets you run the `-m models` tests. The `emotion` extra pulls in TensorFlow
and is not needed for most changes; see README "Optional extras".

- After any change to `uv.lock` (`uv add`, `uv lock --upgrade`), run
  `uv export --frozen --no-dev --no-emit-project --no-hashes -o requirements.txt` and commit the
  result; CI fails when the two files disagree. On Dependabot's uv pull requests the
  `Dependabot requirements` workflow does this for you: it pushes one commit, and CI runs on it
  after a maintainer selects "Approve workflows to run" in the pull request. After that commit
  Dependabot no longer rebases the pull request; comment `@dependabot recreate` if it needs
  refreshing, and the workflow runs again.

## Before you open a pull request

All of these must pass; CI runs the same commands:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run bandit -q -r src
uv run python scripts/check_text.py
uv run python scripts/gen_algorithms_doc.py --check
uv run pip-audit --skip-editable
uv run pytest --cov
```

Run the marker-based groups too if your change touches the area they cover:

- `uv run pytest -m models` after changing `inference.py`, `pipeline.py` or `render.py`.
- `uv run pytest -m e2e` after changing a template, the form or a route.

## Style

- Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `build:`, `ci:`, ...) for commit
  messages and pull request titles.
- No em dash (U+2014) anywhere; `scripts/check_text.py` enforces this. Use a plain hyphen instead.
- British spelling in prose (colour, behaviour, licence as a noun).
- Update `CHANGELOG.md` under `[Unreleased]` for any user-visible change.

## Licensing

Contributions are accepted under this project's licence, AGPL-3.0-or-later (inbound equals
outbound): by submitting a change, you agree it can be distributed under those terms.

## Reporting security issues

Do not open a public issue for a security problem. See [SECURITY.md](SECURITY.md).

## What makes a good issue

Say what you ran (uv, Docker or pip; OS; Python version; extras installed), what you expected,
what happened instead, and the exact error text or log line. For a feature request, say which
existing operation or model it is closest to, and why it belongs in a teaching project rather
than a production one.
