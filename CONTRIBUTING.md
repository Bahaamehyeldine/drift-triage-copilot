# Contributing

This started as a solo portfolio project, so this document is intentionally light — it exists so that if you *do* pick it up (including future-me), the conventions are written down instead of tribal knowledge.

## Development setup

Each service manages its own dependencies independently via [`uv`](https://github.com/astral-sh/uv):

```bash
# Run any service's code against exactly its own locked dependency set
uv run --with-requirements <service>/requirements.txt python3 -m <module>

# e.g.
uv run --with-requirements training/requirements.txt python3 -m training.train
```

There is no single virtualenv for the whole repo — `model_service/requirements.txt` deliberately doesn't include `pandas`/`scikit-learn`, and `training/requirements.txt` doesn't include `fastapi`. Each service ships only what it actually imports.

## Adding or changing a dependency

Edit the relevant `<service>/requirements.in` (the human-edited source, unpinned), then recompile the lock file:

```bash
uv pip compile <service>/requirements.in -o <service>/requirements.txt
```

If the change touches `model_service`, `agent`, `worker`, `dashboard`, or `training`, also regenerate the combined CI lock file so `ruff`/`pytest` in CI see the same set:

```bash
uv pip compile model_service/requirements.in agent/requirements.in worker/requirements.in \
    dashboard/requirements.in training/requirements.in -o requirements-test.txt
```

Never hand-edit a `requirements.txt` — it's a generated lock file.

## Before opening a PR

Run the same four gates CI runs, in order — each one catches a different class of problem, and the smoke test in particular has caught real infrastructure bugs (an MLflow security-middleware gap, a stale-volume registry-seeding gap) that unit tests alone never would:

```bash
uv run --with-requirements requirements-test.txt --with ruff ruff check .
uv run --with-requirements requirements-test.txt --with ruff ruff format --check .
uv run --with-requirements requirements-test.txt --with pytest pytest tests/unit
bash tests/integration/smoke_test.sh
```

`ruff format` won't rewrite string literals to fit the line-length limit — if you hit an `E501` on an f-string, wrap it manually as adjacent string literals (see any existing `raise RuntimeError(...)` for the pattern used throughout this codebase).

## Commit conventions

This repo loosely follows [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `build:`, `ci:`, `chore:`, `test:`, `data:`, optionally scoped (`feat(training): ...`). Look at `git log --oneline` for the actual pattern in use — it's more reliable than this paragraph.

Prefer one commit per coherent, independently-reviewable change over one giant commit per session. A change that touches training, infrastructure config, and documentation together is fine as one commit *if* those three things are genuinely one feature (see `5f1f574` for an example) — don't force an artificial split just to hit a smaller diff.

## Design documents

Before making an architectural change, check whether it's already covered in [`ARCH.md`](ARCH.md) (component ownership) or [`DECISIONS.md`](DECISIONS.md) (contracts and the reasoning behind them). If your change contradicts either, update the document in the same PR — these are meant to track reality, not the other way around.
