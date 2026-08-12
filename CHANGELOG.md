# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [Semantic Versioning](https://semver.org/) once a first release is tagged.

This file tracks user-facing capability changes, not every commit — see `git log` for the full development history, including the fix-up commits that are development detail rather than shipped features.

## [Unreleased]

### Added

**Walking-skeleton platform**
- HMAC-SHA256 signed, deduplicated drift webhook contract between Model Service and Agent
- Agent: webhook signature verification, atomic investigation creation, durable LangGraph checkpoint persistence
- Redis Streams retrain-job dispatch with a consumer-group Worker doing idempotent, at-least-once job processing
- Read-only Streamlit Dashboard over investigation and job state
- Full `docker-compose` stack: Postgres, Redis, Agent, Worker, Model Service, Dashboard, plus a dedicated Alembic migration runner
- GitHub Actions CI: lint, unit tests, and a full end-to-end integration smoke test on every push to `main`

**Training and model lifecycle**
- Preprocessing pipeline with stratified 60/20/20 train/validation/test split, integrity-checked for row overlap and completeness
- Controlled `class_weight=None` vs `class_weight="balanced"` candidate comparison
- Operating-threshold selection on validation data only, under a `recall >= 0.75` business constraint
- MLflow model registration with full training provenance (dataset hash, split-specification hash, exact split-membership hash, git commit, working-tree dirty state) recorded as both run parameters and registered-model-version tags
- Sealed, one-time final test-set evaluation (`training/evaluate.py`) that refuses to re-run against an already-evaluated model version and independently re-verifies dataset/split provenance before touching test data
- `shared/preprocessing.py`: the feature transformer extracted into its own package so Model Service can deserialize the registered pipeline without depending on the training subsystem
- Model Service: loads the registered model from the MLflow registry at startup (`MODEL_VERSION=latest` or pinned), validates its structure, and fails startup fast on any invalid or missing state rather than serving in a broken condition
- `/debug/drift` reports the real registered model's name and version instead of placeholder values
- `training/predict_examples.py`: a small utility that runs the registered model against real validation examples and reports probability, threshold, and prediction alongside ground truth

### Fixed
- MLflow's server rejected in-network requests (`http://mlflow:5000`) due to its default DNS-rebinding protection trusting only `localhost` and private IPs; every earlier host-side test had used `localhost:5000` and never exercised this path
- The integration smoke test called a bare `python` binary not present in every environment; switched to `python3`
- The integration smoke test didn't register a model before starting Model Service after a fresh volume reset, since Model Service's startup now depends on the registry being non-empty

### Documentation
- `README.md` rewritten: architecture diagrams, technology rationale, frozen model metrics, explicit scope boundaries
- `ARCH.md`, `DECISIONS.md`, `MVP_SCOPE.md`, `RUNBOOK.md`
- `LICENSE` (MIT), `CONTRIBUTING.md`, `SECURITY.md`
