# MVP Scope — Walking Skeleton

## Philosophy

The MVP follows a **walking skeleton** approach: implement the thinnest viable version of every component and prove the complete end-to-end workflow before adding production functionality.

At this stage, the primary risk is not implementing an individual service—it is ensuring that the system behaves correctly as a distributed application. The platform consists of multiple independently deployed components (Model Service, Agent, Worker, Dashboard, Postgres, and Redis) whose value depends on reliable communication, shared contracts, and consistent state transitions.

Building one service to completion before integrating it would postpone the discovery of integration failures such as contract mismatches, networking issues, inconsistent identifiers, queue semantics, or persistence assumptions. The walking skeleton minimizes these risks by validating the complete architecture as early as possible.

The initial success criterion is intentionally narrow:

> A hardcoded drift event is emitted by the Model Service, received by the Agent, persisted as an investigation, dispatched as an asynchronous job, processed by a worker with idempotent execution, and displayed in the Dashboard.

The objective of the MVP is to validate **architecture, contracts, connectivity, and ownership boundaries**, not production-ready functionality.

## Post-MVP: Training Vertical

The walking skeleton above is complete. The exclusions listed for the Model Service below ("model training", "MLflow integration beyond basic service wiring") described that first phase only. The next phase implements the real training vertical: preprocessing, a stratified train/validation/test split, a controlled candidate comparison (`class_weight=None` vs `balanced`), threshold selection under the `recall >= 0.75` business constraint, and MLflow model registration — evaluated on training and validation data only, with the test split still untouched. Wiring the registered model into the Model Service's inference path remains a separate, later step.

---

# Scope by Component

## Model Service

The Model Service exposes a minimal debug endpoint capable of emitting a deterministic drift webhook using the agreed webhook contract.

The MVP proves that:

- the webhook payload is generated correctly;
- requests are HMAC signed;
- the Agent is reachable over the internal network;
- delivery succeeds end-to-end.

The MVP intentionally excludes:

- model training;
- inference;
- MLflow integration beyond basic service wiring;
- real drift computation.

---

## Triage Agent

The Agent exposes the webhook endpoint and executes a minimal deterministic workflow.

For each valid webhook it:

- verifies the HMAC signature;
- deduplicates the event using `report_id`;
- creates an investigation record;
- persists a LangGraph checkpoint;
- dispatches a stub retrain job to Redis;
- updates investigation status.

The full supervisor, multi-agent orchestration, recommendation generation, and human approval workflow are intentionally deferred.

---

## Async Tool Worker

The worker consumes retrain jobs from Redis and validates durable idempotent execution.

Before executing work, it atomically claims the job using `retrain_job_id`.

If the claim succeeds:

- mark the job as running;
- complete the stub execution;
- persist the result.

If the job has already been claimed:

- skip execution;
- return the existing state.

The worker performs no real retraining in the MVP. Its purpose is to validate queue consumption, durable idempotency, and result persistence.

---

## Dashboard

The Dashboard provides a minimal operational view backed by Postgres.

It displays:

- investigation ID;
- model name;
- report ID;
- investigation status;
- async job status;
- last update timestamp.

No filtering, charts, registry management, approval inbox, or operator actions are included in the MVP.

The goal is simply to verify that investigation state can be queried and presented independently of LangGraph checkpoints.

---

## Docker Compose

Docker Compose provisions the complete development environment.

Services include:

- Model Service
- Triage Agent
- Async Tool Worker
- Dashboard
- Postgres
- Redis

The stack provides:

- shared networking;
- environment-based configuration;
- persistent storage;
- health checks;
- startup dependency ordering.

Running `docker compose up` should produce a fully connected system without manual service startup.

---

## CI

Continuous Integration validates the walking skeleton as a complete system.

The pipeline will:

- run unit tests;
- build service images;
- validate database schema or migrations;
- start the Docker Compose stack;
- wait for service readiness;
- trigger the deterministic drift event;
- verify investigation creation;
- verify checkpoint persistence;
- verify successful worker execution;
- verify idempotent retry behaviour;
- verify invalid HMAC signatures are rejected.

The CI pipeline validates the architecture rather than production functionality.

---

# Authentication in the MVP

Authentication is included from the first implementation.

Since request authentication is already part of the agreed service contracts, omitting it from the MVP would validate a different execution path than the one intended for production.

The MVP implementation includes:

- a single shared secret provided through environment variables;
- HMAC request signing by the sender;
- HMAC verification by the receiver;
- constant-time signature comparison;
- automated tests for both valid and invalid signatures.

The following are intentionally deferred:

- key rotation;
- multiple active keys;
- external secret management;
- automated secret provisioning;
- advanced replay-window protection.

---

# Explicitly Deferred

The following capabilities are intentionally **out of scope** for the walking skeleton.

## Model Functionality

- Real model training
- Model inference
- Feature engineering
- Dataset versioning
- Performance evaluation
- Production model serving

---

## Drift Detection

- PSI computation
- Chi-square tests
- Output drift analysis
- Scheduled drift checks
- Baseline management
- Severity calculation

The MVP uses a deterministic hardcoded drift event.

---

## Agent Intelligence

- Full LangGraph supervisor
- Multi-agent coordination
- LLM reasoning
- Recommendation generation
- Human approval workflow
- Staleness validation
- Investigation replay

The MVP executes a single deterministic workflow.

---

## Promotion Workflow

- Promotion endpoint
- Promotion idempotency
- `based_on_report_id` validation
- Registry promotion
- Stage transitions
- Promotion audit

The initial walking skeleton ends after successful async job execution.

---

## Async Tool Logic

- Replay implementation
- Retraining pipeline
- Rollback implementation
- Artifact generation
- Model registration
- Evaluation pipeline

Workers execute stub jobs only.

---

## Queue Operations

- Retry strategies
- Exponential backoff
- Dead-letter queue management
- Job cancellation
- Worker concurrency
- Priority queues
- Lease renewal

Only the minimum path required to validate idempotent execution is implemented.

---

## Dashboard Features

- Investigation filtering
- Search
- Pagination
- Approval inbox
- Registry management
- Queue administration
- Charts
- Metrics
- Authentication

The MVP Dashboard is intentionally read-only.

---

## Production Readiness

- Observability
- Distributed tracing
- Metrics
- Alerting
- TLS
- RBAC
- Secret rotation
- High availability
- Disaster recovery
- Multi-environment deployment

These concerns will be addressed after the end-to-end architecture has been validated.

---

## Database Migrations

The MVP requires a reproducible database schema.

The migration mechanism (e.g. Alembic or versioned SQL migrations) will be selected before implementation but is not part of this milestone.