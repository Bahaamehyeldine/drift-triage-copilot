# Decisions Log

## 1. Drift Webhook Contract (Model Service → Agent)

**Decision:**
The Model Service emits one webhook event per drift-check run to the Agent, fired only when the *overall* severity increases relative to the last severity the Agent was successfully notified about (not the last severity computed — see Reasoning). The event bundles every signal evaluated in that run rather than dispatching one call per feature. Numeric features are scored with PSI (severity bands: < 0.1 none, 0.1–0.25 medium, ≥ 0.25 high); categorical features are scored with a chi² test's p-value (bands: ≥ 0.05 none, 0.01–0.05 medium, < 0.01 high — note the direction is inverted relative to PSI, since a *smaller* p-value means a *larger* distributional difference). Delivery uses exponential backoff (3 attempts). The receiver deduplicates using `report_id`. The call is authenticated with a shared-secret header.

Example payload:

```json
{
  "schema_version": "1.0",
  "event_type": "drift.severity.increased",
  "report_id": "drift_report_2026_07_09_18_00_00Z_7f4c92",
  "timestamp": "2026-07-09T18:00:00Z",
  "model": { "name": "customer_churn_classifier", "version": "v3.2.1" },
  "overall_severity": { "previous": "low", "current": "medium" },
  "signals": [
    { "feature": "monthly_income", "test": "psi", "value": 0.34, "severity": "high" },
    { "feature": "country", "test": "chi2", "value": 0.0038, "severity": "high" }
  ],
  "summary": "Overall drift severity increased from LOW to MEDIUM. Two drift signals were detected: monthly_income (PSI=0.34, HIGH) and country (Chi-square p-value=0.0038, HIGH)."
}
```

**Context:**
The platform needs to tell the Agent when something worth investigating has happened, without flooding it with noise or losing alerts to transient failures. The brief requires this to be webhook-driven (or defensibly polled) and treats schema changes as breaking, so the contract needs to be explicit and versioned from day one.

**Reasoning:**
- *Bundled, not per-feature:* the brief refers to "the drift report" in the singular. Bundling every signal from one check cycle into a single event means the Agent always reasons over the full picture at once, and avoids a message-ordering/correlation problem that a per-feature design would create.
- *Fire only on increase, never on improvement:* an "improved" reading carries the same false-signal risk as threshold flapping — one good reading doesn't prove the drift resolved. Investigations are closed by an explicit action, not by a transient "looks better now" event.
- *"Notified" severity, tracked separately from "computed" severity:* if the webhook fails delivery after all retries and we still advanced our internal state to "current = high," a later check where severity doesn't change further would never re-fire — the alert would be silently lost forever. Keeping the two states separate means a failed delivery is retried for free on the next scheduled check.
- *PSI / chi² p-value with fixed bands:* standard, well-documented thresholds; using the p-value (not the raw chi² statistic) keeps the severity rule comparable across categorical features with different numbers of categories.
- *Shared-secret header, not full request signing:* the call never leaves the docker-compose network in this project's scope, so a lightweight shared secret is proportionate. Added anyway as cheap insurance against the network boundary changing later.

**Known limitations (deferred):**
- No debouncing/hysteresis around threshold boundaries.
- No cap on how long the Model Service will silently keep retrying notification if the Agent is down for an extended period.

---

## 2. Promotion Endpoint Contract (Agent → Model Service)

**Decision:**
After a human approves an action in the dashboard's HIL inbox, the Agent calls the Model Service's promotion endpoint with a payload identifying the exact model artifact (name, version, and artifact digest), the approval metadata, and the specific drift report (`based_on_report_id`) the recommendation was generated from. The request is authenticated with an HMAC-SHA256 signature over the request body (keyed by a versioned `key_id`), and is idempotent via `promotion_request_id`. Before executing, the Model Service compares `based_on_report_id` against the latest known report_id it has stored for that model — updated on *every* drift check, regardless of whether that check fired a webhook. A mismatch returns `409 Conflict` with both report IDs; a match proceeds to the existing promotion gate (day-4 checklist). Only the Agent holds the signing key — this endpoint cannot be called directly, bypassing the Agent, in v0.1.

**Context:**
This is the one HTTP call in the system that actually mutates Production. The brief explicitly flags the hard problem here: a human might approve a recommendation that's since gone stale because a newer drift event arrived while the approval was pending.

**Reasoning:**
- *HMAC over the body, not just a shared secret:* this call mutates production state, so it warrants a stronger guarantee than "the caller knows a password" — HMAC also proves the payload wasn't tampered with in transit.
- *`artifact_digest` alongside name/version:* protects against promoting the wrong bytes if a version label were ever reused or corrupted.
- *Staleness via `based_on_report_id` + "latest known report" state:* an optimistic-concurrency check. Tracking "latest report" on every check (not just severity-increasing ones) matters — without it, a quiet improvement between recommendation and approval would go undetected.
- *On a 409, the Agent reuses the existing investigation* rather than discarding it — preserves the audit trail instead of losing prior context.
- *Agent-only caller, no direct/"break-glass" path in v0.1:* allowing a second route to Production would reopen the exact registry/checkpoint desync problem addressed in Decision 5. A proper emergency-override mechanism is real scope, deliberately deferred rather than half-built.

**Known limitations (deferred):**
- No break-glass / manual override path if the Agent itself is down or broken.
- Emergency intervention currently means going around this system entirely via MLflow directly.

---

## 3. Retrain Job — Idempotent Execution

**Decision:**
Every retrain job includes a unique `retrain_job_id`, reused across all retries. Before starting training, the worker atomically checks and claims this ID in Postgres.

**Context:**
Redis may redeliver a job after a network failure, worker restart, or timeout. Without a durable idempotency check, the same logical retrain job could start multiple expensive training runs.

**Reasoning:**
The key must be checked and claimed **atomically** before training begins, not merely included in the queue message. If the ID already exists as running or completed, the worker must not launch another run and should reuse the existing job state or result. A simple "check, then start" approach is not sufficient because two workers could check at nearly the same time, both see that the job has not started, and both begin training. Likewise, checking only Redis or checking after training has already started would still allow duplicate executions during retries or worker restarts.

---

## 4. Checkpoint Resume — Stale Model URI Handling

**Decision:**
When resuming from a checkpoint, the Agent must first ask the Model Service or registry whether `investigating_model_uri` still exists and remains in the expected stage.

**Context:**
A checkpoint contains frozen state that may be outdated if the model was deleted, replaced, promoted, archived, or rolled back while the Agent was paused.

**Reasoning:**
The Agent must not continue using stale checkpoint data. If the model is missing or its state has changed, the investigation is marked stale or invalidated, with the reason recorded and shown in the Dashboard rather than silently discarded.

---

## 5. Checkpoint Store vs. Model Registry — Source of Truth

**Decision:**
Do not attempt to keep the checkpoint store and model registry perfectly synchronized. The model registry is the source of truth, and its live state must be re-verified before every consequential action.

**Context:**
The checkpoint store preserves the Agent's progress and prior context, but registry state can change independently through promotion, rollback, deletion, or other operations.

**Reasoning:**
Trying to maintain strict synchronization would add complexity without eliminating stale-state risks. The safer pattern is to treat checkpointed registry information as a snapshot, defer to the live registry, and stop or invalidate the workflow when the two no longer match. This is the same principle applied concretely in Decision 2 (promotion staleness check) and Decision 4 (checkpoint resume check) — those are two specific instances of this general rule, not separate mechanisms.