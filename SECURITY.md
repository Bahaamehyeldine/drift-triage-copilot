# Security Policy

## Reporting a vulnerability

This is a personal portfolio/demonstration project, not a maintained production service with an SLA — but if you find a real security issue, please report it privately rather than opening a public issue: **mehyeldinebahaa@gmail.com**.

Include what you found, how to reproduce it, and its potential impact. There's no bug bounty and no guaranteed response time, but genuine reports will be read and, where reasonable, fixed.

## Scope and honest limitations

This project is designed to demonstrate an MLOps lifecycle end-to-end, not to be internet-facing. If you deploy it anywhere beyond `localhost`, be aware of what's actually in place versus what isn't:

**In place:**
- Drift webhooks (Model Service → Agent) are HMAC-SHA256 signed and verified; an invalid signature is rejected with `401`.
- `mlflow`'s server has host-header (DNS-rebinding) and CORS protection configured — this was a real gap found and fixed during development (see [`CHANGELOG.md`](CHANGELOG.md)), not something assumed safe by default.

**Not in place — known, not hidden:**
- `.env.example` ships placeholder secrets (`dev-secret-change-me`, `change-me-in-production`, `postgres`/`postgres`). **These must be changed before any deployment beyond local development.** They are intentionally weak defaults for frictionless local setup, not a security posture.
- The MLflow tracking server and registry have no authentication at all — anyone who can reach port `5000` can read, write, or delete registered models. Fine on an isolated `docker compose` network on your own machine; not fine exposed to any untrusted network.
- Postgres and Redis run with default/no credentials appropriate for local development only.
- The `/debug/drift` endpoint is intentionally unauthenticated — it exists to let you trigger a deterministic test event, not for production traffic.
- There is no rate limiting, no TLS termination, and no secrets manager integration. All of that is expected to be handled by whatever you put in front of this stack in a real deployment, not by the application code itself.

If you're evaluating this repository for its engineering practices rather than deploying it, the honest limitations above are as informative as the parts that are done well.
