# ABI Agent System

This document mirrors the repository-wide agent specification. Refer to the
root `AGENTS.md` for the authoritative reference. This copy exists under the
`docs/` tree so contributors can generate documentation bundles without
cross-referencing the root of the repository.

Key points:
- Deterministic controller orchestrating feature tools and provider connectors.
- Strict input/output schemas validated at the controller boundaries.
- Retry, rate limiting, and observability baked into the agent runtime.
- Mock connector is the default in CI to ensure reproducible tests.
