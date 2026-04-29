OpenCode ABI Repo

- This repository hosts the Zig 0.17-dev based ABI runtime, parity tests, and tooling to exercise AI/SDK integrations.

Getting started
- Build CLI: ./build.sh cli or zig build cli
- Build MCP: ./build.sh mcp or zig build mcp
- Run parity checks: zig build check-parity
- Run focused tests: zig build test --summary all -- -test-filter "auth|token|persistence|wal|search"

Parity gating
- Parity checks are environment-aware. If ABI_JWT_SECRET is not set locally, many auth-related tests will be skipped to allow fast feedback on non-auth paths.
- In CI, ABI_JWT_SECRET can be provided to run full parity across auth paths.

### Documentation Overview
- ONBOARDING.md — Quick-start onboarding guide.
- CODEBASE_REVIEW.md — Architecture notes and entrypoints.
- GLOSSARY.md — Repo-wide terms and definitions.
- ONBOARDING_INDEX.md — Central onboarding navigator.
- SUMMARY.md — Documentation at a glance.
- CONTRIBUTING.md — PR workflow and contribution guidelines.
- AGENTS.md — Onboarding guidance for agents and automation helpers.
- README.md — Quick overview and onboarding pointers.
- Doc validation CI workflow (docs) — CI cross-link checks.

### Note
This readme is a safe minimal landing page. If you need full developer onboarding, refer to the above documents for deeper operational guidance.
Onboarding quickstart: See ONBOARDING.md for a concise one-page guide to bootstrapping a new session.


Glossary: See GLOSSARY.md for repo-wide terms.
