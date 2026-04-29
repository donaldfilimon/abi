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

Note
- This readme is a safe minimal landing page. If you need full developer onboarding, refer to AGENTS.md and CODEBASE_REVIEW.md for deeper operational guidance.
