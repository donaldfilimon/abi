# ABI

This repository hosts the Zig 0.17-dev ABI runtime, parity tests, protocol tooling, and AI/SDK integration surface.

## Getting Started

- Bootstrap toolchain: `tools/zigly --bootstrap`
- Build CLI: `./build.sh cli` (use `zig build cli` on Linux/older macOS)
- Build MCP: `./build.sh mcp` (use `zig build mcp` on Linux/older macOS)
- Run parity checks: `./build.sh check-parity`
- Run focused tests: `./build.sh test --summary all -- --test-filter "auth|token|persistence|wal|search"`

The package root remains `src/root.zig`, exposed to consumers as `@import("abi")`. Root-level public wiring is grouped under `src/public/` so the public surface can stay stable while internals continue to modularize.

## Parity Gating

- Parity checks are environment-aware. If ABI_JWT_SECRET is not set locally, many auth-related tests will be skipped to allow fast feedback on non-auth paths.
- In CI, ABI_JWT_SECRET can be provided to run full parity across auth paths.

## Documentation Overview

- `ONBOARDING.md` - quick-start onboarding guide.
- `ONBOARDING_INDEX.md` - central onboarding navigator.
- `SUMMARY.md` - documentation at a glance.
- `CODEBASE_REVIEW.md` - architecture notes and entrypoints.
- `GLOSSARY.md` - repo-wide terms and definitions.
- `AGENTS.md` - onboarding guidance for agents and automation helpers.
- `docs/` - GitHub Pages documentation, specs, reviews, and archived plans.

Doc validation is handled by `.github/workflows/doc-validation.yml` and `scripts/verify-docs.sh`.
