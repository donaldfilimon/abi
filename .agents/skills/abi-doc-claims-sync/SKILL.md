---
name: abi-doc-claims-sync
description: Synchronize ABI Markdown and docs claims with executable behavior. Use before editing README, AGENTS.md, CLAUDE.md, GEMINI.md, docs/**/*.mdx, walkthroughs, changelogs, or any old ABI Markdown that mentions WDBX, MCP, GPU, clustering, compression, FHE, agents, or benchmarks.
---

# ABI Doc Claims Sync

Use this skill to update ABI documentation without drifting from source, tests, or the external-claims policy.

## Workflow

1. Read `references/claim-boundaries.md`.
2. Inspect the specific docs being changed plus the source/test paths they claim to describe.
3. Treat `build.zig`, `src/`, tests, and runnable scripts as authoritative over prose.
4. Keep `AGENTS.md`, `CLAUDE.md`, and `GEMINI.md` synchronized when changing durable conventions: commands, MCP tools, feature flags, build gates, Zig idioms, generated-file rules, listener/auth behavior.
5. Rewrite unproven claims as Current/Partial/Proposed language, or remove them.
6. Run `.agents/skills/docs-validate/validate.sh` after editing `docs/`.

## Claim Checks

- Capability exists only when source plus tests or runnable behavior prove it.
- Benchmark numbers require fresh reproducible benchmark evidence.
- Loopback bearer tokens are hardening, not TLS/authz/rate-limit production security.
- WDBX cluster RPC is real TCP consensus transport, but production distributed deployment and sharding remain unproven.
- Compression and FHE demos are reference-scoped unless audited production parameters and artifacts exist.

Use `references/claim-boundaries.md` as the quick denylist and safe wording source.
