# GEMINI.md — abi

Canonical instruction file is `AGENTS.md`. Read that first and follow its
conventions. If they disagree, `AGENTS.md` wins; if either conflicts with
executable source (`Cargo.toml`, `tools/check.sh`, `crates/`), trust the source.

The primary gate uses locked dependencies, requires the dependency-security
policy and sibling WDBX conformance tests, and builds release binaries.
`CARGO_TARGET_DIR` selects the build and benchmark output; CI uses a fresh target.

ABI pins stable Rust 1.99.1 (edition 2024); the declared Rust floor remains 1.99.
For the 2026-10-20 completion train, local tests are allowed and CI remains paused.

The workspace includes the separate `abi-control-plane` account/enrollment
service (18 local crates). The scheduler remains in-process; this service
does not add CLI/MCP commands. Its encrypted outbound relay enforces a separate
bounded tenant queue and fresh dispatch permits; plaintext remains on user
clients/nodes, with metadata-only control-plane persistence. See its crate README.
