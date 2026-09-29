# GEMINI.md — abi

Canonical instruction file is `AGENTS.md`. Read that first and follow its
conventions. If they disagree, `AGENTS.md` wins; if either conflicts with
executable source (`Cargo.toml`, `tools/check.sh`, `crates/`), trust the source.

The primary gate uses locked dependencies, requires the dependency-security
policy and sibling WDBX conformance tests, and builds release binaries.
`CARGO_TARGET_DIR` selects the build and benchmark output; CI uses a fresh target.
