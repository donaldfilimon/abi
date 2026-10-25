# GEMINI.md — abi

Canonical instruction file is `AGENTS.md`. Read that first and follow its
conventions. If they disagree, `AGENTS.md` wins; if either conflicts with
executable source (`Cargo.toml`, `tools/check.sh`, `crates/`), trust the source.

The primary gate uses locked dependencies, requires the dependency-security
policy and sibling WDBX conformance tests, and builds release binaries.
`CARGO_TARGET_DIR` selects the build and benchmark output; CI uses a fresh target.

ABI pins stable Rust 1.99.1 (edition 2024); the declared Rust floor remains 1.99.
For the 2026-10-20 completion train, local tests are allowed and CI remains paused.

## Local main readiness (2026-10-25)

Follow [docs/superpowers/plans/2026-10-25-local-main-readiness.md](docs/superpowers/plans/2026-10-25-local-main-readiness.md) for this user-approved local integration task.
Use one manual task branch/worktree per independent task; preserve existing
dirty, untracked, ignored and unpushed work and app-managed worktree lifecycles.
Merge owned changes into local `main` only after writer release, recovery custody
and the documented repository gates pass on the candidate revision. Keep failed
or skipped gates explicit. This task's hold takes precedence over older guidance
requiring immediate merge or worktree deletion. Leave owner-held work in place;
remote publication and deployment require separate scope.
