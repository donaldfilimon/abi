# Program 1 Cross-Repository Conformance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:using-git-worktrees before edits, superpowers:executing-plans task-by-task, and superpowers:test-driven-development for all production changes. Treat each repository and each evidence layer independently.

**Goal:** Vendor the exact qualified Program 1 corpus into WDBX, `abbey`, `abbey-bot`, and `AbbeyBot`; make nightly Rust, stable Rust, and Swift independently decode and reject the same fixtures; and make AbbeyBot's local HTTP API fail closed when authentication is absent.

**Architecture:** ABI remains corpus authority. A deterministic vendoring tool copies only manifest-listed bytes and pins source repository, source revision, and aggregate digest in a small lock file. Each consumer implements a native bounded verifier/decoder appropriate to its toolchain and runs it in its existing strict gate. WDBX validates only the episode/evidence/claim/retention/canonicalization subset it semantically owns. No consumer imports ABI runtime types or executes a capability during conformance.

**Tech Stack:** Git worktrees, JSON Schema 2020-12, SHA-256, Rust nightly and stable, Swift 6.4 `Codable` plus `swift-crypto`, Python copy/digest guard scripts, existing repository gates.

**Prerequisite:** `2026-08-22-program-1-abbey-contract-corpus.md` is complete and its exact ABI commit/digest are known.

## Implementation status — 2026-08-22

- Tasks 1 through 6 are implemented and freshly green at the exact revisions
  recorded in `../evidence/2026-08-22-program-1-c1-matrix.md`.
- All four consumer trees pass ABI's read-only exact-byte vendoring check.
- Task 7's identity, digest comparison, full-local-gate, and evidence-boundary
  steps are complete in the closeout branch.
- ABI, WDBX, nightly Abbey, and stable Rust Abbey bot were pushed and merged by
  concurrent provider-side workflows. Their exact hosted outcomes, including
  the two Abbey-family failures found after merge, remain separate matrix rows.
- The Abbey test-environment repair passed its exact-head hosted macOS adjunct,
  merged through PR #91, and passed its post-merge main run. Stable bot
  Windows-byte repair, Swift AbbeyBot stack, and ABI evidence/Pages repair are
  being qualified separately and remain represented by their exact matrix
  states. No production deployment or participant-consented live Discord
  session was authorized or performed.

## Global Constraints

- Create clean `cursor/abbey-contracts-v1-20260822` worktrees from each repository's current `origin/main`. Never edit dirty shared primary checkouts.
- Vendor only `contracts/abbey/` bytes listed by the canonical manifest. Reject symlinks, external refs, extra files, missing files, digest mismatch, and changed line endings.
- Lock fields are `source_repository`, `source_revision`, `contract_major`, `contract_revision`, and `aggregate_digest`; no branch names or mutable URLs establish identity.
- A corpus mismatch disables authorization, execution, consent opening, and durable writes. Tests in this plan decode/validate only and perform no network, Discord, provider, audio, or WDBX mutation.
- Native consumers must not shell out to ABI or Python for their conformance result. Python vendoring guards are additive; Rust/Swift decoders are independently executable.
- Do not claim another repository qualified because ABI passed. Record each exact revision and gate separately.
- Production deployment and participant-consented live Discord remain separately authorized and evidenced.

---

## Task 1: Add deterministic vendoring protocol to ABI

**Files:**
- Create: `tools/vendor_abbey_contracts.py`
- Modify: `tools/tests/test_abbey_contracts.py`
- Modify: `contracts/abbey/compatibility.md`

**Interfaces:**
- `vendor(source: Path, destination: Path, source_revision: str, check: bool) -> VendorReport`.
- Writes `destination/abbey-contracts.lock.json` and `destination/corpus/` only under explicit `--write`.
- CLI: `python3 tools/vendor_abbey_contracts.py --source contracts/abbey --destination <dir> --source-revision <sha> [--write|--check]`.

- [ ] **Step 1: Write failing traversal and byte-equality tests**

Require refusal for dirty/unmanifested source, destination symlink, nonempty unmanaged destination, mutable/non-40-hex revision, extra destination file, and byte mismatch. A `--check` run never modifies mtimes or contents.

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest tools.tests.test_abbey_contracts.VendoringTests -v`

Expected: missing module/function failure.

- [ ] **Step 3: Implement atomic exact-byte vendoring**

Build into a mode-0700 temporary sibling, verify the copied tree using the canonical verifier, write the closed lock, then atomically rename. Refuse to replace a destination in `--check`; in `--write`, replace only an already managed destination after validating its current lock.

- [ ] **Step 4: Verify and commit**

Run focused vendoring tests plus the full ABI gate. Commit `feat(contracts): add deterministic corpus vendoring`.

## Task 2: WDBX native conformance for canonical memory contracts

**Files:**
- Create: isolated WDBX worktree/branch
- Create: `contracts/abbey/abbey-contracts.lock.json`
- Create: `contracts/abbey/corpus/**` from the exact ABI corpus
- Create: `crates/abi-wdbx/tests/abbey_contracts.rs`
- Modify: `crates/abi-wdbx/Cargo.toml`
- Modify: `AGENTS.md`
- Modify: `README.md`

**Interfaces:**
- `WdbxContractCorpus::open_from_repo() -> Result<Self, ContractFailure>` in the integration test/support module.
- Validates episode proposal, evidence, claim, tombstone/retention, and canonical-CBOR profile fixtures.
- Rejects adapter projections as canonical episodes.

- [ ] **Step 1: Create a clean WDBX worktree**

Fetch and branch from current public `origin/main`; run the existing WDBX gate before changes.

- [ ] **Step 2: Vendor the exact ABI corpus**

Run ABI's vendoring tool with the qualified ABI source revision. Verify the lock digest equals ABI's manifest aggregate.

- [ ] **Step 3: Write failing native Rust tests**

Tests independently read the lock/manifest, recompute SHA-256, compile local schemas without external resolution, and evaluate only fixtures whose schema family begins `episode/`. Add explicit failures for transport-JSON-as-commitment, adapter-supplied episode digest, missing deletion key, invalid retention, and projection-as-canonical episode.

Run: `cargo test -p abi-wdbx --test abbey_contracts -- --nocapture`

Expected: missing decoder/support failure.

- [ ] **Step 4: Implement the minimal WDBX subset decoder**

Use existing workspace `serde_json`, `jsonschema`, and `sha2`; add dev dependencies only where required. Keep errors to a closed enum and corpus-relative path. Do not create or modify a DurableStore during tests.

- [ ] **Step 5: Verify WDBX's complete gate**

Run:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets
cargo test --workspace
git diff --check
```

Expected: existing 558-plus tests and new contract tests pass with no clippy diagnostics.

- [ ] **Step 6: Commit**

Commit `feat(wdbx): qualify canonical Abbey episode contracts`.

## Task 3: `abbey` first-host nightly Rust conformance

**Files:**
- Create: isolated `abbey` worktree/branch
- Create: `contracts/abbey/abbey-contracts.lock.json`
- Create: `contracts/abbey/corpus/**`
- Create: `src/abbey_contracts.rs`
- Create: `tests/abbey_contracts.rs`
- Modify: `src/lib.rs`
- Modify: `Cargo.toml`
- Modify: `check.sh`
- Modify: `src/claims.rs`
- Regenerate: `AGENTS.md`, `CLAUDE.md`, `docs/claims.md`

**Interfaces:**
- `ContractCorpus::qualified() -> Result<QualifiedCorpus, ContractMismatch>`.
- `QualifiedCorpus::validate_fixture(&Fixture) -> FixtureResult`.
- `FederationProfile::from_corpus(result) -> DiagnosticOnly | ConsequentiallyQualified`.
- No runtime execution path is enabled by this task.

- [ ] **Step 1: Create a clean worktree and baseline**

Branch from current `origin/main`; confirm the sibling ABI/WDBX layout; run the focused existing workflow/claims tests before edits.

- [ ] **Step 2: Vendor the exact corpus**

Use the ABI tool. Store the exact ABI revision and aggregate digest.

- [ ] **Step 3: Write failing corpus and degradation tests**

Assert per-file/aggregate digest, schema compilation, valid/invalid/boundary/unknown/privacy/cancellation/degraded outcomes, tolerant extension preservation, strict authority unknown rejection, and mismatch-to-diagnostic-only. Assert mismatch cannot negotiate grants, approvals, consent open, tool execution, or memory write.

- [ ] **Step 4: Implement the bounded host decoder**

Reuse `serde_json`, `jsonschema`, and `sha2`. Keep it isolated from `app_core` execution types; Program 1 objects remain data contracts. Provide only validation/qualification state, never a parallel authorization kernel.

- [ ] **Step 5: Add an honest claim**

Add stable claim `program-1-abbey-contracts-host` as Current at C1 only after the strict gate passes. State data-only validation, pinned digest, and explicit absence of production federation/live authority.

- [ ] **Step 6: Run the full Abbey gate and commit**

Run `./check.sh` across default, WDBX, personal, and accel modes. Regenerate claim tables and commit `feat(abbey): qualify Program 1 contract corpus`.

## Task 4: `abbey-bot` stable-Rust conformance

**Files:**
- Create: isolated `abbey-bot` worktree/branch from current `origin/main`
- Create: `contracts/abbey/abbey-contracts.lock.json`
- Create: `contracts/abbey/corpus/**`
- Create: `src/abbey_contracts.rs`
- Modify: `src/main.rs`
- Modify: `Cargo.toml`
- Modify: `scripts/check-abbey-contracts.py`
- Create: `scripts/test-check-abbey-contracts.py`
- Modify: `check.sh`
- Modify: `tasks/goals.md`

**Interfaces:**
- Stable-Rust `ContractCorpus::verify_embedded() -> Result<QualifiedDigest, ContractError>`.
- Python guard verifies vendored byte equality and privacy taxonomy; Rust independently decodes fixtures.
- This task does not wire contracts into live Discord command execution.

- [ ] **Step 1: Baseline the isolated worktree**

Do not use the moved/shared `/Users/donaldfilimon/dev/active/abbey-bot` branch directly. Fetch `origin/main`, create a feature worktree, and run the current focused privacy/WDBX scripts.

- [ ] **Step 2: Write failing vendoring guard tests**

Model `scripts/test-check-wdbx-conformance.py`: verify success, missing corpus, changed byte, extra file, lock mismatch, and privacy sentinel. Failure output may contain normalized relative paths and closed reason codes only.

- [ ] **Step 3: Vendor and verify exact bytes**

Use the ABI vendoring tool and make `check.sh` invoke `scripts/check-abbey-contracts.py` before Cargo.

- [ ] **Step 4: Write failing stable-Rust decoder tests**

Within `src/abbey_contracts.rs`, load fixtures with `include_bytes!` or repository-relative test paths, independently verify digests, validate every family, preserve tolerant extensions, reject authority unknowns, and exercise the complete synthetic operator-verification flow with `local_test` evidence.

- [ ] **Step 5: Implement without nightly features**

Use only stable Rust 1.97.1-compatible dependencies. Do not path-depend on ABI or WDBX. Errors never embed input JSON or validator values.

- [ ] **Step 6: Run the full strict gate**

Run: `./check.sh`

Expected: format, Python locks/syntax/privacy/WDBX/contract guards, clippy `-D warnings`, full locked test suite, and locked release build pass on the exact branch head.

- [ ] **Step 7: Update the goal ledger and commit**

Record only local C1 stable-Rust conformance; do not mark production or live Discord complete. Commit `feat(contracts): qualify stable Rust Abbey bot adapter`.

## Task 5: AbbeyBot Swift `Codable` conformance

**Files:**
- Create: isolated `AbbeyBot` worktree/branch from current `origin/main`
- Create: `Contracts/Abbey/abbey-contracts.lock.json`
- Create: `Contracts/Abbey/corpus/**`
- Create: `Sources/AbbeyCore/Contracts/AbbeyContractCorpus.swift`
- Create: `Sources/AbbeyCore/Contracts/AbbeyContractModels.swift`
- Create: `Tests/AbbeyCoreTests/AbbeyContractCorpusTests.swift`
- Modify: `Package.swift`
- Modify: `Scripts/check-static-security.sh`
- Modify: `AGENTS.md`

**Interfaces:**
- `AbbeyContractCorpus.verify(at:) throws -> AbbeyQualifiedCorpus`.
- Closed `Codable` enums for evidence, consent state, risk, learning state, response terminal state, claim class/state/level, and reason code.
- `AbbeyQualifiedCorpus.decodeFixture(_:) throws -> FixtureDisposition`.

- [ ] **Step 1: Create a clean Swift worktree and baseline**

Preserve the shared dirty checkout. Use Xcode toolchain wrappers and run the focused `AbbeyCoreTests` baseline.

- [ ] **Step 2: Vendor the exact corpus**

Use ABI's tool to populate `Contracts/Abbey`. Add resources to the `AbbeyCore` target without code generation.

- [ ] **Step 3: Write failing Swift digest and `Codable` tests**

Use `Crypto.SHA256` to independently verify bytes. Tests decode valid/boundary fixtures, reject invalid/authority-unknown/over-bound/stale-consent/mismatch fixtures, preserve tolerant extensions, and prove synthetic reports contain no content fields. Verify lock and manifest using native Swift, not a subprocess.

- [ ] **Step 4: Implement bounded Swift models and loader**

Use explicit `CodingKeys`, manual enum decoding, file-size checks before `Data` retention, normalized relative paths, and closed sanitized errors. Do not synthesize protobuf/generated bindings.

- [ ] **Step 5: Run focused tests**

Run via the repository's Xcode-safe wrapper/build path:

```bash
unset TOOLCHAINS
bash Scripts/run-smoke.sh
```

Expected: `AbbeyCoreTests` and the desktop graph pass.

- [ ] **Step 6: Commit**

Commit `feat(contracts): qualify Swift Abbey contract corpus`.

## Task 6: Make AbbeyBot HTTP API fail closed without authentication

**Files:**
- Modify: `Sources/AbbeyServer/Routes/AbbeyAPIAuth.swift`
- Modify: route registration under `Sources/AbbeyServer/Routes/`
- Modify: `Tests/AbbeyServerTests/AbbeyServerAuthTests.swift`
- Modify: `Scripts/run-server-smoke.sh`
- Modify: `Scripts/run-cli-smoke.sh` only if it calls HTTP API routes
- Modify: `.env.example`
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- `requireConfiguredOperatorToken(_:)` refuses when `ABBEY_API_TOKEN` is absent/blank.
- Optional local bypass exists only in `.testing` or explicit `ABBEY_DRY_RUN=1`, never ordinary production/development HTTP.
- Health/static dashboard may remain public; `/api/*` state, sync, ingest, catalog metadata capable of exposing tenant state, and mutations require authentication.

- [ ] **Step 1: Write failing unset-token HTTP tests**

Add actual `VaporTesting` requests with no token for representative GET list/state/status and POST ingest/state endpoints. Expect `.serviceUnavailable` for missing server configuration, `.unauthorized` for missing/wrong caller token when configured, and success only with correct constant-time compared token. Add a test that an explicit bypass is rejected outside `.testing`/dry-run.

- [ ] **Step 2: Verify RED**

Run the repository's server-test filter. Expected: current `requireTokenIfConfigured` returns success with an unset token.

- [ ] **Step 3: Implement fail-closed auth and route coverage**

Replace fail-open operator auth with required configured token semantics. Centralize the protected route group so a new `/api` route cannot silently omit the middleware/helper. Keep provider-native signed webhooks on their signature policy; unsigned generic webhooks remain closed.

- [ ] **Step 4: Update smokes with ephemeral credentials**

Generate a fixed synthetic smoke token inside the isolated smoke environment, pass it only through process environment/headers, and ensure it never appears in output. Do not write it to `.env` or reports.

- [ ] **Step 5: Verify focused security and server gates**

Run:

```bash
unset TOOLCHAINS
bash Scripts/check-static-security.sh
bash Scripts/run-server-smoke.sh
```

Expected: auth tests and real server smoke pass; no secret echo.

- [ ] **Step 6: Run AbbeyBot's complete release gate**

Run: `bash Scripts/verify-all.sh`

Expected: static/security, package graphs, dashboard, all Swift tests, desktop, server, and CLI smokes pass. Snapshot/Linux checks retain their separately documented evidence boundaries unless explicitly run.

- [ ] **Step 7: Commit**

Commit `fix(server): fail closed when Abbey API auth is unset`.

## Task 7: Cross-repository digest matrix and closeout

**Files:**
- Modify: ABI `docs/superpowers/specs/2026-08-22-federation-reconciliation-and-abbey-contracts.md`
- Create: ABI `docs/superpowers/evidence/2026-08-22-program-1-c1-matrix.md`
- Modify: this plan with exact evidence links/status

- [ ] **Step 1: Record exact identities**

For ABI, WDBX, `abbey`, `abbey-bot`, and `AbbeyBot`, record repository, branch/head SHA, lock source SHA, aggregate digest, toolchain, focused test command/result, full gate command/result, and hosted CI state. No raw logs containing private paths or content enter the matrix.

- [ ] **Step 2: Compare vendored corpora byte-for-byte**

Use the ABI verifier against each destination plus each native verifier. All five aggregate digests and every file digest must match.

- [ ] **Step 3: Run every full gate fresh at final heads**

Run:

```bash
# ABI
./tools/check.sh
# WDBX
cargo fmt --all --check && cargo clippy --workspace --all-targets && cargo test --workspace
# abbey
./check.sh
# abbey-bot
./check.sh
# AbbeyBot
bash Scripts/verify-all.sh
```

Run each in its own isolated worktree and retain exact exit status/test totals.

- [ ] **Step 4: Finish branches and observe hosted checks**

Use `superpowers:finishing-a-development-branch` per repository. Open reviewable PRs, observe exact-head jobs, merge only after required checks/review, and observe default-branch runs separately.

- [ ] **Step 5: State the evidence boundary**

The matrix may conclude C1 source/contract conformance only. State explicitly:

- production federation/deployment was not authorized or performed by this plan;
- no real grant, approval, Discord action, WDBX episode write, provider call, or consent epoch occurred;
- installed-artifact qualification is separate;
- participant-consented live Discord verification is separate and remains pending unless Donald later authorizes the exact session.

- [ ] **Step 6: Commit closeout**

Commit `docs(contracts): record cross-repository C1 conformance` after every cited gate is fresh and green.
