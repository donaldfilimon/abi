# Program 2 Capability Authorization Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qualify a local-only, deny-by-default Program 2 authorization kernel and its contract-major-v2 wire surface without changing any Abbey v1 contract byte or granting production authority.

**Architecture:** Program 1 remains the sole owner of wire schemas and fixtures. A new `abi-capability` crate consumes typed equivalents and evaluates a request-scoped intersection of package, grant, platform, tenant, guild, and safety constraints. Every terminal or advancing decision is emitted through a fallible bounded redacted audit sink before a recording-only actuator may observe the call. The actuator stores digests and typed outcomes only; it has no network, credential, Discord, provider, WDBX, or production adapter binding.

**Tech Stack:** Rust 2024, Serde, JSON Schema draft 2020-12, SHA-256, existing `abi-agent-runtime::CancellationToken`, Python corpus tooling, Cargo tests, and the repository `./tools/check.sh` gate.

**Spec:** `docs/superpowers/specs/2026-08-22-spec-capability-authorization-kernel.md`

## Global Constraints

- Preserve every byte under `contracts/abbey/v1/`; contract-major v2 is additive and uses new schema identifiers.
- Use `sha256:<64 lowercase hex>` on the wire and fixed `[u8; 32]` digests internally.
- Keep policy decisions closed: `allow`, `approval_required`, `deny`, and `pause`, each paired with a closed reason code.
- Refuse missing, mismatched, expired, suspended, revoked, prohibited, stale-policy, stale-version, stale-digest, cross-scope, self-approved A2+, and unauditable requests before recording an effect.
- Make time, cancellation, platform facts, approvals, grants, and audit persistence explicit inputs; do not consult ambient time or external state.
- Store no raw parameters, output, credential, message content, transcript, participant identity, or audio in audit, recording, receipt, diagnostic, or replay material.
- Do not add a production adapter, Discord mutation, provider call, real credential, canonical WDBX write, push, merge, or live-validation claim.

## Executed Slice and Explicit Deferrals

The implemented slice covers the closed v2 authority-envelope family consumed
by the kernel, exact request-scoped authorization, recording-only postcondition
evaluation, redacted bounded audit/receipt material, opaque tenant credential
references, and deterministic local replay. It deliberately defers package
source-schema compilation, full §2.2 package field breadth, durable approval
consumption and restart recovery, Guild Constitution compilation, independent
mid-flight fact refresh, all 17 per-stage audit records, compensation execution,
`abi-agent-host` projection, Abbey database migrations, and every production,
Discord, provider, WDBX, shadow, canary, or live-validation binding. Those rows
remain C0 and must not inherit this slice's local C1/C2 evidence.

---

### Task 1: Reconcile the approved v2 authority boundary

**Files:**
- Modify: `docs/superpowers/specs/2026-08-22-spec-capability-authorization-kernel.md`
- Test: documentation inspection plus contract/runtime tests in later tasks

- [ ] Mark Program 1's v1 corpus as existing and qualified while retaining the Program 2 behavioral claim at C0 until this plan's evidence is complete.
- [ ] Replace the stale “carry Rust types until the package exists” statement with an explicit v2 dependency and preserve v1 as immutable compatibility evidence.
- [ ] Resolve the approved questions: C0-C7 is normative, Program 2 is recording-only, tenant means organization/deployment boundary, and `Prohibited` is representable but ungrantable; leave approval UI ownership deferred.
- [ ] Correct the stage count to 17 stages (0 through 16), make recording execute simulated postcondition/receipt stages, and make rollback return to `DenyAllPolicy`.

### Task 2: Add contract-major-v2 closed wire schemas and fixtures

**Files:**
- Create: `contracts/abbey/v2/schemas/common/definitions.schema.json`
- Create: `contracts/abbey/v2/schemas/capability/package.schema.json`
- Create: `contracts/abbey/v2/schemas/authorization/grant.schema.json`
- Create: `contracts/abbey/v2/schemas/authorization/approval.schema.json`
- Create: `contracts/abbey/v2/schemas/authorization/policy-decision.schema.json`
- Create: `contracts/abbey/v2/schemas/authorization/audit-record.schema.json`
- Create: `contracts/abbey/v2/schemas/authorization/credential-ref.schema.json`
- Create: `contracts/abbey/v2/schemas/receipt/outcome-receipt.schema.json`
- Create: focused valid, invalid, boundary, and privacy fixtures under `contracts/abbey/v2/fixtures/`
- Modify: `contracts/abbey/manifest.json`
- Modify: `contracts/abbey/README.md`
- Modify: `contracts/abbey/compatibility.md`
- Modify: `tools/abbey_contracts.py`
- Modify: `tools/tests/test_abbey_contracts.py`
- Modify: `crates/abi-contracts/src/lib.rs`
- Modify: `crates/abi-contracts/tests/corpus.rs`

- [ ] Write failing tests proving a v2 corpus is accepted, v1 remains inventoried, v2 fixtures are validated by their declared schema IDs, and unsupported major values fail closed.
- [ ] Add closed schemas for the exact fields consumed by the kernel: scope, principal, package/grant/version/policy bindings, revocation, approval level, closed decision/reason, credential reference, bounded redacted audit, and receipt.
- [ ] Add synthetic fixtures covering a valid A2 authorization, expired grant, digest mismatch, prohibited package, A2 self-approval, cross-tenant credential reference, and forbidden raw receipt/audit content.
- [ ] Generalize corpus discovery and schema registration from hard-coded v1 paths to the manifest's supported major directories, while retaining strict duplicate-key, size, privacy, unknown-field, and semantic validation.
- [ ] Regenerate and verify the manifest, then prove `git diff --exit-code origin/main -- contracts/abbey/v1`.

### Task 3: Build the typed registry and deterministic authorization kernel

**Files:**
- Create: `crates/abi-capability/Cargo.toml`
- Create: `crates/abi-capability/src/lib.rs`
- Create: `crates/abi-capability/src/types.rs`
- Create: `crates/abi-capability/src/registry.rs`
- Create: `crates/abi-capability/src/audit.rs`
- Create: `crates/abi-capability/src/recording.rs`
- Create: `crates/abi-capability/src/kernel.rs`
- Create: `crates/abi-capability/tests/kernel.rs`
- Modify: `Cargo.toml`

- [ ] Write a failing integration test whose ungranted request returns `deny/no_matching_grant`, writes a redacted audit record, and leaves the recording actuator empty.
- [ ] Implement closed types and a registry that rejects duplicate packages, prohibited registrations for execution, platform writes without postconditions, and package digest drift.
- [ ] Implement exact grant matching across recipient, organization/deployment/tenant/platform scope, capability id/version/package digest, issue window, revocation epoch/state, risk ceiling, and all policy versions.
- [ ] Implement the total approval ladder `A0 < A1 < A2 < A3 < A4 < A5`, with deterministic package/grant/risk/regime/safety maxima, irreversible `A4`, A2+ separation of duties, digest binding, expiry, and single-use decision IDs within the request-scoped replay state.
- [ ] Implement a request-scoped kernel using an injected clock and cancellation token; it emits only fixed decisions/reasons and has no fallback allow path.

### Task 4: Add fail-closed audit, recording actuation, postconditions, and receipts

**Files:**
- Modify: `crates/abi-capability/src/audit.rs`
- Modify: `crates/abi-capability/src/recording.rs`
- Modify: `crates/abi-capability/src/kernel.rs`
- Modify: `crates/abi-capability/tests/kernel.rs`

- [ ] Write failing tests for audit write failure, record-count/byte ceilings, cancellation before actuation, changed platform permission facts, failed postconditions, and receipt redaction.
- [ ] Implement `AuditSink::record -> Result` with fixed record and serialized-byte ceilings; make every sink error terminal before the actuator.
- [ ] Implement `RecordingActuator` so it records only request/capability/scope/parameter/effect digests and evaluates typed postconditions against injected fixture facts.
- [ ] Re-check cancellation, revocation epoch, policy versions, and platform facts immediately before recording the simulated effect.
- [ ] Return a bounded redacted receipt with closed outcome, postcondition, cancellation, and compensation states and no raw request or response material.

### Task 5: Qualify privacy and deterministic replay

**Files:**
- Modify: `crates/abi-capability/tests/kernel.rs`
- Create: `crates/abi-capability/tests/privacy_replay.rs`

- [ ] Replay the same frozen request, registry, grant, approval, clock, platform facts, and cancellation schedule twice and assert byte-identical decisions, audits, recordings, and receipts.
- [ ] Assert authorization changes deterministically for expiry, revocation, capability version, package digest, policy version, tenant, deployment, guild, resource, subject, approval level, approval identity, and cancellation mismatches.
- [ ] Seed raw parameter, message, transcript, audio, identifier, output, and credential canaries and assert none appears in serialized audit, recording, receipt, or error material.
- [ ] Assert a credential reference resolves only within its exact tenant binding and never falls back.

### Task 6: Verify and commit the bounded slice

**Files:**
- Review: every file changed by Tasks 1 through 5

- [ ] Run Python contract tests and the contract verifier.
- [ ] Run `cargo fmt --all -- --check`, `cargo clippy -p abi-capability -p abi-contracts --all-targets -- -D warnings`, and focused crate tests.
- [ ] Run the exact final repository gate `./tools/check.sh` and retain its exit status and stage evidence.
- [ ] Inspect the complete diff, prove v1 byte identity again, check for raw-content canaries and forbidden adapter imports, and classify any residual acceptance rows honestly.
- [ ] Commit the completed local-only slice on `cursor/program-2-capability-kernel-20260822` without pushing or merging.
