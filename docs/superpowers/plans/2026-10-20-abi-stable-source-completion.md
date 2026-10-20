# ABI Stable Source Completion Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for independent implementation tasks and reviews, and superpowers:dispatching-parallel-agents only for disjoint ownership. Continue task-by-task without asking for routine permission.

**Goal:** Qualify ABI and the existing WDBX migration on exact stable Rust 1.99.1, then complete the ABI-owned recording governance, durable host interfaces and evidence joins defined by Donald's directive.

**Architecture:** Preserve existing crates and public contracts. ABI owns canonical capability evaluation and bounded host integration; Abbey owns durable approval/execution transactions; WDBX owns canonical memory. Add versioned internal interfaces without enabling new live effects.

**Tech Stack:** Rust 2024, resolver 3, stable Rust 1.99.1 for ABI/WDBX, existing Serde/JSON Schema/SHA-256, Abbey SQLite, WDBX scratch stores, existing Swift/Metal bridges.

**Spec:** [stable source completion design](../specs/2026-10-20-abi-stable-source-completion-design.md), Donald's 2026-10-20 directive, and the source authorities listed there.

## Global constraints

- Local tests are authorized. CI remains paused. Do not push, dispatch, publish, deploy, download model/data resources, train, send live messages or modify live stores.
- Work in canonical main checkouts. Preserve all pre-existing edits and outgoing commits. Leave reviewable diffs; no new commits are required by this plan.
- Use ABI's tools/cargo.sh. Preserve compiler/linker/DYLD protections and atomic native-library replacement. Hand-invoked Rust tests receive stdin EOF.
- Keep the five required WDBX path dependencies and crate identities. No vendoring, git-dependency substitutes, lockfile churn, bootstrap escape hatches or disabled production features.
- Keep CLI 14, MCP 12, help/completion/output contracts and all existing signed/canonical bytes unchanged. Add new internal types instead of changing closed wire formats.
- Use only recording actuators, scripted providers, generated tiny model fixtures and owned scratch stores. Existing live adapters are not implicitly enabled.
- Every completion claim names source manifests, command, exit status, compiler, target, features and unavailable paths. Historical totals do not qualify a changed tree.
- Each worker owns its assigned paths, is not alone in the codebase, preserves others' changes, and does not spawn its own reviewer or run broad gates concurrently with writers.

## Review focus

- A hostile PATH or explicit compiler override must not turn a stable qualification into a nightly or mixed-toolchain result.
- A formatted working sibling must not be confused with the exact immutable WDBX revision selected by CI.
- A caller-controlled effect declaration, model output, retrieved memory, infallible audit callback or replay set must never become execution authority.
- A crash or persistence failure after possible dispatch must remain unresolved across process restart and must not replay an effect.
- Unauthorized recall must not leak through counts, errors, caches, diagnostics or prompts; generated records must retain inferred provenance.

## Task 1: Preserve baseline and record the execution scope

**Ownership:** controller; external evidence directory plus this plan's ignored SDD ledger.
**Consumes:** three current checkouts and the user decisions in the design.
**Produces:** per-file source manifests, exact identities, task/review ledger and restrictions.

- [x] Capture ABI/WDBX/Abbey HEAD, remotes, branches, worktrees, dirty/untracked file hashes, compiler/tool versions and relevant override names without secrets.
- [x] Keep the existing hook/architecture changes. Record their 10 focused tests as prior evidence, not final stable qualification.
- [x] Register the matrix from the design. Keep CI/publication, production models/training, remote deployment and platform acceptance separate.
- [x] Record that the earlier nightly baseline passed after exposing installed cargo-audit on PATH; retain the first missing-scanner failure. No repeat baseline is needed solely to manufacture another green.
- [x] Establish isolated build output; never reuse mapped MCP libraries or open live ABI/Abbey state.
- [x] Review task overlaps before dispatch and record ownership/ordering in the ledger.

## Task 2: Migrate compiler selection and executable build policy

**Ownership:** rust-toolchain.toml, tools/cargo.sh, tools/check.sh, .github/workflows/ci.yml, tools/tests/test_ci_contract.py, a new tools/tests/test_toolchain.py, and minimal directly dependent toolchain policy code only.
**Consumes:** WDBX candidate 08a69b3f22db61093e06527b3308b10589687022 and the installed stable toolchain.
**Produces:** exact stable default, truthful compiler diagnostics, local workflow diff and focused regression evidence.

- [x] Write executable fake-tool tests for PATH precedence, matched Cargo/rustc/rustdoc/rustfmt/clippy selection, invocation from outside the checkout, preserved linker and DYLD behavior, and override qualification.
- [x] Pin ABI to 1.99.1 with rustfmt, clippy and rust-src; retain edition/resolver and rust-version 1.99.
- [x] Resolve the wrapper's toolchain in repository context without changing Cargo's caller working directory. Retain explicit developer RUSTUP_TOOLCHAIN overrides; stable release qualification rejects any effective release other than 1.99.1.
- [x] Ensure the gate identifies the effective compiler, fails on compiler substitution/bootstrap rather than hiding it, and retains all existing substantive stages.
- [x] Change the CI install command and matching behavioral test together to exact 1.99.1 and one comma-separated component argument. Preserve trusted self-hosted event checks, disabled persisted credentials and isolated target cleanup.
- [x] Stage the immutable WDBX candidate in the local CI diff as pending qualification. Do not claim acceptance until Task 3 and the paired final gate pass against that exact revision.
- [x] Inventory remaining feature gates, std::simd, -Z and RUSTC_BOOTSTRAP across production/build paths; fix demonstrated stable compilation failures without suppressing required code.
- [x] Run focused Python tests and a locked stable workspace/all-target check in isolated output. Report actual compiler path/full version/target and any lockfile change reason.
- [x] Self-review the owned diff; return files, tests, failures and unresolved risks. Do not commit, push or dispatch CI.

## Task 3: Qualify the existing CPU migration and exact dependency

**Ownership:** a new ABI integration test under crates/abi-gpu/tests/ and external immutable-pair evidence; do not change WDBX production source.
**Consumes:** existing abi-compute public CPU/selection contracts and frozen pre-migration reference behavior.
**Produces:** independent numerical/fallback evidence and an exact WDBX candidate gate receipt.

- [x] Inspect pre-migration and candidate source and existing tests. Qualification of existing behavior does not require inventing a failing test.
- [x] Add independently meaningful numeric cases: mismatched dimensions, empty and zero vectors, nonfinite classification, signed zero when observable, ragged tails and cancellation-sensitive reduction inputs, stable top-k ties/zero/oversized limits, and supported CPU fallback.
- [x] Keep cancellation tests at APIs that actually expose cancellation. Do not invent cancellation for synchronous CPU dot or claim unexercised x86 lane/runtime paths on ARM64.
- [x] Use exact known expectations and independently derived frozen cases; do not compare an implementation only to itself or silently change numeric contracts.
- [x] Run focused stable tests through the ABI wrapper with stdin EOF and an isolated target. Preserve hashes/encodings/goldens.
- [x] Export exact WDBX 08a69b3 into an owned temporary qualification checkout, preserving the canonical sibling's formatting diff. Run its own inspected gate with 1.99.1 tools and system cc first on PATH.
- [x] If exact candidate formatting or another gate fails, retain the failure and leave its pin pending. Prepare only the necessary owner-correct repair; do not silently qualify a dirty successor as the published commit.
- [x] Return exact manifests, commands/exit statuses and realistic hardware/feature scope. Coordinate broad gate scheduling with the controller.

## Task 4: Complete capability compilation and pure evaluation

**Ownership:** crates/abi-capability/src/ and its tests only; no host, manifests, wire corpus or Abbey edits.
**Consumes:** Program 2 spec §§2,5–10, existing closed v2 types and recording behavior.
**Produces:** additive closed internal package source and pure authorization evaluation reusable by the host.

- [x] Read the executed-slice/deferral section. Preserve authorize_and_record and its existing tests as a compatible recording API.
- [x] Extract canonical authorization checks into a pure evaluate entry point returning a decision, bindings and reason; do not duplicate a weaker host authority.
- [x] Add closed internal package-source compilation covering the approved §2.2 fields, startup schema binding, version/digest identity, data classes, prerequisites, risk, effect, budgets, postconditions, reversibility/compensator and approval policy.
- [x] Compile Guild Constitution input as a deterministic narrowing intersection; reject unknown fields, widening, prohibited grants, invalid compensator linkage and uncontrolled external schema references.
- [x] Add exact approval preview/binding values for request, principal, full scope, package/grant/policy/schema and precondition commitments; retain A0–A5 hierarchy, TTL ceiling and separation of duties.
- [x] Test wrong principal/tenant/resource, stale policy/package/grant, expiry/replay/delegation, prohibited operations, changed input and deterministic narrowing. Preserve redaction/canary and frozen replay tests.
- [x] Add a durable-lifecycle interface module with fallible bounded stage audit, atomic prepare, terminal outcome and lookup operations. It declares contracts and redacted types, not storage or live execution.
- [x] Distinguish prepared, refused/pending and uncertain outcomes. A prepared handle binds exact authorization; an observational audit or caller-provided consumed-ID set cannot satisfy the governed path.
- [x] Run focused capability tests, lint and size checks; review the additive public API and compatibility. Coordinate interface signatures with Tasks 5–6 before their implementation.

## Task 5: Extend Abbey's existing durable lifecycle owner

**Ownership:** Abbey runtime migration/store modules and their focused tests; Abbey manifest only for the ABI capability dependency. No daemon live route changes.
**Consumes:** Task 4 lifecycle contract, current RuntimeStore schema 7 and existing approval/execution transaction semantics.
**Produces:** a recording-only adapter and additive migration backed by the same SQLite authority.

- [ ] Re-read Abbey instructions, migration ordering and current transaction/recovery tests. Do not resume its canceled Zig rewrite or change its compiler pin without its own qualification.
- [ ] Add migration 8 (or the actual next number after revalidation), using FK-bound governance extensions to existing approval/execution rows and append-only events. Keep migrations 1–7 and legacy methods intact.
- [ ] Refactor transaction-local helpers so governed preparation atomically validates/consumes the exact approval and commits intent, full binding digests and pre-effect audit in the same immediate transaction.
- [ ] Support no-human-approval A0 only under a current exact grant, through the governed admission method; absence of approval never substitutes for a missing grant.
- [ ] Enforce concurrent single-use consumption with two independent connections and actual transactions; duplicate acknowledgement never returns a dispatch-capable second handle.
- [ ] Record richer governed outcomes without weakening old SQL checks. Unknown effects retain an Interrupted base state and explicit unknown reason. Unverifiable postconditions never become verified success.
- [ ] Make corrupt, future-schema, unreadable and inconsistent state fail startup/admission. Never create fresh replay history as a fallback.
- [ ] Exercise child-process crashes after prepare and after possible recording effect, restart/reopen, replay refusal and fresh-call recovery using owner-only scratch state.
- [ ] Retain the existing approval TTL ceiling, bounded raw-free audit and cancellation semantics. Do not migrate a user database or enable federation execution methods.
- [ ] Run focused Abbey tests through a proven pinned-toolchain environment with stdin EOF. Reserve its full check.sh for the final paired source snapshot.

## Task 6: Add the governed recording host

**Ownership:** crates/abi-agent-host/ plus its capability dependency and tests; coordinate root manifest additions with the controller.
**Consumes:** Task 4 canonical evaluation/lifecycle interfaces and Task 5 durable adapter.
**Produces:** an additive GovernedHost API sharing the bounded driver and distinct governance reports.

- [ ] Keep ExecutionPolicy and runtime AuditSink unchanged. Keep legacy AgentHost/report types behavior-compatible; do not reinterpret RequireConfirmation in the legacy path.
- [ ] Add trusted context/fact-provider and governed report/options types. Startup-owned tool/package registration fixes schema, authority and effects; model fields cannot select credentials or permissions.
- [ ] Share the existing continuation driver through internal hooks rather than copying its bounded loop. Retain existing budgets, result correlation, forged-result rejection and post-terminal behavior.
- [ ] Implement the exact 17-stage trace from Program 2 §7. Audit every reached terminal/advancing decision with fallible bounded redacted records.
- [ ] Return PendingApproval with an exact preview and stop before dispatch. A fresh explicit resubmission rechecks all bindings, facts, budgets, time and cancellation.
- [ ] Acquire facts outside pure policy; stop on unavailable/stale/changed facts. Reauthorize immediately before durable preparation and check cancellation again before the recording effect.
- [ ] Restrict this constructor to recording execution. A generic production ToolExecutor is not enabled by the new API.
- [ ] Verify typed postconditions after the effect, record verified/failed/unverifiable/unknown outcomes, and preserve uncertain accounting until observed completion or teardown.
- [ ] Admit compensation separately under exact scope and valid approval/preauthorization; give it its own intent/audit/postconditions/outcome. Stop on first compensation failure; never auto-compensate Interrupted/Unknown.
- [ ] Run meaningful failure/race tests for each authority boundary, stage audit failure, deadlines, cancellation around prepare/dispatch, changed facts, lost outcome writes, postcondition failure, compensation failure and raw-content canaries.

## Task 7: Join authorized evidence and local runtime composition

**Ownership:** SEA/application composition and focused integration tests, plus gateway owner adapters where necessary; keep abi-ai pure.
**Consumes:** governed host receipts, existing VersionedStore transaction and canonical episode gates.
**Produces:** a local end-to-end recording/scratch scenario with provenance and truthful persistence results.

- [ ] Read the canonical episode, WDBX north-star/multiway, September 29 completion and member-erasure specifications plus current substrate APIs before changing memory-facing code.
- [ ] Add a principal/scoped recall entry point that filters before scoring, counts, caches, errors and prompt packing. Existing generic recall remains an explicitly local compatibility API.
- [ ] Carry known/unknown provenance, source/version, observation time, epistemic/lifecycle/contradiction state, compatibility and outcome references. Preserve generated content as inferred and immutable signed source bytes.
- [ ] Keep the existing eight-signal scorer, stable-ID deduplication, candidate/prompt caps and rejection exclusion; retrieved content never supplies instructions or authority.
- [ ] Add internal persistence outcomes for disabled, open failure, committed, refused and unknown; map them explicitly onto unchanged CLI/MCP schemas.
- [ ] Reuse atomic record_vector_pair and canonical episode proposal admission. Retention/erasure hooks call approved owner contracts; do not introduce a second deletion mechanism or exercise live erasure.
- [ ] Trace normalization → authorized recall → persona/model selection → bounded generation → validation → proposed action → authorization → recording effect → observed result → selective persistence.
- [ ] Test scripted provider tool proposals separately from the generated abi-bigram-v1 CPU fixture. Do not claim the tiny model can generate structured calls or useful Abbey weights.
- [ ] Add a worker-control fixture only if this composition actually needs it; no listener/scheduler/remote worker is necessary for the selected local acceptance. Keep such deployment requirements in the external ledger.
- [ ] Test scope isolation, malicious retrieved text, provenance, atomic writes, refusal/unknown outcomes and causal/corpus byte compatibility with owned scratch stores.

## Task 8: Qualify source, artifacts, documentation and handoff

**Ownership:** controller and a fresh independent reviewer; active instruction/docs/task ledgers and packaging evidence.
**Consumes:** reviewed implementation tasks and fixed per-file source manifests.
**Produces:** six separate acceptance conclusions and precise outstanding external packets.

- [ ] Finish the pre-existing hook/architecture slice and preserve unrelated outgoing commits. Update active toolchain wording in AGENTS/CLAUDE/GEMINI, lessons, relevant skills, README and architecture prose; keep historical reports accurate.
- [ ] Run focused hooks/docs policy/shell checks and git diff --check. Run Mintlify with an available supported Node/npx installation; if infrastructure remains absent, record it as blocked rather than passing.
- [ ] Freeze source manifests before authoritative gates. Run ABI ./tools/check.sh with RUSTUP_TOOLCHAIN=1.99.1, ABI_WDBX_PATH=:memory: and a fresh CARGO_TARGET_DIR, with installed cargo-audit reachable and stdin EOF.
- [ ] Run the exact WDBX candidate's own gate and paired conformance under proven 1.99.1 selection. Distinguish its clean immutable receipt from canonical dirty-source results.
- [ ] Run Abbey's inspected full check.sh against the governed-owner changes, with proven compiler selection, isolated output, owned state and stdin EOF. Its compiler qualification is separately labeled.
- [ ] Exercise supported feature configurations; retain Metal numerical checks, deterministic inactive fallback and CUDA unavailability disclosures. No all-features shortcut proves unsupported hardware.
- [ ] Smoke freshly built CLI/MCP through mcp/launcher.sh with scratch stores and both required macOS libraries. Verify copied artifact help/goldens, loader paths/signatures, checksums and absence of release-strip abort warnings.
- [ ] Obtain fresh independent full-diff review for spec compliance, compatibility, authorization, replay/recovery, privacy and truthful claims. Route fixes to owning workers and re-review them.
- [ ] Refresh matrix and exact manifests after any repair; rerun only the checks invalidated by changes plus required final gates.
- [ ] Report stable compilation, behavior/contracts, durability/recovery, packaged runtime, live integration and deployment separately. Leave CI/publication blocked by the explicit pause; no CI receipt is fabricated.
- [ ] Leave reviewable diffs in the correct repositories, local run/recovery recipes, logs/checksums and a next action for every external blocker. Do not mark the full goal complete merely because the stable milestone passes.

## Gate command contract

Use a command-scoped environment; do not change rustup's global default.
Allocate the isolated target outside source and save the actual exit status.

~~~sh
RUSTUP_TOOLCHAIN=1.99.1 ./tools/cargo.sh --version
RUSTUP_TOOLCHAIN=1.99.1 ./tools/cargo.sh metadata --locked --format-version 1
RUSTUP_TOOLCHAIN=1.99.1 ABI_WDBX_PATH=:memory: \
  CARGO_TARGET_DIR="$ABI_OWNED_TARGET" ./tools/check.sh < /dev/null
~~~

The primary ABI gate supplies fmt/lint/build/test/docs/benchmark/security evidence;
do not duplicate all its stages without a new failure or scope reason.
WDBX has no tools/cargo.sh: invoke its bash tools/check.sh with the selected
1.99.1 toolchain bin and /usr/bin first, preserving the required loader path.
Abbey owns its separate check.sh and feature matrix. CI source changes remain
unexecuted remotely.
