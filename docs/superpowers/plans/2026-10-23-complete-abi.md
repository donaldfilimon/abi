# Complete ABI Implementation Plan

Approved for implementation in chat on 2026-10-23. The full inline plan is the
execution authority; this durable copy preserves its milestones and acceptance.
Specification: `2026-10-22-abi-production-ecosystem.md` and its governed-tool,
encrypted-recovery and governed-process supplements.

Goal: complete the approved production ecosystem, including original trained
models, governed tools, private workers/memory, public enrollment, three-platform
distribution and operational qualification. ABI owns execution/models/workers;
Abbey owns approvals/conversations/lifecycle; WDBX owns storage/privacy/key custody.

## Global constraints

Canonical main; preserve existing dirty work. Rust 1.99.1, wrapper-only locked
Cargo, tests with stdin EOF. Never use live user stores. Preserve CLI 14/MCP 12.
Existing receipts qualify only exact inputs and environments. CI, spending,
full training, live-account effects, publication and deployment retain their
existing authorization boundaries. Mobile, billing, public compute marketplace,
production FHE and additional accelerators are excluded from this release.

Platform scope update (Donald, 2026-10-23): build for all desktop platforms;
runtime test the current Mac. Three-OS runtime acceptance below is superseded
for the immediate delivery. Report cross-build artifacts separately from Mac
runtime evidence; do not infer Linux/Windows execution from a successful build.

## Review focus

Cancellation/revocation before dispatch; interrupted persistence and lost ACKs;
tenant/resource/approval mutation; deletion across all retained copies; actual
installed artifact/platform parity. Zero unauthorized or automatically repeated
effects is mandatory throughout.

## Milestone 1: Authoritative completion contract

- [ ] Reconcile every requirement with owner, source, acceptance and evidence.
- [ ] Classify qualified, unqualified, missing and external dependencies.
- [ ] Capture all three repository identities, dirty hashes/modes, dependencies,
      compiler and feature scope; retain baseline exit status and logs.
- [ ] Resolve baseline defects and cohort mismatches; reconcile historical claims.

## Milestone 2: Durable worker and relay guarantees

- [ ] Cover short writes, sync failure, terminal persistence interruption and
      pre-dispatch cancellation with regressions before behavioral repairs.
- [ ] Exercise restart, duplicate delivery, response loss, quota and streaming loss.
- [ ] Preserve recorded-state recovery and unresolved uncertain effects.
- [ ] Run worker/control-plane/browser suites and warning-denied Clippy.

## Milestone 3: Governed process and file operations

- [ ] Implement the existing sealed platform_process design and bounded supervisor.
- [ ] Verify real child lifetime/reap, output bounds, stop/cleanup uncertainty.
- [ ] Join exact approvals, durable lifecycle, pinned files and OS postconditions.
- [ ] Refuse changed resources/authority and replay without another effect.

## Milestone 4: Desktop and messaging adapters

- [ ] Define finite supported operations on macOS/Linux/Windows; refuse unavailable.
- [ ] Govern native desktop, Discord and Twilio through existing lifecycle types.
- [ ] Complete OpenAI/Anthropic/Grok with explicit transmission consent.
- [ ] Qualify approved scratch accounts/workspaces, refusal and uncertain delivery.

## Milestone 5: Memory privacy and lifecycle

- [ ] Complete WDBX-owned retention/erasure/key-custody contracts for active data,
      generations, replicas, backups, histories, outboxes and training artifacts.
- [ ] Implement durable progress, key lifecycle and scope-before-ranking retrieval.
- [ ] Prove interrupted deletion, offline replicas and restore cannot resurrect data.

## Milestone 6: Accounts, enrollment and browser

- [ ] Complete signup/pairing/status/revocation/deletion/recovery and real OIDC.
- [ ] Preserve 2-active/10-queued defaults and private encrypted application data.
- [ ] Qualify browser streaming/cancel/recovery and two separately hosted tenants.

## Milestone 7: Real Rust model pipeline

- [ ] Implement causal decoder transformer, numerical tests, versioned tokenizer,
      checkpoints, optimizer and deterministic resume.
- [ ] Complete licensed provenance/dedup/family split isolation; freeze evaluation.
- [ ] Validate tokenizer/runtime parity, bounded generation and explicit CPU/Metal.

## Milestone 8: Original weights

- [ ] Run approximately 100M/300M/1B pilots within explicitly approved resources.
- [ ] Record real cost/resources/loss/throughput and frozen-suite quality results.
- [ ] Select within the approved envelope or stop before full training.
- [ ] Train, qualify user-hosted inference and bind promotion/rollback to digests.

## Milestone 9: Platforms and installed packages

- [ ] Signed macOS/Linux/Windows packages and platform credential protections.
- [ ] Real OS clean install/upgrade/interruption/rollback/removal; CPU everywhere,
      supported Mac Metal; installed CLI/MCP and complete user journey acceptance.

## Milestone 10: Separate-host recovery and public operation

- [ ] Actual partitions/worker loss/reconnect/revocation and backup/restore tests.
- [ ] Content-free monitoring and tested incident/key/upgrade/shutdown runbooks.
- [ ] Independent assessment and blocker resolution, staging and approved rollout.

## Milestone 11: Release closure

- [ ] Freeze exact cohort; ABI/WDBX/Abbey root+desktop and installed platform gates.
- [ ] Actual audit, contracts, benchmarks and qualified pin promotion.
- [ ] Authorized CI/publication with checksums, models, evaluations and runbooks.
- [ ] Close every requirement with current evidence, zero approved Partial rows.

## Execution

Use superpowers:executing-plans. Regression RED, minimal repair, GREEN package
suite, review and source-bound evidence per task. Model development may proceed
after the baseline independently; integration still requires the full cohort.
No completion is inferred from fixture output, synthetic benchmarks, compilation
or local-only tests. Progress: `docs/verification/2026-10-23-completion-matrix.md`.
