# ABI stable source completion design

**Date:** 2026-10-20, America/New_York.
**Status:** implementation in progress under Donald's continuation instruction; this design alone is not qualification evidence.
**Brief:** Donald's “ABI — Existing-Architecture Completion Directive for Stable Rust 1.99.1,” supplied in this chat on 2026-10-20.

## Intent and current authority

Finish the existing ABI architecture as a usable, evidence-backed local Rust runtime.
The first deliverable is ABI/WDBX stable compiler integration; subsequent deliverables
join canonical capability authorization, the existing Abbey ledger, bounded agent
execution, and WDBX evidence. Preserve Abbey, Aviva and ABI identities and their
existing owners. Preserve Quesar's training/infrastructure role.

Donald explicitly lifted the local-test restriction in this chat: **local tests
are allowed; CI remains paused**. This supersedes the local-test pause recorded
in WDBX's stable migration commit. Do not push, dispatch workflows, create a PR,
publish, deploy, contact live providers, download models, start training, or change
live stores as part of this plan. Editing workflow source locally is permitted.
The clone of donaldfilimon/abbey was separately requested and completed for inspection.

The brief supplies the requested design direction. This document and the companion
implementation plan make the proposed implementation boundaries reviewable; they
do not authorize a live actuator or amend the constitution.

## Starting identities and existing evidence

| Item | Observed state |
| --- | --- |
| Canonical ABI | /Users/donald/sources/repos/abi; donaldfilimon/abi; main at cf3d9a425560a8223f476663ec45551d81efcf2c |
| ABI remote main | ca4a725c07ff75f885eb60bd59fe9797e12e4d2c; confirmed with git ls-remote |
| WDBX | adjacent checkout; main/local and remote at 08a69b3f22db61093e06527b3308b10589687022 |
| ABI's current CI dependency | cf2e326786c0054ac99c48bf512647c13e2bc3b1 |
| WDBX candidate | 08a69b3 is the published direct-child stable migration; local cpu.rs has a formatting-only diff |
| Abbey | requested sibling clone; clean main at f81e2ecfa75173257d466c4ecb220cabe3622dbf |
| Compiler pins | ABI and Abbey: nightly-2026-09-01; WDBX: 1.99.1 |
| Available stable compiler | rustc 1.99.1, 58cf8f9c35b355199727627d07ae34a4e2846d67, host aarch64-apple-darwin, LLVM 23.1.1 |
| Stable Cargo | cargo 1.99.1, 5f94df478, dated 2026-08-27; selected through ABI's wrapper with RUSTUP_TOOLCHAIN=1.99.1 |
| Rust floor | ABI, WDBX and Abbey currently declare 1.99; that declaration is separate from a tested compiler patch |

At planning time ABI contains three modified tracked files (Claude settings,
Codex hooks, and the architecture page), plus the two new hook scripts and their
test module. Preserve these edits. The tracked binary diff SHA-256 before these
planning documents was 2b79ec838d15cbd7df0336ad1fae936b17bc66f68d89da4eecd364b65c7ff053;
this hash excludes untracked files and is not a complete source identity.
Execution must capture a complete per-file manifest, including untracked inputs.

The hook work has 10 passing focused tests and attributable failing tests for
relative paths and extra hook arguments. An earlier full nightly gate passed
after adding the installed cargo-audit directory to PATH. Neither result qualifies
the proposed stable tree. Its exact final isolated gate has not run.
Mintlify validation failed to start because npx was unavailable.

## Selected architecture and alternatives

Use ordered integration milestones with explicit acceptance between them.
A wrapper-only compiler change would leave dependency and behavioral qualification
unresolved. A single compiler/governance/storage rewrite would combine independent
failure sources and make review and rollback harder. The selected approach preserves
the current crates and qualifies each join at its owning boundary.

### Stable compiler boundary

- Pin ABI's toolchain to exact 1.99.1, retaining edition 2024, resolver 3,
  rustfmt, clippy and rust-src. Keep rust-version at 1.99 and disclose that
  qualification covers 1.99.1, not newly tested 1.99.0.
- Keep tools/cargo.sh as the Cargo entry point. Preserve matching tools, system
  linker precedence, DYLD fallback handling and atomic native-library replacement.
- Preserve the wrapper's explicit developer override behavior. Stable acceptance
  sets RUSTUP_TOOLCHAIN=1.99.1 and rejects an effective different compiler,
  bootstrap bypass, or compiler substitution; an intentional developer override
  is a different, unqualified configuration.
- Reuse the published WDBX migration. Accept its immutable CI pin only after
  the exact candidate passes WDBX's own gate and ABI's paired gate. A gate on
  the formatted local sibling alone cannot establish that exact-commit result.
- Preserve lockfiles and dependency identities. Update resolution only for a
  demonstrated compatibility defect.

### Canonical governance and bounded host

Keep abi-agent-runtime's ExecutionPolicy pure and its existing AuditSink
observational. Do not use its infallible audit callback as durable permission
to execute. The new recording integration belongs in abi-agent-host, which may
depend on abi-capability; abi-capability continues to depend on the pure runtime.
This dependency direction is acyclic.

Extract a pure authorization evaluation from the canonical kernel and keep
authorize_and_record as a compatible fixture entry point. Add closed package-source
compilation and deterministic Guild Constitution narrowing in abi-capability.
Keep new internal versioned types separate from the immutable v1/v2 wire corpus.
Do not add a public live protocol or change existing closed wire semantics.

Add a distinct GovernedHost entry point with its own options and report types.
Share the existing bounded continuation driver; retain legacy AgentHost constructors,
reports and behavior. Trusted context supplies identity, exact scope, package,
grant, policy, schema, budgets, and fact freshness. Generated tool text supplies
only a proposal and schema-validated arguments.

The governed recording path follows all 17 specified stages, numbered 0–16.
Each reached decision stage has bounded fallible redacted audit. A pending
approval suspends that run and returns an inspectable preview; it never silently
continues into an effect. Approval resubmission reconstructs and revalidates the
exact proposal with a new run context.

Fresh fact acquisition occurs in the trusted host boundary, outside pure policy.
Unavailable, expired, changed or incompatible facts stop dispatch. Re-read facts,
revocation and approval bindings immediately before durable admission. Retain
finite host budgets and check cancellation before admission and before dispatch.

### Durable lifecycle owner

Abbey already owns schema-v7 SQLite runtime state and the tool_approvals,
tool_executions and append-only event tables. Its prepare_tool_execution uses
TransactionBehavior::Immediate to consume an approval and record intent atomically.
Reopen converts unfinished intent to Interrupted. Extend this owner; ABI receives
an additive ledger interface and no separate approval database.

Add the next numbered Abbey migration, currently 8, without editing migrations
1–7. Store governance bindings and richer observed outcomes as foreign-key-bound
extensions to the existing approval/execution rows for human-approved calls. A0
grant-only intents use the same RuntimeStore transaction and governed intent table
with absent legacy approval/execution links; never manufacture a human approval
to satisfy a legacy foreign key. Preserve legacy SQL constraints,
legacy methods and legacy daemon wire behavior. Implement governed admission with
approval consumption, intent, authorization bindings and the required pre-effect
audit committed in one SQLite transaction. Concurrent admission has at most one winner.

Interrupted and outcome-unknown effects are unresolved, never automatically retried
or compensated. Use the existing Interrupted base state for uncertain governed
executions, with an explicit governed reason in the extension. A postcondition
that is unavailable cannot produce verified success; record a non-success base
outcome plus the explicit unverifiable governed outcome. Do not rewrite history.

Postconditions are typed fresh observations. Compensation is a separately admitted
operation with its own scope, approval or valid preauthorization, audit and receipt.
Stop a compensation chain at its first failure. A cancellation request after
dispatch does not establish cessation or rollback; retain accounting until verified
completion/teardown.

The initial actuator is recording-only and selected explicitly. Existing live
Abbey memory-obsolete behavior remains on its separately qualified path. This plan
does not enable authority-bearing abbey.v1 methods or a generic executor.

### WDBX evidence and runtime composition

Implement store-aware joins in SEA/application composition, never in abi-ai's pure
core or abi-agent-runtime. Carry source/version, provenance, epistemic and lifecycle
state, access scope, unknown confidence and supporting outcome references explicitly.
Filter unauthorized candidates before scoring, counts or prompt construction.
Keep generic persisted model output inferred; no signature, similarity score or
audit pass promotes it to verified knowledge.

Preserve SEA's scorer, deduplication, candidate and byte bounds, and exclusion of
audit-rejected completions. Use record_vector_pair for atomic completion evidence
and the existing canonical episode proposal/write gate for selective outcome memory.
Detached interpretation references immutable signed source; it does not alter signed bytes.

Expose a new internal persistence result distinguishing disabled, open failure,
committed, refused and outcome unknown. Preserve existing CLI/MCP response schemas
through explicit compatibility projections.

Integrate only the existing generated tiny CPU fixture and already implemented
provider interfaces. Use a scripted provider for proposed tool calls; abi-bigram-v1
does not gain structured tool calling by being hosted. Worker-control fixtures may
participate as local evidence, but no network worker listener or scheduler binding
is required for this release's minimum local composition.

## Requirements-to-evidence matrix

All status labels below describe the integrated release requirement, not every
participating crate. Existing tests remain evidence only for their original source.

| ID / requirement | Owning authority and code | Present gap / status | Required acceptance and allowed environment |
| --- | --- | --- | --- |
| S1 exact stable selection | directive §4; ABI toolchain/wrapper/gate/CI policy | nightly pin; partial | hostile PATH/override fixtures plus actual compiler identities; local scratch |
| S2 immutable sibling pairing | directive §§1,4; both manifests and gates | stable candidate only compiler-checked historically; partial | exact WDBX candidate own gate plus paired ABI gate; isolated sibling layout |
| S3 numerical compatibility | WDBX compute public contracts; directive §4 | five existing CPU tests, migration edge cases unqualified; partial | independent numeric corpus, tails, signed zero, nonfinite, ties, cancellation at its actual owning APIs; no speed claims |
| G1 package compilation / narrowing | Program 2 spec §§2,5–7; abi-capability | compact typed package only; partial | closed package/source validation, prohibited operations, scope/delegation and policy narrowing; fixtures |
| G2 canonical host projection | Program 2 executed-deferrals; abi-agent-host | pure policy and observational audit do not complete durable governance; partial | 17-stage trace, fresh facts, A0–A5, exact preview, redaction and fail-closed audit; recording only |
| D1 durable approval admission | Program 2 §§6,7,9; Abbey runtime store | legacy exact-call ledger lacks full governance bindings; partial | atomic consumption/intent/audit, concurrency, restart and corruption refusals; owned scratch SQLite |
| D2 recovery / outcome / compensation | Program 2 §§8–10; Abbey plus ABI host | existing Interrupted recovery, richer outcomes absent; partial | actual process crashes/restarts, uncertainty retained, fresh-call recovery, failed postconditions and compensation; recording state only |
| E1 authorized evidence recall | constitution §§4–5; SEA and WDBX | generic recall is not a principal-scoped governed view; partial | isolation before scoring/counts/errors/cache/prompt, explicit provenance, rejected evidence excluded; scratch stores |
| E2 selective outcome persistence | canonical episodes / member erasure / WDBX contracts | atomic completion API exists; integrated outcome join incomplete; partial | exact transaction/episode interfaces, five persistence outcomes, signed-byte and causal compatibility; scratch only |
| R1 qualified local composition | runtime/host/model contracts; directive §8 | foundations present, governed end-to-end composition absent; partial | scripted tools and distinct generated CPU fixture, bounded continuation/cancellation, truthful placement; local |
| R2 remote workers / new model / real training | directive §8; Programs 6–7 | deferred from this source release | separate selected model/data/resource or transport contract and execution scope |
| Q1 compatibility / packaging | public-api, corpus compatibility, release contracts | no stable exact-tree receipt; partial | unchanged goldens/corpus, full gates, fresh packaged CLI/MCP plus dylibs; local artifacts |
| Q2 hosted CI / publication | explicit user restriction; release receipt policy | blocked: CI paused, publication not requested | keep workflow edits local; resume only under later authorization |
| Q3 live integration / deployment / external review | constitution and claims policy | deferred from local release | separate operator scope, hardware/access and independent evidence |

Preserve P0–P7 identities. In particular, learning remains Program 6, federation
Program 7, canonical memory Program 4, capability authorization Program 2,
and reversible execution Program 5. Do not equate WDBX's claim-ledger C labels
with ABI constitutional labels solely because their names match.

## Completion boundary and unresolved external packets

The local source milestone is complete only when each selected local row is
verified against exact manifests or explicitly blocked with reproducible evidence.
Stable compilation, behavior/contracts, durability/recovery, packaged runtime,
live integration, and deployment receive separate conclusions.

Follow-on packets remain: CI resumption; publication; approval presentation in
Swift AbbeyBot versus Rust Discord surfaces; selected licensed production model;
real training/evaluation; remote worker deployment/certificates; supported non-Mac
runtime; unavailable CUDA/Vulkan/ANE execution; independent review. UI ownership
does not block the internal recording preview and durable lifecycle.

Implementation sequence and concrete gates are in
[the companion plan](../plans/2026-10-20-abi-stable-source-completion.md).
