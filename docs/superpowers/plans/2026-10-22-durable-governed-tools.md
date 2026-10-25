# Durable Abbey approvals and governed tool execution plan

**Status:** In progress, 2026-10-22. Local source boundaries exist; integrated
qualification and product completion remain open. This continues the full
[production ecosystem plan](2026-10-22-abi-production-ecosystem.md) and does not
replace its unfinished requirements.

**Goal:** Carry encrypted enrolled-tenant tool proposals through an exact durable
Abbey-owned local decision, a single governed effect, durable worker history and
verified private result delivery. Reconnect, duplicate submission and uncertain
outcomes must never repeat the effect.

**Contract:** [Durable governed tools](../../spec/durable-governed-tools.md).
**Execution evidence:** [Ecosystem ledger](../../verification/2026-10-22-production-ecosystem.md).

## Constraints and rulings

- Work in canonical `main`; preserve other owners' dirty and untracked files.
  No commit, CI resumption, deployment or live account/resource action is implied.
- Use `./tools/cargo.sh`; resolving commands use `--locked`, tests `< /dev/null`.
  Keep Rust files within the existing size gate and use scratch stores/resources.
- Startup owns source packages, scope, authority, clocks and adapters. Neither a
  model nor an encrypted remote proposal can choose an arbitrary executor.
- Abbey owns local preview/decision persistence and atomic approval consumption
  with intent. No central/network decision endpoint is introduced.
- The initial relay bridge supports only `A1Actor`; higher levels fail closed.
- Preview identity and expiry remain immutable across repeated pending checks;
  source expiry policy and exact grant/package/request binding remain enforced.
- Current tool permits may renew for the same job after fresh authority checks.
  Renewal is not new admission, renewed approval or permission to retry an effect.
- Local pending proposals survive reconnect. Sticky cancellation, abandonment,
  expiry and terminal state prohibit dispatch. Recovery never invokes an adapter.
- Keep inference/model behavior, existing encryption formats and frozen CLI/MCP
  catalogs unchanged. Do not claim the complete objective from library fixtures.

## Tasks and acceptance

- [x] 1. Define the shared local approval contract and explicit startup-owned live
  executor boundary, preserving the recording API.
  Evidence: `abi-agent-host/src/local_approval.rs`, `governed_dispatch.rs`.
- [x] 2. Reuse policy/schema/budgets and durable intent consumption; reobserve full
  authority after prepare and stage 12; preserve uncertain outcomes without retry.
  Focused host tests and Clippy passed during this continuation. Final whole-gate
  evidence is separate.
- [x] 3. Implement Abbey local immutable previews, explicit decisions and durable
  lifecycle composition in sibling `src/runtime/store/governed/local.rs` and
  migration 11. This checkbox records implementation, not final sibling acceptance.
- [x] 4. Implement encrypted tool proposals, tool-only runtime registration,
  local preview/admission composition, bounded actual results and pending recovery.
  Source: `abi-control-plane/src/relay_runtime/tools.rs`; tool-specific current
  permit renewal and sticky outbox guards are part of this integration.
- [x] 5. Finish focused control-plane and Abbey acceptance: stable repeated preview,
  explicit local approval, changed parameters/resource/tenant/user refusal,
  denial/expiry/cancellation, reopen, replay, revoked authority during prepare,
  uncertain adapter/persistence failure and exact result/history bounds.
- [x] 6. Wire the real native outbound worker driver and browser/native secure
  local authority UI. Prove that remote submissions cannot decide approvals.
- [ ] 7. Add concrete platform adapters beyond scratch file effects. Require scoped
  resources, permission observations, output/postcondition evidence and actual
  approved/refused designated-account tests on each supported platform.
- [x] 8. Run full-chain process-kill and network-loss acceptance across local
  approval, worker admission, effect, terminal persistence, outbox and consumer.
  Assert no duplicate effect after every interruption and duplicate delivery.
- [x] 9. Complete final ABI `./tools/check.sh`, required dependency scan, sibling
  Abbey gates, docs/policy validation and immutable source/log receipts. Record
  exact passing commands and remaining platform/deployment gaps in the ledger.

## Review checkpoints

| Risk | Required evidence |
| --- | --- |
| Pending approval changes with time | Same preview digest and immutable expiry after repeated inspection/reopen |
| Approval does not describe effect | Exact resource/parameters/effects and package/request commitment; changed binding refused |
| Revocation crosses prepare/audit | Fresh permit and full host fact rechecks prevent adapter invocation |
| Worker and approval stores disagree | Each crash ordering becomes cached or unresolved, never repeated execution |
| Returned output exceeds delivery bound | Actual schema validation and coordinated host/worker/transport byte reserves |
| Cancellation is forgotten on reconnect | Sticky local flags and consumed/terminal decisions refuse dispatch |
| Local fixtures are mistaken for product completion | Keep native/UI/adapters/full-chain acceptance and final gates open until proved |

The objective closes only when the real user flow and required acceptance pass.
Local source milestones, previous inference-recovery receipts and model/runtime
infrastructure do not substitute for that outcome.

Native assembly checkpoint: sibling Abbey's governed relay example compiles,
passes its real scratch-file configuration/effect fixture and warning-denied
all-target Clippy. Task 6 remains open for actual interactive network acceptance,
not for absence of a native entry point. Actual cross-repository process tests
now kill after prepare and after a synced file effect before terminal persistence;
both reopen unresolved and never repeat the adapter. The broader network-loss,
consumer and platform acceptance rows remain open until separately demonstrated.

Owner-isolation review found a compensation-origin reservation path: an owned
new approval could otherwise reserve another owner's failed execution in the
same scope. Fix round 1 targets the local wrapper before delegation and adds
foreign/unknown/legacy-origin refusal plus legitimate owned-origin acceptance.
Legacy trusted `RuntimeStore` behavior remains a separate boundary.

Owner-isolation fix round 1/5: the compensation-origin finding is addressed;
scoped spec-compliance and code-quality re-review pass. All 27 governed-store
tests and warning-denied Abbey library/test Clippy pass. Foreign lookups hide
existence; cross-owner updates and foreign/unknown/legacy compensation origins
refuse before mutation. The full root gate is rerunning on this final source.

Task 6 acceptance: `governance-native-denial-final.log` passes actual native
TCP/PTY enrollment, exact local approval and denial, encrypted terminal delivery
and identical replay after both restarts. Denied work creates no worker admission,
execution intent or file; the approved effect remains exactly once. Accounts and
sessions are seeded fixtures, so public OIDC login remains separately unqualified.
The adapter is a startup-scoped scratch-file writer, not general platform tools.

Final local gates pass: ABI gate3, WDBX gate2, Abbey root gate6 and desktop final.
The final native-qualified receipt binds actual approve/deny/restart results to
executable digests. The concurrent Metal test target was rebuilt and passes.
Source snapshot and validation manifests retain dirty-file identities and failed
receipts; cross-owner immutable pin promotion remains a separate checkpoint.
Task 8 local scratch acceptance is now covered by eight parent-driven SIGKILL
boundaries, consumer pre/post-commit kills, and native accepted-publication and
consumer-ACK response loss. The 2026-10-23 execution ledger maps all twelve
interruption points to tests and source/log receipts. This is composed local
acceptance: store kill fixtures and native TCP response loss are distinct tests.
Task 7 remains open for general platform adapters. Separate-host partitions,
Windows/Linux runtime and power-loss acceptance remain unqualified. The parent
ecosystem objective remains open.
