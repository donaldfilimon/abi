# Durable local approvals and governed tool relay

Status: **Partial; local interruption acceptance updated 2026-10-23.**
This continuation belongs to the production ecosystem objective. Its local
library boundary does not complete the requested end-to-end product.
Implementation order and open acceptance work are tracked in
[the continuation plan](../superpowers/plans/2026-10-22-durable-governed-tools.md).

## Objective and authority

An enrolled tenant may submit an encrypted tool proposal to its own user node.
Abbey must present an exact durable local preview, accept an explicit decision
from the startup-authenticated local user, and consume that decision atomically
with the execution intent. The service transports proposals and results; it
cannot approve, select an executable, or restore consumed execution authority.
Reconnect and replay must not repeat an effect whose outcome is uncertain.

The proposal contains only protocol, registered tool, immutable tool-set revision
and JSON parameters. Startup supplies the compiled source, scoped registration,
trusted fact provider, local approval store, clock and concrete adapter. Model or
remote input cannot replace these dependencies. Existing inference/model behavior
and the frozen CLI and MCP catalogs remain unchanged.

## Current local implementation

| Boundary | Current source | Acceptance scope |
| --- | --- | --- |
| Local preview and decision contract | `crates/abi-agent-host/src/local_approval.rs` | Exact preview commitment and local authority interface |
| Durable Abbey ownership | Sibling Abbey `src/runtime/store/governed/local.rs` and `src/runtime/migration_11_local_approvals.sql` | Local SQLite preview/decision/lifecycle composition; owner isolation and cross-repository scratch acceptance |
| Live governed host | `crates/abi-agent-host/src/governed.rs` and `governed_dispatch.rs` | Explicit startup-owned adapter; existing recording API retained |
| Encrypted tool proposal and worker admission | `crates/abi-control-plane/src/relay_runtime/tools.rs` and `relay_runtime.rs` | Local library composition with durable worker history |
| Current service authority and recovery | `crates/abi-control-plane/src/relay.rs` and `relay_runtime/recovery.rs` | Tool permit renewal and pending proposal recovery; qualified local receipts in the execution ledger |
| Fixture acceptance | `crates/abi-agent-host/tests/governed/live.rs`, control-plane `service/tests/governed_tools.rs`, sibling Abbey local approval tests | Real scratch-file effects, parent-driven SIGKILL at eight durable boundaries, native accepted-response loss and consumer commit interruption tests; final receipts in execution ledger |

Focused live-host tests cover fresh authority changes during prepare and stage 12,
actual result schema/size refusal, adapter errors/panics, and terminal persistence
loss without replay. Those checks do not establish the complete network/UI chain.
The broader control-plane and Abbey acceptance suites and final gates are tracked
in the execution ledger; source presence alone is not passing-test evidence.

## Exact local preview

The committed preview includes tenant, account, node and local user; canonical
request, principal and resource scope; selected grant and package/binding digests;
tool, operation, human-visible resource, canonical parameters, expected effects,
and immutable creation/expiry times. A decision covers this complete preview,
not a tool name. Changing any bound field requires a new approval.

Creation derives from the immutable signed submission time. Expiry is bounded by
that creation time plus the compiled source expiry policy and maximum approval
TTL, and by job, source and grant absolute expiry. Repeated inspection must not
extend expiry or create a different preview digest merely because the clock
advanced. Pending, denied, expired, cancelled and consumed states never authorize
a new effect. The relay bridge currently admits only source approval policy
`A1Actor`; higher approval levels fail closed until their distinct local identity
and role flows are implemented and accepted. No network decision API is added.

## Execution and crash ordering

1. Verify the signed submission, node/account/tenant/tool-set binding and current
   authority. Decrypt only on the user node; validate the registered source schema.
2. Reserve the private outbox entry. An existing pristine pending proposal can be
   inspected again; cancellation, abandonment, expiry and terminal state prohibit
   dispatch. Preserve unexpired pending proposals across reconnect without
   admitting a worker or invoking an adapter.
3. Persist the exact local preview. Return pending until the local owner supplies
   an explicit matching decision. The service cannot manufacture that decision.
4. Admit the durable worker job, then let the shared governed host prepare the
   local execution intent and consume the exact approval in one local transaction.
   These stores do not claim a distributed atomic transaction.
5. After prepare, and again after the fallible stage-12 audit immediately before
   dispatch, reobserve trusted facts and re-evaluate complete authority, schema
   prerequisites, source/adapter binding, budgets, cancellation and deadlines.
   Ignore only the just-consumed exact decision marker for that same prepared
   execution. Changed authority or preconditions finish as known not dispatched.
6. Invoke the startup-selected adapter once. Validate its actual returned JSON,
   schema, status and byte bound; verify trusted postconditions. Store terminal
   lifecycle evidence, then worker result history containing bounded tool results
   and outcomes. Encrypt output through the existing private outbox/replay path.

An adapter error or panic, lost postdispatch persistence, or interrupted admission
remains uncertain and never authorizes automatic retry. A crash between worker
admission and local prepare may sacrifice execution availability: recovery records
an unresolved job instead of guessing whether an effect occurred. A crash after
local completion but before worker completion also cannot authorize another
adapter invocation. Reconciliation has no adapter argument.

Tool execution permits are short lived and rechecked through the trusted fact
provider. A fresh current-authority check may renew a permit for the same job;
this neither re-admits the worker nor restores a consumed approval. Delivery-only
replay authority remains distinct from execution authority. Cancellation and
revocation must be rechecked at the final dispatch boundary, not inferred from a
retained permit or encrypted artifact.

## Isolation and limits

Plaintext parameters, human previews, decisions and adapter output stay on user
infrastructure; central durable state remains routing/lifecycle commitments.
Existing HPKE encryption, exact submission binding, tenant/node ownership checks,
quotas, durable result replay and client verification remain required. This is
not an owner-rollback defense, external security assessment or production claim.

Result admission reserves space for JSON escaping and lifecycle/result metadata
before invoking the adapter. The host, worker and encrypted transport ceilings
must agree. Invalid output cannot be presented as successful schema-valid output.
Tests and smokes use scratch stores and resources, never `~/.abi/`.

## Open product acceptance

The native local-terminal approval/denial and browser sealed-proposal handoff
are exercised in the execution ledger; no remote decision authority is exposed.
Remaining acceptance includes:

- Add concrete scoped desktop/file/process/messaging adapters beyond the scratch
  file fixture, with platform permissions and designated-account acceptance.
- Qualify separate-host network partitions, platform runtime interruption and
  power-loss behavior beyond the local scratch interruption fixtures below.
- Repeat complete ABI/sibling qualification and documentation/policy checks
  after remaining platform integration; retain exact source/log receipts.

Public deployment, separate-host partitions, Windows/Linux runtime acceptance,
noncooperative termination and independent assessment remain open. No new model
weights, model training, quality result or deployment is established here.

## Native local assembly

Sibling Abbey provides `examples/governed_relay_worker.rs`. It reopens pending
sealed proposals, refreshes authenticated inspection permits, prints the exact
private preview to an interactive local terminal, and requires an explicit
`approve <preview digest>` or `deny <preview digest>` decision. Nonterminal input
refuses. Its `create_scratch` / `scratch-create-v1` adapter writes only a fixed
per-job UUID filename inside the startup-owned private scratch directory, using
create-new semantics and actual file evidence. This is a runnable local assembly,
with a passing offline configuration/effect fixture; native TCP/PTY approval, denial and accepted-response-loss acceptance is
recorded in the execution ledger. General platform adapters remain open.

A tool proposal can be prepared using `relay_job prepare-tool`; enroll its
separate node with `enroll ... --tools`. The browser accepts only a durable sealed
wrapper for tool execution and exposes no decision endpoint. Approval status
returns an opaque call ID for local preview lookup. The immutable creation time
is the first local preview time, reused on subsequent pending checks; expiry is
bounded by source policy, job, source, grant and the fifteen-minute maximum.


## Local interruption acceptance continuation (2026-10-23)

The execution ledger maps each interruption point to its named acceptance test.
Sibling Abbey's real approval/worker/outbox composition now has parent-driven
hard-kill cases after preview, decision, worker admission, lifecycle prepare,
synced file effect, lifecycle terminal, worker completion and complete outbox.
Repeated reopen/replay verifies a synced adapter-invocation counter, unchanged
file bytes/mtime, and exact persisted encrypted packets/manifests. Pending work
retains its exact preview identity and expiry; approved continuation requires
fresh authority. Cached/unresolved recovery neither reads approval authority nor
consults fresh fact providers.

The native TCP proxy discards publication and consumer-ACK responses only after
receiving successful upstream acceptance. The demo worker exits on lost publish
acknowledgement; the harness explicitly restarts it once, then verifies recovery
without a new decision or changed effect. Consumer import tests separately kill
before/after SQLite commit and reject signed conflicting/foreign/changed-binding
pages without advancing the journal. These tests compose with native verified
import and duplicate delivery; they do not simulate a partition between separate
hosts or power loss. Final source identities, gate results and retained failures
belong to the execution ledger, rather than this capability description.
