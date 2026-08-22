# Program 2: Capability and Authorization Runtime

> **Constitutional mapping.** This file was written against the seven-program
> list Donald gave in conversation. The ratified constitution's section 13
> "Delivery programs" uses a different numbering, and the constitution wins. In
> section 13's terms this document specifies **Program 2, ABI authorization and
> capability kernel**, which is the one case where the two numberings agree.
>
> The filename is name-based so no numbering is asserted. Section 13 caps this
> program at recording adapters with no production Discord mutation authority,
> and that bound is honored: live guild writes belong to section 13's Program 5.


Status: **design proposal, C0 (Specified) only.** Nothing in the Proposed
sections of this document is implemented, tested, replayed, shadowed, canaried,
or witnessed. Every mechanism proposed here starts at C0 on the constitution's
§11 ladder and earns nothing further from having been written down.

Governing document: `2026-08-22-abbey-system-constitution.md`. Where this spec
and the constitution appear to disagree, the constitution wins and this spec is
wrong; §15 requires an amendment before any program contradicts it. The
discrepancies I found are listed in section 12 rather than resolved unilaterally.

**Program numbering.** Every cross-reference in this document to another program
by number uses the constitution's §13 list, which is the normative one: Program 3
is read-only Discord guild intelligence, Program 4 is canonical WDBX episodes and
claims, Program 5 is approved reversible guild execution, Program 6 is the model
registry and adaptive arbiter, Program 7 is application federation and production
profiles. Other spec files written on 2026-08-22 use a different numbering; where
they disagree with §13, §13 governs until §15 amends it. This is recorded as
discrepancy C11 in section 12.

Scope, in Donald's words: "Define typed capabilities, API-learning packages,
credential isolation, guild grants, approval levels, actuator validation,
postconditions, audit, and rollback."

Constitutional ceiling on the deliverable, from §13 Program 2: "Implement
deny-by-default authorization, capability compilation, approval, revocation,
policy versioning, and redacted receipts **against recording adapters. No
production Discord mutation authority.**" This spec designs the full mechanism
including guild grants, actuator validation, postconditions, and compensation,
and then scopes the *shipped* binding to recording adapters (section 11). Live
guild mutation authority is Program 5.

---

## 1. What already exists

Everything in this section was verified by reading the named source file in this
session. Line-level behavior is quoted from the source, not from documentation
about the source.

### 1.1 `abi-agent-runtime` (crate `abi-agent-runtime`)

`crates/abi-agent-runtime/src/tool.rs` defines the tool vocabulary:

- `ToolEffect { ReadOnly, Mutating, Destructive }`, with an explicit doc comment
  saying the declaration "is the tool author's claim about the tool, not a
  property this crate can verify."
- `ToolSpec { name, description, input_schema: String, effect }`. `input_schema`
  is an opaque string; this crate parses no JSON.
- `ToolCall { id, name, input: String }`, `ToolStatus { Ok, Error, Denied }`,
  `ToolResult { call_id, status, payload: CapturedText }` with a byte ceiling on
  capture and a disclosed `truncated` flag.
- `ToolRegistry` (object-safe, no `invoke` method by design) and
  `StaticToolRegistry`.

`crates/abi-agent-runtime/src/policy.rs` defines authorization:

- `PolicyDecision { Allow, RequireConfirmation { reason: String }, Deny { reason: String } }`,
  with `as_str()` labels `allow` / `require_confirmation` / `deny`.
- `ExecutionPolicy` trait: `name()` and
  `authorize(&self, call: &ToolCall, spec: Option<&ToolSpec>) -> PolicyDecision`.
  The doc comment requires implementations to be a pure function of their inputs.
- `DenyAllPolicy` (the safe default) and `EffectScopedPolicy`. `EffectScopedPolicy`
  allows `ReadOnly` outright, confirms `Mutating`, denies `Destructive`, and denies
  an undescribed tool. Its own doc comment states it "trusts it and does not
  verify it."
- `AuditEntry { policy: String, call_id: String, tool: String, decision: PolicyDecision }`,
  `AuditSink` (object-safe, `&self`), `NullAuditSink`, `MemoryAuditSink`
  (in-memory `Vec`, explicitly documented as having "no ceiling of its own").
- `authorize_and_audit(policy, audit, call, spec) -> PolicyDecision`, the single
  join point, which performs no execution.

### 1.2 `abi-agent-host` (crate `abi-agent-host`)

`crates/abi-agent-host/src/host.rs` is the crate closest to invariant A3.
`AgentHost::new` compiles the registry's schemas at startup, so duplicate tool
names and invalid schemas fail before a model can run. `AgentHost::handle_call`
runs, in this exact order:

1. cancellation check;
2. deadline check (`check_deadline`);
3. `admit(state.tool_calls, budget.max_tool_calls, ToolCalls)`;
4. duplicate call-id rejection (`HostError::DuplicateCallId`);
5. `self.schemas.validate(call)` (JSON Schema validation);
6. event-budget reservation and `ToolCall` event emission;
7. `authorize_and_audit(self.policy, self.audit, call, Some(&tool.spec))`;
8. on `Allow` only, `self.executor.execute(...)`, then another deadline check,
   then `output_to_result` under `max_tool_result_bytes`.

`crates/abi-agent-host/src/schema.rs` compiles each `ToolSpec.input_schema` with
`jsonschema::validator_for` once at startup and validates every call's `input`
against it, producing `UnknownTool`, `MalformedToolInput`, or
`ToolSchemaViolation`. Validator diagnostics are truncated to 1,024 bytes by
`bounded()`.

`crates/abi-agent-host/src/executor.rs` defines `ToolExecutor`,
`ToolExecutionContext { cancellation, deadline }` with `should_stop()`,
`ToolOutput`, and `ToolExecutionError`. Its doc comment states the model supplies
only `call.name` and validated JSON, and that "executable paths, argv prefixes,
environment policy, credentials, and workspace mapping stay encapsulated inside
the implementation."

`crates/abi-agent-host/src/budget.rs` defines `HostBudget` with finite defaults:
`max_events` 1024, `max_event_bytes` 65536, `max_output_tokens` 16384,
`max_output_bytes` 1048576, `max_tool_calls` 32, `max_tool_rounds` 8,
`max_provider_runs` 9, `max_tool_result_bytes` 65536, `max_duration` 300s. The
matching `HostBudgetLimit` enum names each ceiling.

`crates/abi-agent-host/src/error.rs` is a fail-closed vocabulary including
`ProviderToolResult` (a provider fabricating a tool result), `PostTerminalEvent`,
`InvalidTerminalSequence`, `ToolResultTooLarge`, and `BudgetExceeded`.

**What `abi-agent-host` does not have.** There is no principal, no scope, no
tenant, no guild, no grant, no capability id, no capability version, no approval
round trip, no precondition, no postcondition, no rollback, no rate class, and no
idempotency key. `PolicyDecision::RequireConfirmation` is converted at
`host.rs:235-240` into a `ToolResult` with `ToolStatus::Denied` and the payload
`"confirmation required: {reason}"`, which is handed back to the model and the
loop continues. There is no channel through which a human could supply the
confirmation.

### 1.3 `abbey` (product runtime host)

`abbey` holds the only durable approval and execution ledger I found anywhere in
the federation.

`src/runtime/migration_5_tool_approvals.sql` creates `tool_approvals` and
`tool_approval_events`. The approval row is keyed by `call_id`, carries a
`call_digest` constrained to 64 lowercase hex characters, a `state` constrained
to `pending | approved | denied | cancelled | expired | consumed`, a `UNIQUE`
`decision_id`, a `UNIQUE` `cancellation_id`, and a table-level `CHECK` that
enforces which id may be present in which state. `tool_approval_events` is
append-only with an autoincrement `sequence` and `ON DELETE RESTRICT`.

`src/runtime/migration_6_tool_executions.sql` creates `tool_executions` and
`tool_execution_events`, with states `prepared | interrupted | succeeded | failed`,
a 64-hex `result_digest`, and a `CHECK` that a `prepared` row has no digest and no
finish time while a terminal row has both. `tool_executions.call_id` is a foreign
key into `tool_approvals` with `ON DELETE RESTRICT`.

`src/runtime/store/tool_approval.rs` implements the state machine:
`MAX_TOOL_APPROVAL_TTL_MS = 15 * 60 * 1_000` is a server-enforced ceiling on
approval lifetime; `create_tool_approval` rejects a zero or over-ceiling TTL;
`decide_tool_approval` requires the exact digest, applies `expire_if_needed`
first, refuses a non-pending record or a reused `decision_id`;
`cancel_tool_approval` may only terminate `pending` or `approved`;
`tool_approval` applies durable expiry on read. Its type doc says the record
"contains no raw tool input," and `ToolApprovalDecision` is documented as
"Explicit user decision; absence is never interpreted as approval."

`src/runtime/store/tool_execution.rs` is documented as a "Crash-recoverable
admission ledger for approved tool effects." `prepare_tool_execution` atomically
consumes the digest-bound approval and writes execution intent inside one
`TransactionBehavior::Immediate` transaction before any effect may run; a
daemon reopen marks unfinished intent `Interrupted`; "retrying an ambiguous
effect therefore requires a fresh call and approval"; and the module "never
stores raw tool input or output."

`src/daemon/runtime_v3.rs` is the actuator today. `invoke_approved_memory_effect`
recomputes the call digest, reads the approval, refuses on `tool_id` or digest
mismatch or non-`Approved` state, calls `prepare_tool_execution`, exercises a
debug failpoint `after_prepare`, records a `v3_tool_authorization` / `approved`
audit event, and only then performs the local memory effect. `record_tool_audit`
writes metadata `{ call_id, tool_id, input_digest, effect, policy }` plus an
optional `{ output_digest, output_bytes }`; raw input and output never appear.
Audit write failure maps to `internal_failure()`, so a call that cannot be
audited does not proceed.

`src/runtime/store/audit.rs` bounds every audit record: `MAX_METADATA_BYTES`
4096, `MAX_STRING_BYTES` 512, `MAX_COLLECTION_ITEMS` 32, `MAX_DEPTH` 4, metadata
must be a JSON object, and `validate_audit_label` rejects empty, over-64-byte, or
control-character-bearing action and outcome labels.

`src/app_core/v3.rs` defines a real typed capability set: `V3Capability` with
fifteen variants including `ListTools`, `InvokeTools`, `DecideToolApprovals`,
`CancelTools`, `ReadMemory`, `ReadModels`, `ManageModels`, `InferModels`, and
`V3CapabilitySet`, an ordered duplicate-free negotiated grant set whose
declaration order is the canonical serialized order. `src/app_core/contracts.rs`
defines a second, coarser `AppCapability` / `CapabilitySet` pair for application
operations, with `ReadRoutes` documented as granting "no execution authority."

`src/mcp_host/tools.rs` is the safe MCP registry. Its `EffectClass` enum has
exactly one variant, `ReadOnly`, so "a shell/exec tool is not 'absent by
configuration' here, it is unrepresentable," and adding a second variant is
explicitly named the reviewable event. It consumes `abi_agent_host::ToolExecutor`
and `abi_agent_runtime::{ToolCall, ToolEffect, ToolSpec}`, so `abbey` is already
a downstream consumer of the ABI seam this program extends.

`src/mcp_host/redact.rs` performs value-based outbound secret redaction over the
serialized MCP frame, with `REDACTION_PLACEHOLDER`, a documented
`MIN_REDACTABLE_SECRET_BYTES = 8` floor, and a `SENSITIVE_NAME_FRAGMENTS` list.
It is explicitly described as defence in depth, not the primary control.

`src/claims.rs` provides the honesty gate: `Status { Current, Partial, Proposed,
Blocked, OutOfScope }` over a compiled `CLAIMS` registry with
`CLAIMS_SCHEMA_VERSION`.

### 1.4 `abi` credential handling

`crates/abi-cli/src/auth.rs` uses `abi_foundation::credentials::{Backend,
CredentialField, Secret, backend_is_keychain, credentials_path}`. Fields are a
fixed set: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DISCORD_TOKEN`,
`GROK_API_KEY`, `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`. The backend label
states "keychain (macOS login keychain, opt-in)" on macOS and "keychain requested
(unsupported on this OS; using file, Windows/Linux Proposed)" elsewhere; the
default is a file at `~/.abi/credentials.json`.

There is no tenant, guild, or capability dimension on a credential. One machine
user has one credential set.

### 1.5 `abi` os-control policy, the narrowing precedent

`crates/abi-cli/src/os/policy.rs` states the invariant this program should copy:
"a policy file can only narrow, never widen." `CEILING` is a compiled-in
`&["true", "pwd", "ls", "whoami", "date"]`; a policy-file `allow` entry outside
the ceiling is ignored rather than honored, precisely because the file lives in a
writable home directory. Ignored names are reported back to the user rather than
silently dropped. Timeouts are bounded (`DEFAULT_TIMEOUT_SECS` 30,
`MIN_TIMEOUT_SECS` 1, `MAX_TIMEOUT_SECS` 3600), unknown keys are a loud error,
and `Source { Builtin, File(PathBuf) }` discloses provenance.

### 1.6 `abbey-bot` (Rust Discord adapter)

`src/tools.rs` defines a five-tool vocabulary (`remember_fact`,
`lookup_reputation`, `recall`, `switch_persona`, `recent_messages`) with
`MAX_RESULT_CHARS = 600`, `MAX_RECENT = 50`, `MAX_TOOL_ROUNDS = 3`, both provider
wire shapes (`openai_tools_json`, `anthropic_tools_json`), a `ToolHost` trait the
runtime implements over `AppState`, and `dispatch`. Every result is truncated by
`truncate`. No tool posts, moderates, or changes configuration.

`dispatch` does **not** validate incoming arguments against the JSON Schema it
published. It hand-extracts with an `arg_str` closure, clamps `recent_messages`
`limit` into `1..=MAX_RECENT`, and returns a plain string for a missing argument
or an unknown tool (`"Unknown tool \`{other}\`."`). An unknown tool is a model
message, not a refusal record.

`src/perms.rs` is an explanation surface, not an authorization surface. `Scope
{ Everyone, Role { id, name }, Member { id, name }, Unrecognized }`, `Overwrite`,
and `Subject { name, user_id, role_ids, is_admin, is_owner }` feed `applicable()`
(which imposes Discord's evaluation order rather than storage order) and
`explain()` (which returns prose). Owner and Administrator short-circuit before
any overwrite is enumerated. `Scope::Unrecognized` exists so an unmodelled
overwrite kind is reported as unknown instead of impersonating `@everyone`.
Nothing here returns a decision an actuator could consume.

`src/guild.rs` `GuildSettings` carries `enabled`, `default_persona`,
`learning_enabled`, `voice_enabled`, `vision_enabled`,
`reply_cooldown_seconds`, `epsilon_override`, `locale`, and `unsolicited`.
`learning_enabled` defaults to `true`; `unsolicited` defaults to false. These are
per-guild settings, not grants: they gate speech, not authority.

### 1.7 Summary of the gap

| Program 2 concern | Exists today | Where |
|---|---|---|
| Tool schema validation before policy | Yes, in ABI only | `abi-agent-host/src/schema.rs` |
| Deny-by-default policy default | Yes | `DenyAllPolicy` in `abi-agent-runtime/src/policy.rs` |
| Digest-bound single-use approval with TTL | Yes, in `abbey` only | `abbey/src/runtime/store/tool_approval.rs` |
| Crash-recoverable admission ledger | Yes, in `abbey` only | `abbey/src/runtime/store/tool_execution.rs` |
| Bounded, raw-free audit record | Yes, in `abbey` only | `abbey/src/runtime/store/audit.rs` |
| Typed negotiated capability set | Yes, session-scoped only | `abbey/src/app_core/v3.rs` |
| Narrow-never-widen policy precedent | Yes | `abi-cli/src/os/policy.rs` |
| Capability package with version, risk, reversibility | No | Proposed |
| Principal, delegation chain, scope on a decision | No | Proposed |
| Guild-scoped grant | No | Proposed |
| Approval levels | No | Proposed |
| Preconditions and postconditions | No | Proposed |
| Compensation and rollback receipts | No | Proposed |
| Per-tenant credential isolation | No | Proposed |
| Audit of refusals with principal and capability version | No | Proposed |

---

## 2. Proposed: the typed capability

### 2.1 Home

A new crate `abi-capability` in the `abi` workspace, depending on
`abi-foundation` and `abi-agent-runtime` and nothing else. `abi-agent-host`
gains a dependency on it. The dependency order stays acyclic and matches the
existing convention in `CLAUDE.md`: `abi-foundation` -> `abi-agent-runtime` ->
`abi-capability` -> `abi-agent-host`.

The wire schema is owned by the Program 1 Abbey contracts package (§10). Until
that package exists and is qualified, `abi-capability` carries the Rust types and
a versioned canonical JSON encoding, and §2 of the constitution forbids
documenting the contracts package as existing.

### 2.2 Declaration

A capability is declared as a `CapabilityPackage`, carrying exactly the §6 field
list. Nothing is added to that list and nothing is dropped from it.

```
CapabilityPackage {
    id: CapabilityId,                    // stable, lowercase dotted, <= 128 bytes
    version: SemVer,
    input_schema:  SchemaText,           // compiled once, as ToolSchemas does today
    output_schema: SchemaText,
    error_schema:  SchemaText,
    streaming: StreamingContract,
    cancellation: CancellationContract,

    required_platform_permissions: PermissionRequirement,
    resource_scope: ResourceScopeSpec,

    side_effect: SideEffectClass,        // None | LocalState | PlatformRead | PlatformWrite
    reversibility: Reversibility,        // Reversible{compensator, pre_state} | ReversibleWithLoss | Irreversible
    risk_class: RiskClass,               // Informational < Low < Medium < High < Prohibited

    data_classes: DataClassSet,
    residency: ResidencyPolicy,
    retention: RetentionClass,           // §5: Ephemeral | Session | Operational | Durable | MandatoryIncident

    preconditions:  Vec<Predicate>,
    invariants:     Vec<Predicate>,
    deadline: Duration,
    rate_class: RateClassId,
    budgets: BudgetRequirement,

    confirmation_policy: ConfirmationPolicy,
    delegation_policy:   DelegationPolicy,
    expiry_policy:       ExpiryPolicy,
    revocation_policy:   RevocationPolicy,
    approval_policy:     ApprovalPolicy,

    idempotency: IdempotencyPolicy,      // RequiresKey | NaturallyIdempotent | NotIdempotent
    compensation: CompensationSpec,
    rollback: RollbackSpec,
    postconditions: Vec<Predicate>,

    adapter_bindings: Vec<AdapterBinding>,   // adapter id + compatible API version range
    receipt_requirements: ReceiptSpec,       // redacted evidence and outcome-receipt fields

    fixtures: FixtureSet,                    // deterministic, adversarial, failure
    promotion_thresholds: PromotionThresholds,
    expiration_criteria: ExpirationCriteria,
}
```

`RiskClass::Prohibited` is representable so a package can declare an action the
kernel must refuse to grant at any level. It is the type-level analogue of
`abbey`'s `EffectClass` having exactly one variant: the reviewable event is
someone trying to remove it.

### 2.3 Compilation and registration

`CapabilityRegistry::compile(packages)` mirrors `ToolSchemas::compile`: it runs at
startup, before any provider can be invoked, and fails there rather than at call
time. Compilation:

1. rejects duplicate `(id, version)` pairs;
2. compiles `input_schema`, `output_schema`, and `error_schema` with
   `jsonschema::validator_for`, exactly as `schema.rs` does today, so a package
   with an unresolvable `$ref` fails at startup;
3. rejects a `Reversible` package whose named compensator is not also registered,
   or whose compensator declares a strictly higher `risk_class`;
4. rejects a `PlatformWrite` package with an empty `postconditions` list;
5. rejects a `NotIdempotent` package that does not declare
   `IdempotencyPolicy::RequiresKey`;
6. computes `package_digest`, a domain-separated SHA-256 over the canonical JSON
   encoding, matching the 64-lowercase-hex shape already enforced by
   `migration_5_tool_approvals.sql`.

`package_digest` is the anchor for drift detection. §6 requires that "schema
drift, permission mismatch, unsafe output, calibration regression, or missing
evidence disables the affected version and preserves the last approved version."
The kernel implements that by binding every grant to a `package_digest`; a
recompiled package with a different digest does not satisfy an existing grant.

### 2.4 Relationship to `ToolSpec`

A capability projects down to a `ToolSpec` at the generation boundary so the
existing `abi-agent-host` loop keeps working unchanged:

```
ToolSpec {
    name:         capability.id.as_tool_name(),
    description:  capability.model_facing_description(),
    input_schema: capability.input_schema,
    effect:       capability.side_effect.as_tool_effect(),   // presentation only
}
```

`ToolEffect` stops being an authorization input. It remains a model-facing hint
and an audit label. This is the single most important type-level change in the
program: today `EffectScopedPolicy` allows a `ReadOnly` call on the strength of
the tool author's own declaration, which is authority derived from a claim rather
than from a grant. Section 12 records that as a conflict.

Tool availability at the generation boundary is itself grant-derived (decision
44): the `ToolSpec` list handed to a provider is the projection of the
capabilities the current principal actually holds a live grant for. A capability
the principal cannot use is not described to the model at all.

---

## 3. Proposed: capability packages, API learning, and what evidence each stage earns

### 3.1 Accepted sources

Verbatim from §6: OpenAPI and JSON Schema; MCP and ABI tool definitions; Discord
application-command schemas; reviewed human-authored capability packages; and
unstructured documentation only as input to a candidate contract that still
requires validation.

A source that is not on that list produces nothing. A `CandidateContract` derived
from documentation is a distinct type from `CapabilityPackage` and is not
registrable, is not grantable, and cannot reach an actuator. Decision 42:
"Documentation may propose a schema but cannot authorize a guessed call."

### 3.2 Promotion stages and the evidence they earn

The brief's "L1 schema-validated / L2 deterministic qualification / L3 sandbox"
maps onto §6's numbered promotion stages. The constitution's normative ladder is
**C0 to C7** (§11); there is no L-ladder in the constitution and this spec does
not create one. The mapping:

| §6 stage | Brief's shorthand | Highest C-level the stage can earn | Permitted conclusion |
|---|---|---|---|
| 1. static schema and policy validation | L1 schema-validated | C1, and only once the package's unit, property, privacy, schema, and failure-path tests pass | The package conforms under test. Not executable. |
| 2. deterministic replay | L2 deterministic qualification | C2 | Replay-qualified: equivalent decisions and cancellations on frozen fixtures. |
| 3. sandbox execution | L3 sandbox | C2 for the adapter contract; C3 only with a baseline, an ablation, calibration, and adversarial evaluation | Sandbox-qualified. Measured offline value requires the C3 evidence, not sandbox success alone. |
| 4. proposal-only shadow use | | C4 | Predicts acceptably in the target environment. |
| 5. bounded canary use | | C5 | Works under restricted live authority, within a fixed scope, budget, monitor, and rollback. |
| 6. owner/admin-approved promotion | | C6 only when an authorized operator witnesses the exact end-to-end outcome | Live-qualified for that environment and version. |
| 7. monitoring, drift detection, revocation, rollback | | C7 | Sustained operational evidence with drift bounds. |

Three rules that are not negotiable, from §11 and decision 63: no stage
auto-promotes the next; a claim record names the exact binary, model, adapter,
platform, policy, schema, and fixture identities; and the capability cannot be
its sole evaluator (decision 67). Program 2 therefore ships a
`CapabilityEvaluator` that is a separate component from the kernel, and
`PromotionThresholds` are frozen in the package before results are inspected
(decision 65).

Abbey's own authority over this pipeline, from §6: she may improve selection,
parameter suggestions, workflow composition, and presentation. She may not
promote her own package, change its risk class, invent an undocumented endpoint,
guess a production schema, or convert successful use into new authority. The
kernel enforces the last one structurally: a `CapabilityGrant` is only ever
created by an explicit issuance operation, never as a side effect of a successful
execution, and the audit record for a successful execution has no field that
could carry a grant.

### 3.3 Package state

`proposed | partial | current | failed | revoked | superseded | expired`, from §0.
A package is `current` only at the C-level actually demonstrated, and the
registry stores the level alongside the state so nothing can read a `current`
package as C6 when its evidence is C2.

---

## 4. Proposed: credential isolation per tenant

### 4.1 The problem stated honestly

Today there is one credential set per machine user
(`abi_foundation::credentials`, file backend `~/.abi/credentials.json`, keychain
backend macOS-only). There is no tenant dimension, so "per-tenant credential
isolation" is entirely new work, not a hardening of something existing.

### 4.2 Design

A `CredentialRef` is an opaque, non-secret handle:

```
CredentialRef {
    tenant: TenantId,
    adapter: AdapterId,
    purpose: PurposeId,      // e.g. "discord.bot_token", "anthropic.api_key"
    version: u32,            // bumped on rotation
}
```

Rules:

1. **Capability packages and grants reference credentials only by
   `CredentialRef`.** No package, grant, proposal, prompt, tool input, tool
   output, audit record, receipt, or WDBX episode may carry a credential value.
   This extends the existing rule in `abi-agent-host/src/executor.rs`, whose doc
   already says credentials "stay encapsulated inside the implementation."

2. **Resolution happens inside the adapter boundary and nowhere else.** The
   actuator hands the adapter a `CredentialRef`; the adapter asks a
   `CredentialResolver` for a `Secret`; the `Secret` never leaves the adapter's
   call frame. `Secret` already exists in `abi_foundation::credentials` and must
   keep a hand-written `Debug` that does not print the value.

3. **Resolution is tenant-bound and fails closed across tenants.** A resolver
   configured for tenant A returns `Err(CredentialNotAvailable)` for a ref naming
   tenant B. Cross-tenant resolution is a typed error, never a fallback to a
   default credential. This is the credential analogue of §5's "No cross-guild
   recall."

4. **Resolution requires a live grant.** The resolver is called only from inside
   an actuation that has already passed authorization for a capability whose
   package declares that `PurposeId`. A capability may not name a purpose its
   package did not declare.

5. **Presence is not consent.** Decision 54: "A credential's presence is not
   consent to use a provider." A configured `CredentialRef` with no grant naming
   it authorizes nothing, and the kernel must not fall back to a provider merely
   because its key resolves. This mirrors the existing `abi` local/live connector
   split, where `complete --live` requires `abi auth signin` *and* an explicit
   flag.

6. **Rotation and revocation.** Bumping `CredentialRef.version` invalidates every
   in-flight actuation holding the older version at its next re-check.
   Decision 17: "Revocation takes effect before new work begins."

7. **Platform honesty.** The keychain backend is macOS-only today, per
   `auth.rs`'s own status string. On Windows and Linux the store is a file, so
   per-tenant isolation there is filesystem-permission isolation and must be
   declared as such in the claim record, not described as keychain-backed. The
   `windows credential ACL` CI job named in `abi/CLAUDE.md` is the existing
   evidence surface for the Windows side.

8. **Defence in depth on egress.** `abbey/src/mcp_host/redact.rs` already performs
   a value-based redaction pass over the serialized MCP frame. The same pass runs
   over every receipt and audit record the kernel emits, with its documented
   `MIN_REDACTABLE_SECRET_BYTES = 8` boundary restated rather than quietly
   assumed away. It is defence in depth, not the primary control; the primary
   control is that no secret is ever placed in those structures.

---

## 5. Proposed: guild-scoped grants

### 5.1 The grant

Field list taken verbatim from §3: "A capability grant names the action family,
issuer, recipient, scope, issue and expiry conditions, risk class, confirmation
policy, revocation state, and policy version."

```
CapabilityGrant {
    grant_id: GrantId,

    action_family: CapabilityId,             // or a family prefix
    capability_version_req: VersionReq,
    package_digest: Digest,                  // binds the exact compiled package

    issuer:    PrincipalRef,
    recipient: PrincipalRef,

    scope: GrantScope {
        tenant:    TenantId,
        guild:     Option<GuildId>,          // None = tenant-local, never cross-guild
        resources: ResourceSelector,
        subjects:  SubjectSelector,
    },

    issued_at, not_before, expires_at,
    max_uses: Option<u32>,

    risk_ceiling: RiskClass,
    confirmation_policy: ConfirmationPolicy,
    revocation: RevocationState,             // Active | Suspended | Revoked{at, by, reason_code}

    policy_version: PolicyVersion,
    guild_constitution_version: Option<Version>,
    safety_policy_version: Version,
}
```

Deny-by-default (§3, decision 11): the kernel's answer when no grant matches is
`Deny`, and the reason is a closed `reason_code`, not free text, so a refusal
cannot leak the existence or shape of a protected resource (§3: "Abbey identifies
the failed invariant without exposing protected data").

### 5.2 No cross-guild grant

A grant whose scope names guild G authorizes nothing in guild H. There is no
wildcard guild selector. §5: "Guild isolation is the correctness boundary;
guild-plus-user isolation is the member privacy boundary." A DM principal gets
its own scope (decision 24); `abbey-bot` already scopes DMs as
`"{network}:dm:{user}"` in `SocialEvent::scoped_guild_id`, and the grant scope
adopts that same identifier shape so the two cannot diverge.

### 5.3 Effective authority

```
effective = platform_permission_facts
          ∩ capability_package_ceiling
          ∩ tenant_policy
          ∩ guild_constitution
          ∩ grant_scope
```

Two rules on that intersection:

- **Platform permissions are necessary, not sufficient** (§3, decision 10).
  Having Discord `MANAGE_CHANNELS` does not authorize an Abbey channel change; a
  grant does, and only when the platform permission is also currently present.
  The permission facts must be freshly observed, not cached: `abbey-bot`'s
  existing rule is "fetch over REST, not from the cache," and the same reasoning
  applies here for a different reason (a cache read yields a silently thinner
  answer rather than an error).

- **A Guild Constitution may narrow platform authority and may never widen it**
  (§3). The implementation copies `abi-cli/src/os/policy.rs` exactly: the
  compiled package set is the `CEILING`; a guild-constitution entry outside the
  ceiling is **ignored, not honored**; ignored entries are **reported back** to
  the guild owner rather than silently dropped; unknown keys are a loud error
  rather than a setting that does nothing. That file's own justification carries
  over unchanged, because a guild constitution is likewise editable by someone
  other than the reviewer of the ceiling.

### 5.4 Grant issuance

Only these principals may issue a grant:

| Recipient scope | Minimum issuer |
|---|---|
| Tenant-wide | Tenant owner |
| Guild-wide | Guild owner |
| Guild, risk class Low or Informational | Guild administrator |
| Guild, risk class Medium or above | Guild owner |
| Any grant of an `Irreversible` capability | Guild owner, and never delegated |
| Any grant of `RiskClass::Prohibited` | Nobody. Refused at issuance. |

Decision 12: owner decisions outrank administrators; administrators outrank
learned preferences; platform and safety constraints outrank all. A learned
policy is never an issuer. §8: "No reinforcement learner directly controls roles,
channels, permissions, moderation, integrations, or command registration."

### 5.5 Revocation

Revocation is evaluated before new work begins (decision 17), which in pipeline
terms means the grant's `revocation` field is re-read at stage 5 and again at
stage 10 immediately before the effect. An in-flight actuation whose grant is
revoked between those two reads stops at stage 10 and produces a `Cancelled`
receipt with no effect.

---

## 6. Proposed: approval levels

### 6.1 The ladder

```
ApprovalLevel {
    A0None,          // a current grant is sufficient
    A1Actor,         // the requesting human confirms, in the same surface
    A2Manager,       // a guild manager holding the corresponding platform permission
    A3Admin,         // guild administrator
    A4Owner,         // guild owner
    A5DualControl,   // two distinct A3+ principals, neither of whom is the proposer
}
```

`ApprovalLevel` is `Ord`. It is not learned, not model-computed, and not derived
from engagement (decision 37).

### 6.2 How the required level is computed

```
required = max(
    package.approval_policy.floor,          // set by the package author, reviewed
    grant.confirmation_policy.level,        // set by the grant issuer
    risk_floor(package.risk_class),         // Medium -> A2, High -> A4, Irreversible -> A4
    regime_floor(current_regime),           // incident or emergency regimes raise it
    safety_floor(safety_policy_version),    // the safety path may raise unilaterally
)
```

The function is total, pure, and deterministic over typed inputs. No model output
is an input. This is invariant A3 expressed at the approval layer.

### 6.3 Who may raise and who may lower

- **Anyone in the chain may raise.** A guild administrator may set a grant's
  confirmation policy above the package floor. The safety path may raise without
  consulting a model (§3, decision 14) and may do so online.
- **Only the issuer of a given constraint may lower that constraint, and never
  below the package floor.** A grant issuer may lower the grant's own
  contribution; nobody may lower `package.approval_policy.floor` except by
  publishing a new package version, which resets the evidence ladder for that
  version.
- **Safety is never learned online** (decision 15). `safety_floor` is not
  writable by any runtime path.
- **Repeated approval does not become standing authority** (decision 16). The
  approval record is single-use and digest-bound. This property already exists in
  `abbey`: `tool_approvals.decision_id` is `UNIQUE`, and the `consumed` state is
  reached by `prepare_tool_execution` atomically consuming the approval.

### 6.4 Binding and lifetime

An approval binds to (`call_id`, `call_digest`, `capability_id`,
`capability_version`, `package_digest`, `grant_id`, `approver_ref`,
`approval_level`). `abbey` today binds the first two; the rest are added. The
digest is a domain-separated SHA-256 over canonical JSON, 64 lowercase hex,
matching the existing `CHECK` constraint.

The server-enforced TTL ceiling stays at the existing
`MAX_TOOL_APPROVAL_TTL_MS = 15 * 60 * 1_000`. Per-risk defaults sit under it:
A1 ten minutes, A2 and A3 five minutes, A4 and A5 two minutes. Expiry is a
first-class terminal state, already present as `expired`, and "absence is never
interpreted as approval" is already the documented contract on
`ToolApprovalDecision`.

### 6.5 Separation of duties

At A2 and above, the proposer principal may not be the approver.

Observation, not a defect claim: `abbey`'s default safe daemon negotiates both
`V3Capability::InvokeTools` and `V3Capability::DecideToolApprovals` on the same
owner-only authenticated Unix socket, so one session can both propose and decide.
That is coherent while the only principal is the machine owner acting on their
own local state. It must not survive into multi-principal guild grants, where the
same shape would be self-approval. The kernel therefore treats
`DecideToolApprovals` as scoped to a principal and refuses a decision whose
`approver_ref` equals the proposal's `principal_ref` whenever
`required_level >= A2Manager`.

### 6.6 The three visible stages

§3 requires that consequential interaction has three visible stages: **Recommend**
(explain what could be done), **Propose** (an inspectable action plan and
predicted effects), **Execute** (only under a current grant and validated
preconditions). The approval preview rendered at level A1 and above is the
Propose artifact, and it carries, verbatim from §3, "a bounded preview, expected
effects, uncertainty, risk, expiration, and rollback path."

---

## 7. Proposed: the actuator validation pipeline

Sixteen ordered stages. Every stage fails closed. Every stage that terminates or
advances a decision writes an audit record (section 9). No stage after stage 0
reads model output as an authorization input.

| # | Stage | Fails with |
|---|---|---|
| 0 | **Admission.** Envelope well-formed; correlation id present; idempotency key present when the package requires one; deadline in the future; request size within bounds. | `StaleOrIncompatibleSchema`, `DeadlineExceeded` |
| 1 | **Principal authentication and delegation-chain verification.** The chain is verified end to end; an unverifiable link is a refusal, not a downgrade. | `AuthorizationDenied` |
| 2 | **Capability resolution.** `(id, version)` resolves to a registered package; `package_digest` matches the digest the grant is bound to; package state is `current` at a C-level that permits live execution. | `UnsupportedOrRevokedCapability` |
| 3 | **Schema validation.** Parameters validated against the compiled input schema. This is exactly what `ToolSchemas::validate` does today, moved in front of authorization rather than behind it. | `StaleOrIncompatibleSchema` |
| 4 | **Data-class and privacy check.** Declared parameter data classes against the grant's permitted classes, residency policy, and retention class. | `AuthorizationDenied` |
| 5 | **Authorization.** Deny-by-default grant match over tenant, guild, resource, subject, capability, version range, time window, revocation state, and policy version. Effective-authority intersection per section 5.3. | `AuthorizationDenied` |
| 6 | **Risk and regime gate.** The safety path evaluates independently and may refuse, pause, revoke, cancel, or force a safe state without consulting a model (§3). | `AuthorizationDenied`, `SafetyPause` |
| 7 | **Rate limit and budget.** Rate class; per-guild budgets for speech, observation, planning, external API calls, command installation, and structural changes (§8); tenant cost budget. | `ResourceExhausted`, `RateLimited` |
| 8 | **Deadline and cancellation re-check.** Same cooperative model as `ToolExecutionContext::should_stop`. | `DeadlineExceeded`, `Cancelled` |
| 9 | **Approval check.** Required level computed per section 6.2. If unmet, emit `PendingApproval` with the §3 preview and **stop with no effect**. This is the stage `abi-agent-host` currently collapses into a `Denied` tool result. | `ApprovalRequiredOrExpired` |
| 10 | **Precondition revalidation against freshly observed current state**, immediately before the effect (§4 step 8). Grant revocation re-read here. If platform state changed since the proposal, stop rather than apply a stale operation (§10). | `PlatformPreconditionFailure` |
| 11 | **Admission-ledger write.** Atomically consume the approval and persist execution intent before any effect. This is `prepare_tool_execution` today, extended with capability, grant, and approval-level columns. | `ApprovalRequiredOrExpired`, `Conflict` |
| 12 | **Effect.** The adapter executes, resolving `CredentialRef`s inside its own boundary, under the shared cancellation token and deadline. | `ProviderUnavailableOrUnqualified` |
| 13 | **Postcondition verification** against freshly observed platform state (section 8). | `PlatformPostconditionFailure` |
| 14 | **Outcome-ledger write.** `succeeded | failed | interrupted` plus a result digest. Never raw output. This is `finish`/`interrupt` on the existing execution ledger. | `MemoryUnavailable` |
| 15 | **Compensation** on postcondition failure (section 10). | `RollbackComplete | RollbackPartial | RollbackFailed` |
| 16 | **Receipt and episode.** A redacted receipt returns to the caller; a proposed episode goes to the WDBX selective write gate, which Program 4 owns and which may decline it. | |

Two structural properties the pipeline is designed to make testable:

- **A model-selected capability with no grant never reaches an actuator.** It
  terminates at stage 5. This is §12's named authorization test.
- **Nothing between stage 1 and stage 16 reads generated text.** The only model
  contribution is the proposal admitted at stage 0, and it has already been
  schema-validated by stage 3 before any authority is consulted.

---

## 8. Proposed: postconditions

### 8.1 Predicate kinds

Postconditions are typed predicates evaluated by the actuator, not sentences a
model judges:

- `ResourceExists { selector }`
- `ResourceAbsent { selector }`
- `FieldEquals { selector, field, expected }`
- `FieldWithin { selector, field, range }`
- `PermissionEffective { subject, resource, permission, expected }`
- `CountWithin { selector, range }`
- `DigestMatches { selector, declared_fields, expected_digest }`
- `NoUnintendedDelta { selector, declared_fields, pre_state_digest }`

`NoUnintendedDelta` is the one that catches a correct-looking effect with a wrong
blast radius: the actuator digests the declared observed fields before the effect
and requires everything outside the intended change to be unchanged after it.

### 8.2 Observation and freshness

Every predicate names its observation source and a freshness bound. A predicate
evaluated against a cached read is not evaluated. A predicate whose source is
unavailable within the bound yields a third outcome:

```
PostconditionOutcome { Satisfied, Failed, Unverifiable { reason_code } }
```

`Unverifiable` is not success. It appears in the receipt, it appears in the audit
record per predicate, and it blocks a `Succeeded` terminal state: an execution
with any `Unverifiable` postcondition terminates as `SucceededUnverified`, which
is a distinct receipt state the operator can see. Collapsing `Unverifiable` into
`Satisfied` would be exactly the "integrity is not truth" error §5 forbids,
transposed onto outcomes.

### 8.3 Registration rule

Compilation rejects a `SideEffectClass::PlatformWrite` package with no
postconditions (section 2.3, rule 4). A capability that writes to a platform and
cannot say what should then be true is not a capability, it is a hope.

---

## 9. Proposed: the audit record

### 9.1 Coverage

One record per stage outcome that terminates or advances a decision, plus one
terminal record per attempt. **Refusals are audited with the same completeness as
allows.** An attempt that produced no record is a defect, and the acceptance
matrix tests for it explicitly.

### 9.2 Fields

Extending `abbey`'s `NewAuditEvent { run_id, action, outcome, metadata }` and
`abi-agent-runtime`'s `AuditEntry { policy, call_id, tool, decision }`:

```
CapabilityAuditRecord {
    attempt_id, correlation_id, idempotency_key_digest,

    tenant_id, guild_ref,            // pseudonymous
    principal_ref,                   // pseudonymous
    delegation_chain_digest,

    capability_id, capability_version, package_digest,
    grant_id: Option<GrantId>,
    policy_version, guild_constitution_version, safety_policy_version,
    schema_version,

    stage,                           // which pipeline stage produced this record
    decision,                        // allow | approval_required | deny | pause
    reason_code,                     // closed enum, never free text

    parameter_data_classes,          // classes only
    parameter_digest,                // never raw parameters

    approval_level_required, approval_level_satisfied,
    decision_id: Option<String>, approver_ref: Option<PrincipalRef>,

    rate_class, budgets_consumed,
    deadline_ms, elapsed_ms, cancellation_state,

    outcome_state, result_digest: Option<Digest>,
    postconditions: Vec<(PredicateId, PostconditionOutcome)>,
    compensation_state,

    retention_class,                 // §5
    record_digest, signer_identity,
}
```

### 9.3 Hard rules

1. **No raw parameters, no raw outputs, no credentials, no message content, no
   transcripts, no participant identities, no raw audio.** Digests and classes
   only. `abbey`'s `tool_approval.rs` and `tool_execution.rs` already hold this
   line and their doc comments say so; the kernel does not relax it.

2. **Bounds are inherited, not reinvented.** `AuditMetadata`'s existing ceilings
   apply: 4096 bytes total, 512 bytes per string, 32 collection items, depth 4,
   object-only, and `validate_audit_label`'s 64-byte control-character-free
   labels.

3. **`reason_code` is a closed enum.** Today `PolicyDecision::Deny { reason:
   String }` carries free text into `AuditEntry` and, through
   `host.rs:241-246`, straight back to the model. Program 2 splits the two: a
   `reason_code` for the audit record and the operator, and a separately
   authored, bounded, model-visible message that is not permitted to name a
   protected resource. §3 requires that a denial "identifies the failed invariant
   without exposing protected data."

4. **Audit write failure fails the attempt closed.** `runtime_v3.rs` already maps
   a failed `record_tool_audit` to `internal_failure()`. Keep that.

5. **The sink is bounded.** `MemoryAuditSink` has no ceiling and is documented as
   the caller's responsibility to drain. The kernel's production sink is the
   `abbey` SQLite ledger; the in-memory sink stays a test fixture and gains an
   explicit ceiling so a long-lived process cannot grow one.

6. **Retention is classed at write time** (§5, decision 28): `Ephemeral` records
   are never written; `Session`, `Operational`, `Durable`, and
   `MandatoryIncident` carry their class and a deletion-key reference.

---

## 10. Proposed: rollback semantics

### 10.1 What "rollback" honestly means here

This is **compensation**, not transactional rollback. §8 states plainly that
"Discord does not provide a transaction spanning multiple structural calls." No
mechanism in this program can make a sequence of platform writes atomic. What it
can do is capture pre-state, attempt a declared inverse, and report honestly.

### 10.2 Declaration

`Reversibility::Reversible { compensator: CapabilityId, pre_state: PreStateSpec }`
names the compensating capability and the exact fields whose pre-effect values
must be captured. Compilation refuses a `Reversible` package whose compensator is
unregistered or declares a higher risk class (section 2.3, rule 3).

`ReversibleWithLoss` declares that the inverse restores the resource but not some
named property (ordering, timestamps, identifiers). The receipt must name the
loss.

`Irreversible` declares there is no inverse. An `Irreversible` capability may not
execute below `A4Owner` and may never execute on the strength of a learned
policy.

### 10.3 Pre-state capture is a precondition

For a `Reversible` capability, the actuator captures and digests the declared
pre-state fields at stage 10, inside the same freshness bound as precondition
revalidation, and **refuses the call if capture fails**. A reversible capability
executed without a captured pre-state is not reversible, and calling it so would
be a false claim of the kind §11 exists to prevent.

### 10.4 Trigger and authority

Compensation triggers on a `Failed` postcondition at stage 13, on an adapter
error after a partial effect, or on an operator `undo`.

- Compensation runs under the original grant's scope. It does not require a fresh
  human approval when the package declares `compensation_preauthorized = true`
  **and** the compensator's risk class is at or below the forward capability's.
  Otherwise it raises an approval at the same level as the forward call.
- Compensation is itself a full actuation: its own admission-ledger row, its own
  audit records, its own postconditions. It can fail, and the design assumes it
  sometimes will.

### 10.5 Multi-step plans

Compensate in reverse order. **Stop at the first compensation failure** rather
than continuing, per §8 step 9 ("continue, compensate, or stop safely"). Blindly
continuing past a failed compensator is how a partial rollback becomes a worse
state than the partial forward run.

### 10.6 Terminal receipt states

```
RollbackState {
    NotApplicable,                  // no effect occurred
    RolledBack,                     // every step reverted and verified
    PartiallyRolledBack { completed, reverted, unresolved },
    RollbackFailed { completed, reverted, unresolved, reason_code },
    CompensationUnavailable,        // capability declared Irreversible
}
```

§10 requires that "if a rollback is incomplete, the receipt identifies completed,
reverted, and unresolved steps without exposing private content." The
`completed / reverted / unresolved` triple carries capability ids and resource
selectors in pseudonymous form, never parameter values.

### 10.7 Interrupted executions are never auto-anything

An `Interrupted` execution record, which the existing ledger already produces on
daemon reopen, is never auto-retried and never auto-compensated. The effect may
or may not have happened and the runtime cannot tell. `tool_execution.rs`'s own
documented behavior applies: "Retrying an ambiguous effect therefore requires a
fresh call and approval." An `Interrupted` row surfaces in the receipt and to the
operator as unresolved.

---

## 11. Deliverable scope, canary boundary, and rollback path for the program itself

### 11.1 What ships in Program 2

Per §13, against recording adapters, with no production Discord mutation
authority:

1. `abi-capability`: capability types, registry compilation, grant types, the
   authorization kernel, the approval-level function, the typed error taxonomy.
2. A `RecordingActuator` that runs stages 0 through 11, records the exact
   platform call it *would* have made, evaluates postconditions against a
   recorded fixture platform state, and performs no platform write.
3. Extension of `abbey`'s existing tables with capability, grant, principal,
   approval-level, and postcondition columns, additively.
4. Extension of the audit record to section 9.2, additively.
5. Guild grants and Guild Constitution narrowing: compiled, evaluated, tested,
   bound to the recording actuator.
6. Credential `CredentialRef` indirection and the tenant-bound resolver, with the
   existing `abi_foundation::credentials` store behind it as the single-tenant
   case.
7. Cross-language fixtures for the grant, approval, receipt, and audit envelopes,
   feeding Program 1's contracts package.

### 11.2 The one live-effect binding permitted

`abbey`'s existing local `abbey_memory_mark_obsolete` path, re-expressed through
the capability runtime **with no new authority**: same owner-only authenticated
socket, same single mutating descriptor, same digest-bound approval, same local
memory effect. It is the only place the kernel is wired to something that
actually changes state, and it changes state that `abbey` already owns.

Verified in `abbey/src/daemon/runtime_v3/tool_catalog.rs`: `build()` binds every
descriptor from `crate::mcp_host::v3_descriptors()` / `v3_specs()` with
`ToolRoute::DirectReadOnly`, and appends exactly one further tool,
`MEMORY_MARK_OBSOLETE_TOOL_ID = "abbey_memory_mark_obsolete"`, only when
`ACTIVE == Edition::Safe`. That one carries `V3ToolEffect::Mutating`,
`ToolRoute::ApprovalRequired`, and an input schema accepting a single
`record_id` string bounded at 1 to 128 bytes matching `^[A-Za-z0-9._:-]+$` with
`additionalProperties: false`. Program 2 does not widen that schema, does not add
a second mutating descriptor, and does not change the edition gate.

### 11.3 What explicitly does not ship

- No Discord mutation adapter is registered. `abbey-bot` gains a read-only
  capability projection only. Structural guild writes are Program 5.
- No dynamic command registration. §7: "Dynamic command registration is not an
  LLM side effect."
- No WDBX canonical episode write. Program 4 owns the write gate; Program 2
  produces proposed episodes that the gate may decline.
- No model-routing change. Program 6 owns the arbiter and the model registry.

### 11.4 Rollback path for the program

The kernel is additive at every seam. `abi-agent-host` keeps its existing
`ExecutionPolicy` and `ToolExecutor` traits unchanged; the kernel is installed as
an implementation of both. Reverting the program means reinstalling
`EffectScopedPolicy` and the previous executor, and the only schema change to
leave behind is additive columns and additive tables, which the previous code
does not read. There is no dual canonical writer at any point (§5, decision 77).

---

## 12. Conflicts and discrepancies found

Each of these is a place where either the brief, the constitution, or existing
source disagrees with something else. None is resolved unilaterally here.

**C1. The brief names an "L0-L8 evidence ladder." The constitution's normative
ladder is C0 to C7 (§11).** No L-ladder appears anywhere in the constitution.
The brief's "L1 schema-validated / L2 deterministic qualification / L3 sandbox"
maps cleanly onto §6's numbered promotion stages 1, 2, and 3, and section 3.2
above states that mapping explicitly. This spec uses C0 to C7 as normative. If
Donald intends a separate L-ladder, §15 requires it be added to the constitution
first.

**C2. `EffectScopedPolicy` grants authority from a self-declaration, not a
grant.** `abi-agent-runtime/src/policy.rs:108-120` allows any `ToolEffect::ReadOnly`
tool outright, and its own doc comment says it "trusts it and does not verify
it." That is authority derived from the tool author's claim. Constitution §3
requires a grant, and decision 11 requires deny-by-default. Program 2 closes this
by removing `ToolEffect` from the authorization path (section 2.4). Until it
does, any live deployment of `EffectScopedPolicy` is allowing calls without a
grant.

**C3. `abi-agent-host` converts "approval required" into "denied."**
`host.rs:235-240` maps `PolicyDecision::RequireConfirmation` into a `ToolResult`
with `ToolStatus::Denied` and payload `"confirmation required: {reason}"`, then
continues the model loop. §3 requires a distinct outcome: "Approval required:
Abbey renders a bounded preview, expected effects, uncertainty, risk, expiration,
and rollback path." Telling the model "denied" is both semantically wrong and
operationally dead-ended, since no approval channel exists in the host. Section
7 stage 9 is the fix.

**C4. `abbey-bot/src/tools.rs` publishes a JSON Schema it does not enforce.** The
`ToolSpec.parameters` schema is rendered into both provider wire shapes, but
`dispatch` validates by hand (`arg_str`, a `clamp` on `limit`) and returns a
plain string for a missing argument or an unknown tool. `abi-agent-host`'s
`schema.rs` does validate. §12 requires schema validation on the authorization
path. The asymmetry means the same logical tool is schema-enforced through ABI
and not through the Discord adapter.

**C5. `AuditEntry` cannot support §11's claim record or §12's authorization
tests.** It carries `{ policy, call_id, tool, decision }`. There is no principal,
no scope, no capability version, no outcome, no digest, no policy version. And
`MemoryAuditSink` has no ceiling, which its own doc acknowledges. Section 9
extends both.

**C6. Denial reasons are free text and flow back to the model.**
`PolicyDecision::Deny { reason: String }` reaches both the audit entry and, via
`ToolResult::with_limit`, the model. §3 requires a refusal that does not expose
protected data. Section 9.3 rule 3 splits `reason_code` from the model-visible
message.

**C7. No credential has a tenant dimension.** `abi_foundation::credentials` is
one set per machine user, keychain on macOS only, file elsewhere. "Per-tenant
credential isolation" is entirely new. Any claim that isolation exists today
would be false.

**C8. `GuildSettings.learning_enabled` defaults to `true`.** Verified at
`abbey-bot/src/guild.rs:104`. Constitution decision 31 requires adaptive learning
to be opt-in and default-off. §8 already identifies this mismatch and assigns the
migration to the implementation program, with the constraint that it must not
silently rewrite an existing guild's explicit choice. Program 2 does not own that
migration (it is guild adaptation, section 8, not authorization), but the grant
model must not be built on the assumption that learning is off.

**C9. One negotiated capability set currently carries both `InvokeTools` and
`DecideToolApprovals`.** Verified in code, not in prose:
`abbey/src/daemon/runtime_v3.rs:102-113` builds the available list, pushing
`V3Capability::InvokeTools` whenever the bound tool catalog is non-empty and
pushing `V3Capability::DecideToolApprovals` and `V3Capability::CancelTools`
whenever any bound tool carries `ToolRoute::ApprovalRequired`. There is no
principal partition between the two: one `V3CapabilitySet` holds both, and
`src/daemon/client/v3.rs:76` and `:167` gate invoke and decide against that same
set. This is coherent for a single machine-owner principal acting on their own
local state over an owner-only authenticated socket, and I do not read it as a
defect today. It becomes self-approval the moment grants are multi-principal.
Section 6.5 blocks it at A2 and above.

**C10. The brief's scope exceeds §13's Program 2 ceiling.** The brief asks for
guild grants, actuator validation, postconditions, and rollback; §13 scopes
Program 2 to "recording adapters" with "no production Discord mutation
authority." This spec designs all of it and ships the binding against recording
adapters only (section 11). If Donald wants live guild mutation inside Program 2,
that is an §15 amendment, not a design call.

**C11. Sibling program specs written the same day carry titles that do not map
one-to-one onto §13's program list.** Observed on 2026-08-22 in
`docs/superpowers/specs/`, from filenames and title lines only:
`2026-08-22-program-3-canonical-wdbx-episodic-contract.md`,
`2026-08-22-program-4-guild-world-model-and-arbiter.md`,
`2026-08-22-program-5-discord-organization-slice.md`,
`2026-08-22-program-6-abbey-api-federation.md`, and
`2026-08-22-program-7-learning-evaluation-promotion.md`. I did not open their
bodies, so I make no claim about their contents or about which §13 entry each
intends. The observation is only that the numbering does not line up, which makes
any cross-program reference by number unsafe until the set is reconciled. This
document uses §13's numbering throughout; if the intended numbering is the new
one, §13 needs an §15 amendment rather than a silent renumber.

---

## 13. Acceptance matrix

Derived from §12. Every row is a test this program must pass before any claim
above C1 is made for the kernel.

### Contract

| Test | Passes when |
|---|---|
| Golden envelope round-trip | Rust and Swift encode and decode the same grant, approval, receipt, and audit envelopes byte for byte |
| Additive-field compatibility | Unknown additive fields round-trip or fail per the declared compatibility policy, deterministically |
| Canonical digest agreement | `package_digest`, `call_digest`, and `record_digest` match across languages |
| Malformed envelope | Invalid, duplicate, stale, oversized, and contradictory envelopes fail deterministically, never partially |

### Authorization and capability

| Test | Passes when |
|---|---|
| Principal matrix | Owner, administrator, manager, member, bot, revoked, and delegated principals each get the specified decision for each risk class |
| Hierarchy change mid-flight | A Discord permission change between proposal and execution stops the call at stage 10 |
| Expired grant | An expired grant denies at stage 5, and the audit record names `grant_expired` |
| Approval replay | A reused `decision_id` is refused (the existing `UNIQUE` constraint), and the refusal is audited |
| Capability version drift | A recompiled package with a different `package_digest` does not satisfy an existing grant |
| Revoked package | A revoked version denies, and the previous approved version remains usable |
| Idempotent retry | The same idempotency key does not produce a second effect |
| Partial effect and compensation | A forced adapter failure after a partial effect produces `PartiallyRolledBack` naming completed, reverted, and unresolved steps |
| Postcondition failure | A forced postcondition failure triggers compensation and a `RollbackComplete` or `RollbackFailed` receipt, never `Succeeded` |
| **No grant, no actuator** | A model-selected capability with no grant terminates at stage 5 and the recording actuator records zero calls |
| Guild constitution widening | A guild-constitution entry outside the compiled ceiling is ignored and reported, exactly as `os/policy.rs` does |
| Cross-guild grant | A grant scoped to guild G authorizes nothing in guild H |
| Self-approval | At A2 and above, an approval whose `approver_ref` equals the proposal's `principal_ref` is refused |

### Privacy

| Test | Passes when |
|---|---|
| Raw material exclusion | Raw parameters, outputs, credentials, transcripts, message content, and identifiers appear in no audit record, receipt, or diagnostic log |
| Credential cross-tenant | A `CredentialRef` for tenant A returns `CredentialNotAvailable` under tenant B, with no fallback |
| Redaction pass | The value-based redaction pass runs over every receipt and audit record, with `MIN_REDACTABLE_SECRET_BYTES` restated in the test |
| Retention expiry | An `Operational` record's expiry removes payload and every derived projection |
| Refusal disclosure | A denial names the failed invariant without naming a protected resource |

### Failure and recovery

| Test | Passes when |
|---|---|
| Crash after prepare | A failpoint after `prepare_tool_execution` yields `Interrupted` on reopen, no fabricated result digest, and a refusal to replay the consumed call |
| Crash after effect | The same, and a fresh call plus a fresh approval is required even though the effect may already have happened |
| Audit sink failure | A failed audit write fails the attempt closed; no effect occurs |
| Kernel unavailable | With the authorization kernel unavailable, consequential execution is denied (§10) |
| Cancellation race | Cancellation observed between stages 9 and 11 produces a `Cancelled` receipt with no effect |
| No silent fresh start | A corrupt ledger is a startup error, never a silent empty state |

---

## 14. Open questions for Donald

1. **C1 above:** is "L0-L8" a distinct ladder you intend, or shorthand for the
   §6 promotion stages? This spec assumed the latter and used C0 to C7.
2. **C10 above:** does Program 2 stay bound to recording adapters, or do you want
   an §15 amendment permitting a narrow live guild-write canary inside it?
3. **Tenant definition.** Is a tenant a machine user, a Donald-owned deployment,
   or a paying customer? Section 4 assumes the deployment, which makes today's
   single-credential store the single-tenant case. A customer-level tenant
   changes the credential store design materially.
4. **Approval surface ownership.** §2 assigns "approvals and UI" to the Swift
   `AbbeyBot` and "guild policy and command shell" to `abbey-bot`. Which surface
   renders the A2-through-A5 preview for a guild action, and does the answer
   differ for an action proposed in Discord but approved on the Mac?
5. **`RiskClass::Prohibited`.** Do you want a representable prohibited class that
   the issuer refuses, or should a prohibited action simply have no package? The
   representable version makes the refusal auditable; the absent version makes it
   unreachable. Section 2.2 chose representable.

---

## 15. Evidence statement

This document is C0 (Specified) for the mechanisms it proposes. It contains:

- **Observations:** every statement in section 1 and every source reference
  elsewhere, each verified by reading the named file on 2026-08-22.
- **Proposed criteria:** everything in sections 2 through 11 and 13.
- **Inferences:** the conflict analysis in section 12, which depends on reading
  the constitution's requirements against the observed source.

No mechanism proposed here has been implemented, tested, replayed, evaluated,
shadowed, canaried, or witnessed. Writing this document promotes nothing.
