# ABI Production Ecosystem Implementation Plan

Status: implementation in progress. Approved by Donald on 2026-10-22.
Execution ledger: `docs/verification/2026-10-22-production-ecosystem.md`.

## Goal and release boundary

Build an open-signup, best-effort research service with a browser client and
native CLI/workers on macOS, Linux, and Windows. Deliver an original general
assistant trained from scratch in Rust, governed desktop and messaging
integrations, and user-hosted inference and memory. No availability SLA,
billing system, or public compute marketplace is included in v1.

## Ownership and architecture

- ABI owns model execution, capability evaluation, bounded agent execution,
  worker protocols and connectors. Abbey owns conversations, human approvals
  and durable execution lifecycle records. WDBX owns persistent memory,
  replication, retention, deletion and key custody.
- Add a Rust public control plane and browser client, deployed as portable
  single-region containers. It stores account, enrollment, routing, quota and
  consent metadata. Prompts, conversations, memory, inference and credentials
  stay on user infrastructure.
- User nodes establish outbound authenticated connections. Operator pools
  accept public training/evaluation jobs only, in a separate namespace.
- Preserve CLI 14 and MCP 12. Put configuration under existing command
  families; version new service/worker protocols separately. Reuse signed
  worker envelopes and lifecycle types.

## Task 1: Freeze and qualify the cohort

- Capture exact ABI/WDBX/Abbey revisions, dirty-file manifests, dependency
  identities, compiler versions and features. Preserve historical receipts.
- Reconcile remaining ledger entries against source; name one owner and
  acceptance test per gap.
- Run all three repository gates, including Abbey root/desktop, on identical
  immutable inputs. Promote pins only after the exact cohort passes.

## Task 2: Public accounts and node enrollment

- Configured OIDC issuer; verified account required before enrollment.
- Tenant-scoped registrations, node keys, revocations, quotas and append-only
  administrative audit. Expiring single-use pairing codes confirmed locally;
  certificates bind tenant, node and capability.
- Outbound authenticated connections route encrypted application payloads
  without persisting content. Default two active jobs/account and ten queued,
  configurable bounded request/output sizes and explicit queue-full errors.
- Browser signup, pairing, status, revocation and deletion. Acceptance: two
  independent tenants enroll and execute isolated jobs through the service.

## Task 3: Worker execution and governance

- Actual transport, scheduler integration, durable job store, bounded executor
  and streamed results. Schedule only onto the tenant's own enrolled nodes.
- Preserve signatures, deadlines, leases, quotas, cancellation, digests and
  replay checks. Reconnect returns prior state, never fresh dispatch authority.
- Abbey consumes exact approval durably; fresh authority/precondition checks
  precede dispatch. Revocation refuses undispatched work. Uncertain possible
  effects remain unresolved and are never automatically repeated.
- Acceptance: inference and scripted tools survive disconnect/restart without
  duplicate dispatch.

## Task 4: Memory and live tools

- Implement retention, member-erasure and key-custody contracts in WDBX;
  specify active data, replicas, generations, backups and training artifacts.
- Scope retrieval before ranking and prompt construction.
- Govern file/process/desktop/Discord/Twilio adapters. Support OpenAI,
  Anthropic and Grok model adapters with explicit transmission consent.
- Approval previews bind account, recipient/resource, operation, parameters,
  scope, expiry and effects. Material changes invalidate approval.
- Secrets stay in user-node platform credential facilities. Local approval
  is required when the browser cannot securely reach its authority.
- Platform adapters share a capability contract; unavailable operations refuse.
  Acceptance uses designated accounts and scratch workspaces on all platforms.

## Task 5: Original-model pipeline in Rust

- Decoder-only causal transformer, versioned tokenizer/checkpoints, optimizer
  state, deterministic ordering and resumable training, all in Rust.
- Curated provenance/rights records; exclude user stores/conversations.
  Deduplicate before splitting and isolate related examples across splits.
- Run approximately 100M/300M/1B parameter pilots. Record configurations,
  tokens, hardware, measurable energy, throughput, losses and evaluation.
- Select largest candidate within an approved measured compute/memory/cost
  envelope; if none fits, stop before full training and report options.
- Freeze broad language/knowledge/reasoning/writing/coding/tool/refusal/
  privacy/persona evaluations before comparison. Publish scores and limits.
- Rust runtime supports selected architecture, tokenizer parity, validated
  checkpoint conversion, bounded generation, cancellation and explicit device.
- Releases bind dataset/config/checkpoint/evaluation/executable digests and
  support promotion/rollback. Acceptance requires real trained weights,
  reproducible receipts and successful user-hosted inference.

## Task 6: Distribution and public operation

- Signed macOS/Linux/Windows CLI/worker packages with upgrade and rollback.
- CPU acceptance everywhere, Metal on supported Macs. Advertise other
  accelerators only after real runtime acceptance.
- Content-free health, queues, authentication failures, job outcomes and
  resource metrics. Runbooks cover key rotation, compromise, revocation,
  recovery, upgrades and emergency shutdown.
- Control-plane backup/restore; separate user-hosted WDBX multi-host recovery.
- Independent security assessment/external public-service testing; resolve
  release blockers before open signup.

## Test and release gates

- Exact-source full ABI/WDBX/Abbey gates; legacy goldens and versioned protocol
  compatibility. Tenant crossing, forged enrollment, expiry/revocation,
  replay, approval mutation, audit failure, quota exhaustion and redaction.
- Crash injection before admission, after approval consumption, during
  dispatch and before terminal persistence. Real separate-host partitions,
  reconnect, worker loss, recovery and retained-data deletion tests.
- Every OS: signup, pair, load original model, converse, recall authorized
  memory, approve/refuse tools, restart and recover. Zero unauthorized or
  replayed effects in acceptance. Publish measured performance and quality.

## Execution boundaries

Work in canonical checkouts and preserve unrelated edits. Spending, full
training, live-account tests, CI resumption, publication and deployment require
explicit execution authorization. Mobile apps, production FHE, verified ANE
residency and additional accelerator backends are outside release completion.
Do not mark the ecosystem complete using fixture/local source evidence.
