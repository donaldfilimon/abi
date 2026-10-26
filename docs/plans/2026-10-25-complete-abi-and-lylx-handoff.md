# Complete ABI, then qualify its LYLX migration

Date: 2026-10-25. Status: proposed execution plan; implementation and release
requirements remain open. Audience: ABI implementers, Abbey/WDBX owners, LYLX
integrators and release operators.

## 1. Decisions, scope and current baseline

Donald selected **both, staged** and **full product** in the planning conversation:
finish the retained ABI product, then port and qualify its capabilities in LYLX.
This plan extends the [approved ABI completion plan](../superpowers/plans/2026-10-23-complete-abi.md)
and preserves all [C01–C44 requirements](../verification/2026-10-23-completion-matrix.md).
The [production ecosystem design](../superpowers/plans/2026-10-22-abi-production-ecosystem.md)
and adopted recovery/process contracts remain the implementation baseline.

Two unanswered planning choices use explicit defaults: original-model development
stays ABI-first, and compute is local-first with no assumed paid resources. These
defaults do not authorize new training runs, purchases or deployment. LYLX's
Quesar requirements, original source material and frozen numerical thresholds
remain independent; an ABI model does not automatically become qualified Quesar.

The product remains an open-signup, best-effort research service with private
user-hosted inference/memory, governed tools, original trained weights, browser
account management and desktop CLI/workers. Mobile, billing, a public compute
marketplace, production FHE and additional accelerator qualification are outside
this release. There is no availability SLA.

Platform delivery retains the prior decision: build for macOS, Linux and Windows;
perform immediate runtime acceptance on the current Mac. Publish Linux/Windows
build results separately. Native runtime acceptance is required before either
platform is advertised as runtime-qualified; its absence cannot be hidden inside
a blanket cross-platform completion claim.

### Observed baseline

- Planning began at `69910bf` with substantial existing work. A later read found
  clean local `main` at `89f950e9a851e9ced6f5361f6f3d82a87c3d35e3`, one commit
  ahead and one behind the locally known `origin/main`. That checkpoint was
  created outside this planning task; remote freshness and publication are unverified.
- The isolated planning `./tools/check.sh` attempt exited 1 at the required audit.
  Its 186 Python policy tests passed with eight skips, the CI contract passed,
  and both Abbey oracles verified 113 artifacts. Later gate stages did not run.
- The lockfile still selects Tokio 1.53.1. The scanner rejects it under
  RUSTSEC-2026-0386; the fetched advisory identifies 1.53.3 or later as patched
  on this release line. This is the first executable blocker, not a reason to
  expand audit exceptions. The planning attempt is not a frozen-source receipt.
- Durable worker/outbox recovery and a closed Unix `pwd` adapter have existing
  source and historical local evidence. The real Abbey durable-owner join,
  general adapters, privacy lifecycle and operational acceptance remain open.
- `abi-nn` currently describes a miniature character-level demo; the local model
  runtime implements the `abi-bigram-v1` fixture. Neither completes the original
  transformer/model requirements.
- The company register records bounded `ABI-POLICY` and `ABI-WORKFLOW` ports as
  implemented with `replacement_qualified=false`; other migration entries remain
  queued. These records are inputs to Stage B, not current replacement proof.

## 2. Stage A — complete the retained ABI product

Use canonical checkouts and preserve existing work. Before each source slice,
record writer ownership, revisions, dirty/untracked inputs, consumed sibling
sources, toolchain and feature scope. Use isolated build directories, scratch
stores and ports. Do not merge the divergent branches, update dependency pins or
commit another owner's files as incidental cleanup.

ABI owns capabilities, execution, models, workers, connectors and the public
control plane. Abbey owns conversations, human approvals and durable execution
lifecycle. WDBX owns persistent memory, retention, deletion and key custody.
Implement each change in its owning repository and consume a qualified source
handoff; do not create competing approval or deletion authorities inside ABI.

### A1. Restore a reproducible baseline — C01, C02, C42, C44

- Preserve the existing checkpoint and reconcile the local/remote documentation
  divergence without discarding either history. Record the resulting source set.
- Update the affected lockfile to patched Tokio on the compatible release line,
  starting with 1.53.3; inspect transitive changes and any current advisories.
  Coordinate consumer lockfiles with their owners. Keep existing audit policy.
- Run ABI's complete gate, WDBX's own gate and Abbey root/desktop gates on stable
  inputs. Resolve consumer lock/source mismatches before promoting any WDBX pin.
- Reconcile C01–C44 into one execution ledger with owner, dependency, status,
  test command, actual result, evidence identity and remaining external input.
  Retain old receipts as dated evidence; remove conflicting present-tense claims.

Exit: actual full local gate results bound to the consumed source set, with all
remaining failures or external gaps individually recorded.

### A2. Finish durable execution and relay recovery — C03–C06

- Requalify the existing file history, encrypted outbox and recorded-result paths.
  Keep admission authority separate from result redelivery authority.
- Recheck cancellation, revocation, trusted time and remaining lease immediately
  before provider/adapter dispatch, including after authority/audit callbacks.
- Cover short/zero writes, sync failure, corrupt/truncated records, failed terminal
  persistence, process kill, lost responses, duplicate delivery and exhausted quotas.
- Preserve recorded results and exact encrypted packets across reopen. An uncertain
  possible effect stays unresolved; recovery must not authorize another dispatch.

Exit: invocation counters and durable histories prove no unauthorized or repeated
effect across each fault barrier; accepted stream ordering and bounds survive recovery.

### A3. Join real approvals to process and file operations — C07–C09

- Integrate the sealed process adapter with Abbey's real `RuntimeStore` and local
  approval authority. Keep in-memory ledger tests as component coverage only.
- Retain the closed startup-pinned process design. Public input cannot choose a
  shell, executable, argv, environment or working directory outside its bound plan.
- Extend governed files to bounded reads and exact create/replace/delete operations
  within an approved workspace. Bind canonical resource identity, expected prior
  content and requested change; replacement races and changed inputs invalidate approval.
- Persist approval consumption and execution intent before effects; verify actual
  filesystem/process postconditions and bounded child cleanup. Failed observation
  or cleanup cannot produce a successful receipt or invent a compensation result.

Exit: real-owner restart tests cover approve, deny, revoke, mutate, cancel and
replay, with unchanged effect counts after reopening completed or uncertain work.

### A4. Complete governed desktop and connector journeys — C10–C12

- Add a finite capability catalog: inspect the active application, capture an
  explicitly selected screen/window, activate an approved application, and perform
  an exact click or text entry against a verified target. Use native permission
  checks; changed focus/target or unavailable platform support refuses execution.
- Route Discord message sends and Twilio SMS/calls through the same lifecycle.
  Exact approval binds account, recipient, content/action, scope and expiry.
  Store delivery uncertainty durably; never blindly retry an ambiguous send.
- Finish OpenAI/Anthropic/Grok request paths with explicit transmission consent,
  bounded streaming, cancellation, secret redaction and defined provider errors.
  Keep `connector_test` deterministic and local.
- First exercise protocol fixtures and scratch resources, then designated live
  test accounts when available under their existing execution authorization.

Exit: every advertised operation has positive, refusal and recovery acceptance;
live acceptance names the actual account/environment and never exposes its secrets.

### A5. Complete memory privacy and credential lifecycle — C13–C17

- Inventory active records, segments/generations, replicas, backups, summaries,
  histories, outboxes, exports and training derivatives. Assign retention and
  deletion ownership to every retained copy.
- In WDBX, provide durable erasure/retention progress and key provision, rotation,
  revocation, recovery and destruction contracts. ABI/Abbey consume those receipts.
  Preserve historical event identities and distinguish logical refusal, key
  destruction and physical cleanup; none implies removal from an unobserved backup.
- Enforce tenant/member scope and current revocation before retrieval ranking,
  prompt assembly and export. Interrupted deletion resumes; offline replicas and
  restores reconcile tombstones/key revocation before serving old data.
- Replace user-node demonstration key files with supported platform credential
  custody. Missing or locked backends refuse without falling back to weaker storage.
- Reconcile older erasure proposals with current owner contracts before adapting
  them; do not silently adopt historical algorithms, ownership or approval assumptions.

Exit: restart, offline-replica and restore tests cannot resurrect retrievable
deleted content; each receipt reports incomplete/offline obligations honestly.

### A6. Finish accounts, private routing and browser flows — C18–C22

- Complete real configured OIDC acceptance, JWKS refresh/rotation, session
  revocation, node credential renewal/rotation and local recovery. Preserve exact
  pairing approval and tenant-bound identity; enrollment never approves future tools.
- Preserve existing account API and versioned relay contracts. Keep the service
  behind the controlled HTTPS proxy; add bounded proxy/abuse controls and audited
  metadata archival before public operation.
- Finish browser signup/sign-in, pairing, status, cancel acknowledgment, revocation,
  deletion and restart/reconnect journeys. Prompt/result plaintext stays with the
  native client/user node; approval remains local to Abbey's authority.
- Retain default limits of two active and ten queued jobs per account, explicit
  queue-full refusal and metadata-only central persistence. Operator public
  training/evaluation jobs use a distinct namespace and admission capability;
  tenant private jobs cannot be routed into that pool.

Exit: two isolated tenants complete the journeys, including cross-tenant attacks,
key rotation, queue exhaustion, interruption and response loss.

### A7. Implement the real Rust model pipeline — C23–C28, C31

- Add a versioned decoder-only causal transformer alongside the demo path using
  the existing Rust/Candle, Tokenizers and Safetensors stack. Use a common model
  definition for training and inference; retain fixture compatibility explicitly.
- Implement causal attention, positional encoding, normalization, feed-forward
  layers, stable cross-entropy, gradients, clipping and AdamW. Validate numerical
  behavior against independent small reference cases before scaling.
- Version architecture/configuration, tokenizer/template and checkpoints. A
  resumable checkpoint includes weights, optimizer/schedule, RNG, data order/cursor,
  accumulation state and source/data identities. Commit it atomically and reject
  corrupt, incompatible or incompletely committed artifacts.
- Admit licensed/provenance-backed data, deduplicate before splitting and preserve
  related-family separation across all snapshots, including revoked history.
  Private user stores/conversations are excluded from the training corpus.
- Freeze the evaluation corpus, scorer and quality acceptance packet before pilots:
  language, knowledge, reasoning, writing, coding, tools, refusal, privacy and persona.
  Numerical correctness, quality and authority tests are separate release gates.
- Extend explicit CPU/Metal loading, bounded generation and cancellation to the
  real architecture. Releases bind dataset, configuration, tokenizer, checkpoint,
  evaluation and executable digests; promotion and rollback verify those identities.

Exit: real parameter updates, exact same-backend deterministic resume, independently
checked CPU/Metal numerical agreement and tokenizer/runtime parity. Useful model
quality remains open until A8; fixture output cannot close these requirements.

### A8. Train and qualify original weights — C29, C30

- Prepare concrete approximately 100M/300M/1B pilot configurations and a local
  capacity report. Before each resource-consuming run, bind admitted data, model,
  hardware, memory/storage, duration/token cap and stop conditions in its run sheet.
- Use only an explicitly available resource envelope. A candidate exceeding local
  limits produces a measured additional-resource proposal; it is not silently
  trained remotely or treated as a passed pilot.
- Record loss, held-out scores, throughput, wall time, peak memory, cost and energy
  where measurable. Preserve failed runs and negative quality results.
- Select the largest candidate that fits the approved measured envelope and passes
  the frozen quality gates. If none qualifies, keep the requirement open and report
  the failing dimensions; do not lower gates after inspecting evaluation results.
- Complete the selected training run, independently evaluate its final artifact,
  exercise user-hosted conversations/tools and verify promotion/rollback receipts.

Exit: actual original weights with reproducible identity, data rights, measured
quality and stated limitations. Dataset licenses, quality thresholds and resource
envelope are required release inputs, not facts established by this plan.

### A9. Deliver desktop packages and the installed journey — C32–C35

- Package CLI, MCP, worker and required native libraries with exact version/build
  identity, checksums, signing where applicable, install/upgrade/rollback/removal.
- Provision Windows C SDK/linker and a usable Linux build environment. Rust target
  installation alone does not complete either build. Implement equivalent private
  file/credential protections before advertising those platform paths.
- On the current Mac, test a fresh installed package through account signup,
  pairing, original-model load, conversation, authorized recall, approve/refuse,
  cancellation, process restart and recovery. Test interrupted upgrade and rollback.
- Preserve the 14 CLI commands, 12 MCP tools, golden outputs and launcher-based
  stdio behavior. Publish CPU/Metal execution evidence separately from build-only
  Linux/Windows results and unverified CUDA/Vulkan/ANE paths.

Exit: usable Mac delivery plus reproducible desktop build artifacts and explicit
platform qualification labels. Native Linux/Windows acceptance remains a separately
tracked prerequisite for corresponding runtime support claims.

### A10. Qualify operation and recovery — C36–C41

- Exercise actual separate hosts for worker loss, partition, reconnect, stale
  credential, revocation and backup/restore. Local multi-process tests remain
  useful controls but do not replace these runs.
- Add content-free health, queue, authentication-failure, job-outcome and resource
  metrics. Test alerts and bounded audit retention without recording private payloads.
- Execute key compromise/rotation, tenant revocation, restore, upgrade/rollback and
  emergency shutdown runbooks against staging. Restore must preserve deletion,
  revocation and uncertain-execution history.
- Obtain independent assessment, resolve blocking findings and run the staging
  user journeys before authorized single-region public rollout. Credentials, real
  IdP configuration, extra hosts and assessor access remain named dependencies.

Exit: reproducible operational and security receipts for the deployed artifact,
with tested rollback and no unresolved release-blocking findings.

### A11. Close and publish the release — C43

- Freeze final ABI/WDBX/Abbey revisions, consumed dependencies and artifacts; run
  all required gates on that exact set. Promote the WDBX pin only after its exact
  revision passes, then rerun consumer qualification.
- Resume CI and publish only under their existing authorization boundaries. Verify
  the published commit, actual check results, signatures and artifact checksums.
- Publish installation/recovery instructions, model/data/evaluation identities,
  measured performance, platform limits, runbooks and the C01–C44 evidence index.
- Separate three verdicts: source-qualified, Mac-delivery-qualified and
  operational-release-qualified. Close Stage A only when all requirements for the
  declared release scope have current evidence; never relabel missing work as done.

Execution order: A1 first; A2 then A3; A4 and A6 build on the governed/recovery
contracts. A5 and A7 may advance independently once their owner inputs are stable.
A8 follows A7; A9 joins A3–A8; A10 joins A5/A6/A9; A11 closes the exact release.
An unavailable external input blocks its gate, not unrelated local source work.

## 3. Stage B — migrate qualified ABI capabilities into LYLX

Use the [company consolidation register](../../../../docs/consolidation/register.json)
and LYLX's existing requirements/claims registers. Stage B begins from the Stage A
source/artifact handoff; preserve already implemented LYLX work and compare it
before adding another port. This stage closes the ABI migration obligations, not
every independent LYLX product or Quesar training requirement.

| Order | Work and owner | Acceptance |
| --- | --- | --- |
| B1 | ABI/LYLX: freeze independent source, executable and fixture oracles; map C01–C44 to relevant LYLX requirement IDs and company capability entries. | Every migrated behavior has provenance, a selected compatibility contract and an independent expected result. |
| B2 | LYLX authority/workflow: qualify existing `ABI-POLICY` and `ABI-WORKFLOW` work, then port missing behavior. | Default-deny, exact scopes, approval mutation, durable recovery and uncertain-effect non-replay pass. Native independent approval and commit-bound consumption remain authoritative; legacy eager/no-refund or approval-free behavior is explicitly incompatible. |
| B3 | LYLX runtime: port `ABI-WORKER`, bounded scheduling, cancellation, relay/recovery and governed adapter integration. | Lease expiry, delayed callbacks, revocation, duplicate requests, resource accounting and crash recovery preserve the selected contract without adding a second authority. |
| B4 | Storage/operator owners: integrate required `WDBX-STORAGE`, `ABBEY-OPERATOR`, `ABBEY-DAEMON` and `ABBEY-DESKTOP` contracts; retain `PERSONALITIES` identity provenance. | Historical storage/CLI/protocol behavior is independently tested; unsupported behavior is versioned/refused. Persona selection changes neither facts nor permissions. |
| B5 | Model owners: selectively port reusable ABI training/runtime pieces and versioned artifact adapters. | LYLX's original Quesar numerical, provenance, designation and quality gates remain intact. ABI training receipts do not auto-promote Quesar or persona claims. |
| B6 | LYLX/release owners: execute native end-to-end parity, consumer cutover and rollback/recovery drills. | Exact-source LYLX verification plus company checks pass; intended-use qualification and migrated consumers precede any replacement claim or retirement review. |

For each port: preserve the independent legacy oracle, observe a failing LYLX
contract, implement the smallest coherent change, run focused and full affected
checks, then attach checksummed evidence to the existing register entry. Preserve
native authority semantics where legacy behavior conflicts; do not weaken them
to achieve superficial parity.

Keep Git histories and source ownership independent. No permanent LYLX runtime
Cargo dependency on a legacy checkout, wholesale history import or unreviewed
store conversion is part of this plan. Legacy repositories remain usable until
replacement qualification, consumer migration and restoration-tested recovery
custody are complete. Retirement is a separate reviewed action.

## 4. Interfaces, verification and completion evidence

### Public interfaces

- Preserve CLI 14/MCP 12 and existing goldens. Add configuration/operations inside
  existing command families or the separate versioned control-plane API. Any
  deliberate frozen-contract change requires coordinated source, fixtures and tests.
- Reuse signed worker envelopes, exact approval bindings and lifecycle terminals.
  Version new service/worker payloads explicitly; old data remains readable or is
  rejected with an explicit unsupported-version result, never silently reinterpreted.
- Add owner-defined retention/erasure/key-lifecycle interfaces and resumable status
  receipts; ABI consumes them rather than redefining WDBX storage authority.
- Add versioned transformer/tokenizer/checkpoint/evaluation manifests while retaining
  the existing fixture format. LYLX migration adapters must identify intentional
  incompatibilities in authority, storage or protocol semantics.

### Required checks

For each behavioral repair, retain the failing regression, repair, relevant package
suite and final full gate. Do not fabricate a historical failing test for source
that already behaves correctly. Tests use scratch stores; never the live user store.

ABI commands use `./tools/cargo.sh`, locked dependency resolution, stable Rust
1.99.1 and stdin EOF for tests. Required acceptance includes:

- `./tools/check.sh < /dev/null`: actual dependency audit, policy/contracts,
  sizes, format, warning-denied Clippy, builds/tests, WDBX cross-language conformance,
  Metal tests/docs, same-system benchmark, rustdoc and release binaries.
- Focused worker/host/control-plane/model suites and browser Node suites; real
  Abbey `RuntimeStore` integration, launcher-based CLI/MCP smoke and installed
  artifact identity checks.
- WDBX's own gate and Abbey root/desktop gates against the captured owner inputs.
- LYLX's exact verification runner with its independently pinned historical CLI
  and hash, fresh evidence directory and recorded command budget; run its existing
  requirement/claim audits and preserve prior failed experiments and thresholds.
- Company boundary/provenance checks and coordination tests when updating the
  migration register. Documentation changes run policy tests, whitespace checks
  and the repository Mintlify validator.

Each receipt records source and dependency identities, toolchain/features,
environment/device, commands, exits, test counts/skips, artifact hashes and known
limits. Source drift invalidates the affected receipt. Compile-only, local
simulation, installed execution, model quality and separate-host operation are
reported independently.

### External inputs and next executable slice

Track these dependencies explicitly: sibling source/lock handoffs; admitted data
and frozen model-quality criteria; measured training resources; Linux/Windows
build and native-test environments; designated live connector accounts and OIDC;
signing identities; separate hosts; assessment access; CI/publication/deployment
authorization. Work to prepare each input stays concrete and reviewable.

The first implementation slice is A1: preserve and reconcile the source set,
repair the denied Tokio dependency, rerun the required audit and complete ABI
gate, then record exact consumer compatibility. The next functional slice is
A3's real Abbey durable-owner/process join after A2 recovery requalification.
This plan itself does not close any C01–C44 or LYLX qualification row.
