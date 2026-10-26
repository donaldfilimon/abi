# ABI completion contract — 2026-10-23

Authority: approved inline Complete ABI Implementation Plan, retained at
`../superpowers/plans/2026-10-23-complete-abi.md`. This contract supersedes older
whole-product completion wording; historical source receipts remain historical.

Status: **ecosystem incomplete**. Q = qualified for explicitly named scope only;
U = existing implementation awaiting current qualification; M = missing;
E = requires external environment, resources or existing execution authorization.
Q local is never equivalent to public, separate-host or cross-platform acceptance.

Platform scope update from Donald (2026-10-23): build support for all desktop
platforms remains required; runtime acceptance for the immediate delivery is on
the current Mac only. Linux/Windows execution is no longer an immediate gate.
Public operation and original-model resource authorization remain separate.

| ID | Requirement | Owner / implementation | Status | Acceptance and remaining evidence |
| --- | --- | --- | --- | --- |
| C01 | Exact cohort and requirement coverage | ABI verification | U | All source hashes/modes, revisions, locked dependencies, compiler/features and gate receipts match |
| C02 | Canonical full gate | ABI tools/check.sh | Q local baseline | Planning run exit 0; `/tmp/abi-plan-baseline-20261023.log`; not a frozen cohort receipt |
| C03 | Durable bounded worker control history | ABI abi-worker/file_store | Q local Unix | Short/zero writes, sync failure, poison, reopen, corruption, lock and record-budget tests; no power-loss proof |
| C04 | Authenticated worker admission and inference | ABI abi-worker | Q local Mac | Signed scope, deadlines, quotas, pre-dispatch cancel, observed chunks and cached retries; no trained model implied |
| C05 | Outbound encrypted transport and stream | ABI abi-control-plane | U | Two isolated tenants, ordered encrypted deltas, explicit cancel acknowledgment |
| C06 | Reconnect and response-loss recovery | ABI worker/control-plane | U | Real process kills and lost TCP ACKs retain packet bytes and provider/effect counts |
| C07 | Durable exact local approval/lifecycle | Abbey owner + ABI governed host | U | Real RuntimeStore join, consumed approvals, denial audits, restart and zero repeat |
| C08 | Governed process supervisor | ABI abi-agent-host/platform_process | Q local Mac adapter | Pinned pwd, bounded supervisor, receipt, host A1/replay and full gate pass; durable-owner integration stays C07 U |
| C09 | Governed files | ABI adapter + Abbey lifecycle | U | Pinned scratch resource, exact effect and trusted postcondition; general adapter coverage pending |
| C10 | Governed desktop | ABI platform adapters | M | Finite operations, native permission checks and real platform refusal/approval/recovery |
| C11 | Governed Discord/Twilio | ABI connectors + Abbey lifecycle | M | Exact recipient approval, durable uncertain delivery, no blind retry; live account E |
| C12 | Consented OpenAI/Anthropic/Grok | ABI connectors | U | Explicit transmission consent, cancellation, bounded diagnostics; live account E |
| C13 | Platform credential custody | WDBX foundation + ABI adapters | U | Actual macOS/Linux/Windows backend runtime; credentials never enter public service |
| C14 | Scope retrieval before ranking/prompt | ABI SEA/CLI/MCP + WDBX | U | Cross-tenant and revoked access cannot enter ranked results or provider request |
| C15 | Complete retained-copy inventory | WDBX owner; ABI/Abbey consumers | M | Active/generation/replica/backup/history/outbox/training copies and retention obligations |
| C16 | Durable retention/member erasure | WDBX owner | M | Progress receipt; interruption/offline replicas/restore cannot resurrect deleted member |
| C17 | Key provision/rotate/revoke/recover/destroy | WDBX owner | U | Consumer key-custody contract and restore/revocation runtime acceptance |
| C18 | OIDC accounts and sessions | ABI control-plane/oidc | U | Issuer/audience/expiry/rotation negatives; real configured issuer E |
| C19 | Tenant pairing/enrollment/revocation | ABI control-plane/enrollment | U | Single-use exact confirmation, fresh revocation and tenant-bound identity |
| C20 | Browser signup/status/delete/recovery | ABI control-plane/web | U | Browser journeys with two tenants, interruption and cancellation |
| C21 | Quotas and bounded private routing | ABI control-plane | U | 2 active / 10 queued defaults, queue-full refusal and metadata-only persistence |
| C22 | Operator public train/eval namespace | ABI worker/control-plane | M | Operator resources cannot receive tenant private inference/memory jobs |
| C23 | Rust causal decoder training | ABI abi-nn | M | Attention/masking/gradient/optimizer reference cases; actual parameter updates |
| C24 | Versioned tokenizer/checkpoint/optimizer | ABI nn/models/runtime | M | Round-trip integrity, tokenizer parity and invalid artifact refusal |
| C25 | Deterministic resumable training | ABI nn/SEA | M | Restored optimizer/order/random state reproduces uninterrupted updates |
| C26 | Licensed provenance and split isolation | ABI SEA training registry | U | Rights records, dedup before split, cross-snapshot related-family isolation |
| C27 | Frozen broad evaluation | ABI model evaluation | M | Versioned language/knowledge/reasoning/writing/coding/tool/refusal/privacy/persona scoring |
| C28 | CPU/Metal real model runtime | ABI model-runtime | M | Transformer artifact inference; explicit device, bounds and cancellation; bigram fixture insufficient |
| C29 | 100M/300M/1B pilot measurements | ABI training | E | Approved resource envelope, hardware/tokens/loss/cost/energy/throughput receipts |
| C30 | Original trained weights and quality | ABI training/evaluation | E | Real selected weights, frozen-suite scores/limits and user-hosted conversations |
| C31 | Model promotion/rollback identities | ABI models/runtime | U | Bind dataset/config/tokenizer/checkpoint/evaluation/executable digests |
| C32 | Signed macOS/Linux/Windows packages | ABI distribution | M | Package identities, native libraries, install/upgrade/interruption/rollback/removal |
| C33 | Immediate Mac full journey | ABI + Abbey/WDBX | U | Current Mac installed package signup/pair/model/converse/recall/approve/refuse/restart; Linux/Windows builds only in current scope |
| C34 | Supported Mac Metal acceptance | ABI runtime/GPU | U | Executed paths and stated verification scope; no inferred CUDA/Vulkan/ANE qualification |
| C35 | Frozen CLI/MCP/protocol compatibility | ABI CLI/MCP/contracts | Q local Mac CLI/MCP | Full contract gate and copied package verify help bytes and MCP 12 via launcher; public service compatibility remains separate |
| C36 | Real separate-host partitions/recovery | ABI/WDBX/Abbey | E | Physical separate hosts, loss/reconnect/revocation, zero unauthorized/replayed effects |
| C37 | Control-plane and memory backup/restore | ABI control-plane; WDBX owner | U | Restore preserves revocation/deletion/execution history and key boundaries |
| C38 | Content-free operational metrics | ABI service | M | Health/queue/auth failures/job outcomes/resources without private content |
| C39 | Tested operator runbooks | ABI operations | M | Key rotation/compromise/revocation/recovery/upgrades/shutdown drills |
| C40 | Independent assessment/public testing | External assessor + ABI fixes | E | Actual assessment, resolved blockers and explicit residual findings |
| C41 | Portable single-region staging/public service | ABI operations | E | Deployment credentials/authorization; staging journey then public rollout |
| C42 | Exact-source final gates and dependency pin | ABI/WDBX/Abbey owners | E | All three gates incl. Abbey desktop; exact WDBX pin qualified before promotion |
| C43 | CI and publication receipts | ABI release | E | Explicit CI resumption/publication authorization, published SHA and artifact checksums |
| C44 | Documentation and measured claims | ABI docs/release | U | Current receipts, model scores, measured benchmarks, limits and reproducible instructions |

## Execution record

- Initial source: ABI `69910bf61205458dec27dd4b616201e0a5f3898a`, canonical
  main with existing dirty/untracked continuation preserved. WDBX and Abbey also
  have active source changes; no historical cohort receipt covers those changes.
- Baseline run started during planning and its process returned exit 0 when
  collected during implementation. It reports `check: all green`; existing
  sibling `AdmissionSimulator` unused-import warning remains disclosed.
- Ruling: the approved canonical-main constraint overrides the execution skill's
  default isolated worktree. Do not commit unrelated existing source; record exact
  owned changes and retain workspaces while the overall plan remains unfinished.
- Ruling: external requirements stay open while local work progresses. No spending,
  external account effects, CI resumption or deployment is inferred from approval
  to implement source. Mistaking local tests for those requirements would falsely
  qualify an untested release.
- Divergence reconciliation (2026-10-25): local `main` (through `ef6a496`) merged
  the locally known `origin/main` `42f1446` (README architecture/evidence rewrite)
  as merge commit `f65d5e5`, preserving both histories; no force-push, no rebase.
  The rewrite dropped the literal `abi agent train` behavior name that
  `tools/tests/test_docs_policy.py` requires in `README.md`, so
  commit `2113824` restored the exact persistence-behavior naming without
  resurrecting the replaced bullet layout.
- Post-merge gate (2026-10-25, ABI `main` `2113824`):
  `PATH="$HOME/.cargo/bin:$PATH" ./tools/check.sh < /dev/null` exit 0,
  `check: all green` — 186 Python policy tests OK (8 documented skips),
  1,241 Rust test executions passed with 0 failed and 0 ignored across the
  workspace and required sibling conformance suites, `cargo-audit` 0.22.2 ran
  with only the two existing scoped exceptions (RUSTSEC-2025-0141,
  RUSTSEC-2024-0436), warning-denied Clippy, size limits, formatting,
  workspace build/tests, sibling WDBX v3 cross-language conformance, local
  Metal tests and docs, same-system benchmark regression (`benchmark
  regression gate: PASS`), rustdoc, and release `abi-cli`/`abi-mcp` builds.
  CUDA source checks remain unavailable (no `nvcc` on this host). Log:
  `abi-finish-main-gate2.log` in the opencode temporary directory.
- Binding status unchanged for external inputs: the gate consumed the live
  sibling WDBX at `550451b` carrying another owner's staged/uncommitted changes,
  not the CI pin `2969d71`; WDBX's own gate and the Abbey root/desktop gates
  remain their owners' actions. This receipt is local source evidence for C02's
  current baseline, not a frozen cohort receipt; C42 stays external, and no
  C01–C44 row closes.

## Current implementation and review

- Worker inference now rechecks trusted time after the authority callback and
  budgets only the remaining lease. The actual regression dispatched a provider
  after callback-induced expiry/clock rollback; the repair preserves the existing
  admission and refuses another dispatch. Inference suite: 16 passed.
- File history has a private I/O boundary for instance-owned fault injection.
  Actual file writes cover short writes, partial length/payload/checksum failure,
  zero-byte writes and sync uncertainty. Before/after-terminal-commit reopen
  tests preserve one provider invocation. These are additional durability proofs,
  not claims that the already-correct append code originally failed every case.
- The closed Unix process module pins the system pwd executable and canonical
  directory, seals exact compiled source/approval contracts, bounds supervision
  and publishes consumable request-bound actual OS observations. Native Mac
  host tests prove A1 approval/refusal/replay; the lifecycle in these tests is
  in-memory. C07 is not qualified by this evidence.
- Independent fresh review found a final-filesystem-observation cancellation
  defect, missing execution-binding negatives and an APFS-invalid-name test
  setup error. All three were repaired in one pass. Deterministic cancellation,
  deadline, rollback and expiry regressions retain the original completion TTL;
  changed request/prepared/source/tool and cancelled-host cases refuse.
- Internal source compilation now supports the specification's dotted fact
  names while the frozen public capability constructor keeps its old grammar.
  Darwin zombie-only group cleanup requires nonreaping exit observation and an
  independent sole-leader kernel group query; arbitrary EPERM is never ignored.
- Focused worker/host/capability/control-plane suites: 285 passed, zero failures
  or ignored (before the subsequent dependency-only patch). Focused Clippy exits
  0; docs policy 3 passed; Mintlify validation passes. Final full qualification
  and installed artifact acceptance are recorded in the external evidence folder.
- The first final-gate attempt stopped on newly published ctutils advisory
  RUSTSEC-2026-0385. ABI's exact lock entry is updated from 0.4.3 to 0.4.4 using
  the registry checksum and unchanged dependency metadata; required dependency
  rescan exits 0 with existing exceptions unchanged. All subsequent Cargo stages
  remain locked. Original failure log and successful rescan are retained.
- Windows cross-check fails in ring C compilation because the Windows SDK's
  `assert.h` is absent. Only the Windows Rust std target is installed. Docker
  has no running daemon. No Windows/Linux executable or runtime proof is claimed.
- Current Abbey locked metadata refuses the new host libc dependency. Concrete
  minimal Abbey/WDBX lock patches are retained externally for their owners;
  neither sibling's active working tree is modified. Current sibling source
  drift also prevents inheriting older exact-cohort receipts.

## Qualified local delivery

- Complete `./tools/check.sh`: **exit 0**, 1,234 Rust test executions, zero
  failures/ignored. Python policy: 186 cases, eight preexisting personal-sync
  skips. Required audit actually ran; sizes, formatting, strict Clippy, workspace
  build/tests, required WDBX contracts, Metal tests/docs, same-system benchmark,
  rustdoc and release builds passed. CUDA compilation remains unavailable without
  nvcc; the existing sibling AdmissionSimulator warning remains disclosed.
- ABI source files/modes remained identical throughout the gate. Consumed WDBX
  crates/manifests/lockfile also stayed identical. Three unrelated WDBX release
  catalog/checker files changed concurrently; this is not a full WDBX own-gate
  or a complete three-repository qualification receipt. Abbey stayed unchanged
  through this run, but its newer consumer lock/source handoff remains required.
- Store-safe `run-abi` smoke: 18 passed, zero failures. Browser Node tests:
  nine passed, zero failures/skips.
- A fresh extracted local Mac CLI/MCP candidate verifies four existing code
  signatures, exact help bytes, memory-only completion, backend reporting and
  the exact launcher-based 12-tool MCP golden. The initial artifact verifier
  wrongly expected CLI help on stdout; current CLI emits the exact golden on
  stderr. The corrected verifier checks both streams, with the failed setup
  retained externally. No product code or golden was changed for this.
- Candidate is an archive, not a notarized installer or completed assistant.
  It contains CLI/MCP and native dylibs; no original weights, general desktop
  tools, public service deployment or Windows/Linux executable is included.
- Evidence: `/Users/donald/.codex/verification/abi-completion-20261023/`.
  `local-source-receipt.json` binds source digests, logs and installed artifact
  checks. `installed-candidate-receipt.json` binds the candidate archive SHA-256.
  The matrix/task-ledger result additions are reporting-only changes after the
  executable source gate; documentation policy and Mintlify validation are rerun.

## Next execution gates

1. Integrate the exact process adapter with Abbey's real durable local approval
   owner after its source/lock handoff; do not substitute the in-memory Ledger.
2. Complete real file/desktop/messaging adapter coverage and WDBX-owned retention,
   member-erasure and key-custody contracts. Existing scratch effects are narrower.
3. Implement the actual transformer/training/runtime pipeline and obtain an
   explicit pilot resource envelope before resource-consuming training.
4. Supply Windows C SDK/linker and a Linux cross/native build environment for
   desktop build artifacts. Immediate runtime testing remains Mac-only.
5. Real OIDC, independent assessment, CI/publication and public operation retain
   their existing external prerequisites. None is silently closed by this slice.
