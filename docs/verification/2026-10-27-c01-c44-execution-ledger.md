# C01–C44 execution ledger — 2026-10-27

Authority: [`2026-10-23-completion-matrix.md`](2026-10-23-completion-matrix.md).
The matrix table remains the requirement text. This ledger records, for each row,
the present evidence, owner, next action, and whether the remaining evidence is
local or external. Stage names (A1–A11, Stage B) refer to
[`../plans/2026-10-25-complete-abi-and-lylx-handoff.md`](../plans/2026-10-25-complete-abi-and-lylx-handoff.md).

Status legend (verbatim from the matrix): Q = qualified for explicitly named
scope only; U = existing implementation awaiting current qualification;
M = missing; E = requires external environment, resources or existing execution
authorization. Q local is never equivalent to public, separate-host or
cross-platform acceptance.

Bound source set (from the A1 receipt,
[`2026-10-27-a1-gate-receipt.md`](2026-10-27-a1-gate-receipt.md)): ABI
`1dd7f0bf73c644b75d4b65b9ce2574bef1872cf3`, WDBX
`9d386ddbe57c9e8423dc4755f231e1387a0cd1f7` (both from clean shared clones).
Latest bound gate: ABI `b8ca1ec` + WDBX `52a2143` (clean shared clones), exit 0 —
[`2026-10-27-finish-gate-receipt.md`](2026-10-27-finish-gate-receipt.md).

Rules:

- A status that differs from the matrix requires an evidence path dated in the
  run that changed it (2026-10-27 or later).
- External rows never close locally.
- This ledger and the matrix must agree on every status.

Scope ruling (Donald, 2026-10-27): the Rust model-training stack (C23–C25,
C27, C28) belongs to LYLX, not to legacy ABI. Those rows keep the matrix status
`M` for ABI and are marked External here. C29/C30 (training resources and
original weights) follow them, and C26/C31 migrate with them in Stage B. No
training stack is built in legacy ABI.

| ID | Requirement | Status | Evidence path | Owner | Next action | Local/External |
| --- | --- | --- | --- | --- | --- | --- |
| C01 | Exact cohort and requirement coverage | U | `docs/verification/2026-10-27-a1-gate-receipt.md` | ABI verification | A1: rebind exact source set once owner gates exist | Local |
| C02 | Canonical full gate | Q local baseline | `docs/verification/2026-10-27-a1-gate-receipt.md`; `docs/verification/2026-10-27-finish-gate-receipt.md` | ABI tools/check.sh | A1: rerun bound gate on each source change; quiet-host run for load-sensitive tests | Local |
| C03 | Durable bounded worker control history | Q local Unix | `crates/abi-worker/src/file_store/fault_tests.rs`; `crates/abi-worker/tests/durable_control.rs`; `crates/abi-worker/tests/terminal_durability.rs`; `docs/verification/2026-10-26-a2-relay-requalification.md`; `docs/verification/2026-10-27-a2-gate-receipt.md` | ABI abi-worker/file_store | Done for A2 (2026-10-27); limits: no power-loss, separate-host, public or Windows/Linux proof | Local |
| C04 | Authenticated worker admission and inference | Q local Mac | `crates/abi-worker/tests/authority_recheck.rs`; `crates/abi-worker/tests/inference.rs`; `crates/abi-worker/tests/contracts.rs`; `docs/verification/2026-10-26-a2-relay-requalification.md`; `docs/verification/2026-10-27-a2-gate-receipt.md` | ABI abi-worker | Done for A2 (2026-10-27); limits: no power-loss, separate-host, public or Windows/Linux proof | Local |
| C05 | Outbound encrypted transport and stream | Q local Mac | `crates/abi-control-plane/src/service/tests/relay.rs`; `crates/abi-control-plane/src/relay/tests/streaming.rs`; `crates/abi-control-plane/src/service/tests/durable.rs`; `docs/verification/2026-10-26-a2-relay-requalification.md`; `docs/verification/2026-10-27-a2-gate-receipt.md` | ABI abi-control-plane | Done for A2 (2026-10-27); limits: no power-loss, separate-host, public or Windows/Linux proof | Local |
| C06 | Reconnect and response-loss recovery | Q local Mac | `crates/abi-control-plane/src/service/tests/reconnect_process.rs`; `crates/abi-control-plane/src/relay/tests/recovery.rs`; `crates/abi-control-plane/src/delivery_consumer/interruption_tests.rs`; `docs/verification/2026-10-26-a2-relay-requalification.md`; `docs/verification/2026-10-27-a2-gate-receipt.md` | ABI worker/control-plane | Done for A2 (2026-10-27); limits: no power-loss, separate-host, public or Windows/Linux proof | Local |
| C07 | Durable exact local approval/lifecycle | U | `crates/abi-agent-host/src/local_approval.rs`; `crates/abi-control-plane/src/service/tests/governed_tools.rs` | Abbey owner + ABI governed host | A3: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A3 | External |
| C08 | Governed process supervisor | Q local Mac adapter | `crates/abi-agent-host/tests/platform_process.rs`; `crates/abi-agent-host/tests/governed/process.rs`; `crates/abi-agent-host/src/platform_process/unix.rs` | ABI abi-agent-host/platform_process | A3: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A3 | Local |
| C09 | Governed files | U | `crates/abi-control-plane/src/service/tests/governed_tools.rs`; `crates/abi-agent-host/tests/governed/live.rs` | ABI adapter + Abbey lifecycle | A3: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A3 | Local |
| C10 | Governed desktop | U | `crates/abi-agent-host/src/governed_desktop.rs`; `crates/abi-agent-host/tests/governed_desktop.rs`; `docs/verification/2026-10-27-finish-gate-receipt.md` | ABI platform adapters | A4: screen capture, click/text entry, macOS permission detection; native acceptance beyond app activation | Local |
| C11 | Governed Discord/Twilio | U | `crates/abi-agent-host/src/governed_message.rs`; `crates/abi-agent-host/tests/governed_message.rs`; `crates/abi-connectors/src/governed_sender.rs`; `docs/verification/2026-10-27-finish-gate-receipt.md` | ABI connectors + Abbey lifecycle | A4: live test-account acceptance (external) and Abbey lifecycle join | External |
| C12 | Consented OpenAI/Anthropic/Grok | U | `crates/abi-connectors/src/providers.rs`; `crates/abi-connectors/src/transport.rs` (no transmission-consent test) | ABI connectors | A4: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A4 | External |
| C13 | Platform credential custody | U | `crates/abi-capability/src/credential.rs`; `crates/abi-cli/src/auth.rs` | WDBX foundation + ABI adapters | A5: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A5 | External |
| C14 | Scope retrieval before ranking/prompt | U | `crates/abi-sea/tests/scoped.rs` | ABI SEA/CLI/MCP + WDBX | A5: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A5 | Local |
| C15 | Complete retained-copy inventory | M | none | WDBX owner; ABI/Abbey consumers | A5: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A5 | External |
| C16 | Durable retention/member erasure | M | none (spec only: `docs/superpowers/specs/2026-09-29-spec-member-erasure.md`) | WDBX owner | A5: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A5 | External |
| C17 | Key provision/rotate/revoke/recover/destroy | U | none in ABI (WDBX-owned) | WDBX owner | A5: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A5 | External |
| C18 | OIDC accounts and sessions | U | `crates/abi-control-plane/src/service/tests.rs`; `crates/abi-control-plane/src/oidc.rs` | ABI control-plane/oidc | A6: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A6 | External |
| C19 | Tenant pairing/enrollment/revocation | U | `crates/abi-control-plane/src/authority/tests.rs`; `crates/abi-control-plane/src/service/tests.rs` | ABI control-plane/enrollment | A6: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A6 | Local |
| C20 | Browser signup/status/delete/recovery | U | `crates/abi-control-plane/web/tests/recovery.cjs`; `crates/abi-control-plane/web/tests/submission.cjs`; `crates/abi-control-plane/web/tests/streaming.cjs` | ABI control-plane/web | A6: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A6 | Local |
| C21 | Quotas and bounded private routing | U | `crates/abi-control-plane/src/relay/tests.rs`; `crates/abi-control-plane/src/relay/tests/recovery.rs` | ABI control-plane | A6: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A6 | Local |
| C22 | Operator public train/eval namespace | M | none | ABI worker/control-plane | A6: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A6 | Local |
| C23 | Rust causal decoder training | M | none (demo only: `crates/abi-nn/src/train.rs`) | ABI abi-nn | Owned by LYLX (Donald, 2026-10-27): no training stack in legacy ABI | External |
| C24 | Versioned tokenizer/checkpoint/optimizer | M | none (demo only: `crates/abi-nn/src/checkpoint.rs`) | ABI nn/models/runtime | Owned by LYLX (Donald, 2026-10-27): no training stack in legacy ABI | External |
| C25 | Deterministic resumable training | M | none | ABI nn/SEA | Owned by LYLX (Donald, 2026-10-27): no training stack in legacy ABI | External |
| C26 | Licensed provenance and split isolation | U | `crates/abi-sea/src/training_registry/tests/split_isolation.rs`; `crates/abi-sea/tests/training_foundation.rs` | ABI SEA training registry | Stage B: migrate to LYLX with the training stack (Donald, 2026-10-27) | Local |
| C27 | Frozen broad evaluation | M | none | ABI model evaluation | Owned by LYLX (Donald, 2026-10-27): no training stack in legacy ABI | External |
| C28 | CPU/Metal real model runtime | M | none (scratch fixture only: `crates/abi-model-runtime/tests/contracts.rs`) | ABI model-runtime | Owned by LYLX (Donald, 2026-10-27): no training stack in legacy ABI | External |
| C29 | 100M/300M/1B pilot measurements | E | none | ABI training | Owned by LYLX training (Donald, 2026-10-27); needs resource authorization | External |
| C30 | Original trained weights and quality | E | none | ABI training/evaluation | Owned by LYLX training (Donald, 2026-10-27); needs resource authorization | External |
| C31 | Model promotion/rollback identities | U | `crates/abi-sea/src/training_registry/runs.rs`; `crates/abi-ai/src/training_contracts.rs` (digest validation only) | ABI models/runtime | Stage B: migrate to LYLX with the training stack (Donald, 2026-10-27) | Local |
| C32 | Signed macOS/Linux/Windows packages | U | `tools/package/install.sh`; `tools/tests/test_package_install.py`; `docs/verification/2026-10-27-finish-gate-receipt.md` | ABI distribution | A9: signing/notarization, Linux/Windows packages, installed journey (C33) | Local |
| C33 | Immediate Mac full journey | U | none (nearest: `tools/run_contract_cli.sh`) | ABI + Abbey/WDBX | A9: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A9 | External |
| C34 | Supported Mac Metal acceptance | U | `crates/abi-model-runtime/tests/contracts.rs`; `crates/abi-gpu/src/metal_kernels.rs`; `crates/abi-gpu/tests/stable_cpu_contract.rs` | ABI runtime/GPU | A9: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A9 | Local |
| C35 | Frozen CLI/MCP/protocol compatibility | Q local Mac CLI/MCP | `crates/abi-cli/tests/golden.rs`; `crates/abi-mcp/src/transport_contract.rs`; `tools/tests/test_mcp_smoke.py` | ABI CLI/MCP/contracts | A9: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A9 | Local |
| C36 | Real separate-host partitions/recovery | E | none | ABI/WDBX/Abbey | A10: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A10 | External |
| C37 | Control-plane and memory backup/restore | U | `crates/abi-control-plane/src/service/tests/reconnect_process.rs`; `crates/abi-control-plane/src/relay/tests/recovery.rs` | ABI control-plane; WDBX owner | A10: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A10 | External |
| C38 | Content-free operational metrics | U | `crates/abi-control-plane/src/metrics.rs`; `crates/abi-control-plane/src/service/tests/operator.rs`; `docs/verification/2026-10-27-finish-gate-receipt.md` | ABI service | A10: alerting/export, audit retention and staging operation | Local |
| C39 | Tested operator runbooks | U | `crates/abi-control-plane/RUNBOOKS.md`; `crates/abi-control-plane/src/service/tests/runbook_drills.rs`; `docs/verification/2026-10-27-finish-gate-receipt.md` | ABI operations | A10: execute runbooks against staging/separate hosts (C36/C41) | Local |
| C40 | Independent assessment/public testing | E | none | External assessor + ABI fixes | A10: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A10 | External |
| C41 | Portable single-region staging/public service | E | none | ABI operations | A10: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A10 | External |
| C42 | Exact-source final gates and dependency pin | E | `docs/verification/2026-10-27-a1-gate-receipt.md` | ABI/WDBX/Abbey owners | WDBX own gate + Abbey root/desktop gates by owners against WDBX `9d386dd`; pin promotion after exact-rev pass | External |
| C43 | CI and publication receipts | E | `tools/ci_contract.py`; `tools/tests/test_release_gate.py`; `.github/workflows/ci.yml` | ABI release | A11: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A11 | External |
| C44 | Documentation and measured claims | U | `tools/tests/test_docs_policy.py`; `tools/tests/test_docs_templates.py` | ABI docs/release | A1: see `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` §A1 | Local |

## Blocked external inputs

A1's exit criterion ("ABI's gate, WDBX's own gate and Abbey root/desktop
gates") stays unmet until both receipts below exist. Neither is executed from
this repository.

- [ ] WDBX owner: own full gate at `52a2143179e1ec0bdbf23576d0399a64128713a7` (latest ABI-consumed revision; A1 bound `9d386dd`) — receipt path pending
- [ ] Abbey owner: root and desktop gates against WDBX `52a2143` and Abbey `0ad2fef80f0c1ec38dd1367882541e6a2cdec661` (observed, 12 dirty entries) — receipt path pending
