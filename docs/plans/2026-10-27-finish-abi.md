# Finish ABI (local scope) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the locally buildable `M` rows Donald selected on 2026-10-27 (C38 metrics, C39 runbooks, C11 governed messaging, C10 governed desktop, C32 Mac packaging), moving each to `U`: implemented, with positive, refusal and recovery tests. Qualification beyond that stays with the named external owners.

**Architecture:** Each row is one vertical slice that follows the existing patterns. Control-plane operator surfaces live in `abi-control-plane`. Governed adapters are new `GovernedLiveExecutor` implementations in `abi-agent-host`, built like `platform_process.rs` with closed bindings, single-shot execution and trusted postconditions. Packaging is shell plus a Python policy test, built like the existing `tools/` gates.

**Tech Stack:** Rust 1.99.1 stable (edition 2024) through `./tools/cargo.sh`; axum/rusqlite in `abi-control-plane`; `abi-agent-host` governed lifecycle; POSIX shell and Python 3.11+ `unittest` for packaging.

**Spec:** `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md`, sections A4 (C10/C11), A9 (C32) and A10 (C38/C39), plus the C-row acceptance text in `docs/verification/2026-10-23-completion-matrix.md`. Current state: `docs/verification/2026-10-27-c01-c44-execution-ledger.md`.

Scope (Donald, 2026-10-27): the Rust training stack (C23–C28) belongs to LYLX and is not built here. External rows (E, live accounts, separate hosts, signing, Linux/Windows runtime, publication) are never closed by this plan.

## Global Constraints

- Run Cargo only through `./tools/cargo.sh`. Every dependency-resolving stage takes `--locked`, and every `cargo test` ends with `< /dev/null`.
- Work on branch `task/finish-abi-20261027` in the canonical checkout. At the end, fast-forward it into local `main`. Never push, never change the CI `WDBX_REVISION` pin, never edit `../wdbx` or `../abbey`, never open `~/.abi/`.
- `unsafe_code = "deny"` and `clippy -D warnings` hold. Every `.rs` file stays at or under 1000 lines. New tests go in new files.
- No new `ABI_*` environment variables (their registry lives in `../wdbx`). Configuration goes through existing config files or arguments.
- Frozen surfaces stay unchanged: 14 CLI commands, 12 MCP tools and the goldens. No new `abi` top-level command.
- A row moves `M`→`U` only with tests dated in this run. `Q` requires a qualification receipt and is not claimed here.
- Commit messages are Conventional Commits and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Metrics leak private content.** Tenant IDs, issuer/subject, node or job IDs, tokens or digests appear in metrics output. Expect only fixed keys and integers. Owned by Task 1's `metrics_contain_no_identifiers` test.
2. **Metrics endpoint reachable without operator authority.** Expect 404 when unconfigured and 401 for a wrong or missing bearer token, with no counts in either body. Owned by Task 2.
3. **Ambiguous send is retried.** A transport error after the request may have left the host must surface as an uncertain failure and make exactly one transport call. Owned by Task 4's `ambiguous_send_is_not_retried`.
4. **Approved recipient or content differs from the executed one.** Expect refusal before any transport call. Owned by Task 4's `recipient_or_content_mismatch_refuses_before_send`.
5. **Interrupted upgrade leaves no runnable install.** Expect `current` to keep pointing at the previous complete version. Owned by Task 6's `test_interrupted_upgrade_keeps_previous_current`.

---

### Task 1: Content-free operator metrics snapshot (C38)

**Files:**
- Create: `crates/abi-control-plane/src/metrics.rs`
- Create: `crates/abi-control-plane/src/authority/metrics_tests.rs`
- Modify: `crates/abi-control-plane/src/lib.rs` (add `mod metrics; pub use metrics::OperatorMetrics;`), `crates/abi-control-plane/src/authority.rs` (add the method and register the test module)

**Interfaces:**
- Produces: `pub struct OperatorMetrics` (Serialize) with fields `accounts: u64`, `sessions_active: u64`, `nodes_active: u64`, `nodes_revoked: u64`, `pairings_pending: u64`, `jobs: BTreeMap<&'static str, u64>`, `audit_window_secs: i64`, `audit_events: BTreeMap<&'static str, u64>`. Also `impl Authority { pub fn operator_metrics(&self, now: i64, window_ms: i64) -> Result<OperatorMetrics> }`.
- Job-state keys use the closed set `queued, offered, dispatched, cancel_requested, cancelled, completed, expired, unresolved`; any other stored value counts under `other`. Audit-kind keys use the closed set of the eleven literal kinds now written by `audit()` (`session.created`, `session.deleted`, `node.revoked`, `job.queued`, `job.offered`, `job.dispatched`, `job.frame`, `job.finished`, `job.cancelled`, `job.stopped`, `relay.recovered`), with unknown kinds counted under `other`. Keys are `&'static str` taken from these arrays, never from database text.

- [ ] **Step 1: Write failing tests** in `metrics_tests.rs`, using the existing authority test fixtures. Test `metrics_count_accounts_sessions_nodes_and_jobs`: two accounts, one session each, one revoked node and one queued job give `accounts == 2`, `sessions_active == 2`, `nodes_revoked == 1`, `jobs["queued"] == 1`. Test `metrics_contain_no_identifiers`: serialize to JSON and assert that no account id, tenant, issuer, subject, session token, node id or job id string is a substring. Test `expired_sessions_and_old_audit_are_excluded`: a session with `expires <= now` and audit rows older than `now - window_ms` are not counted.
- [ ] **Step 2: Run** `./tools/cargo.sh test --locked -p abi-control-plane --lib -- metrics < /dev/null`. Expected: compile failure (no `operator_metrics`).
- [ ] **Step 3: Implement** `operator_metrics` as read-only `SELECT count(*)` / `GROUP BY` queries over `accounts`, `sessions`, `nodes`, `pairs` (not consumed and not expired), `relay_jobs` and `audit`, mapping every group label through the closed key arrays.
- [ ] **Step 4: Run** Step 2's command. Expected: `3 passed`.
- [ ] **Step 5: Commit** `feat(control-plane): content-free operator metrics snapshot`.

### Task 2: Health, authenticated metrics endpoint, auth-failure counter and CLI (C38)

**Files:**
- Modify: `crates/abi-control-plane/src/service.rs` (routes, `App` fields), `crates/abi-control-plane/src/main.rs` (`Config.operator_token_sha256: Option<String>`, `metrics DATABASE` subcommand), `crates/abi-control-plane/src/lib.rs` (`router_with_limits` gains an `Option<[u8; 32]>` operator digest through a new `router_with_operator` entry point; the existing signature is kept)
- Create: `crates/abi-control-plane/src/service/tests/operator.rs`

**Interfaces:**
- Consumes: `Authority::operator_metrics` (Task 1).
- Produces: `GET /healthz` → 200 `{"status":"ok"}` when the authority answers `SELECT 1`, otherwise 503 `{"status":"unavailable"}`. `GET /v1/operator/metrics` → 404 when no digest is configured; 401 unless `Authorization: Bearer <t>` with `sha256(t)` equal to the configured digest (constant-time compare); 200 with `{"metrics": OperatorMetrics, "auth_denied_total": u64}` otherwise. The window is 86,400 s (timestamps in this crate are seconds). `auth_denied_total` is an `AtomicU64` in `App`, incremented wherever `credentials()` returns `Error::Denied`. CLI: `abi-control-plane metrics DATABASE ORIGIN` prints the same snapshot as JSON with `auth_denied_total` omitted, since that counter is process-local.

- [ ] **Step 1: Write failing tests** in `operator.rs`. `healthz_reports_ok_without_content`: 200, and the body equals `{"status":"ok"}` exactly. `operator_metrics_is_absent_when_unconfigured`: 404. `operator_metrics_refuses_missing_and_wrong_token`: 401 for both, and the body contains no digits. `operator_metrics_counts_denied_requests`: two `GET /v1/account` calls without a session, then the correct token, give `auth_denied_total >= 2` and `metrics.accounts` present.
- [ ] **Step 2: Run** `./tools/cargo.sh test --locked -p abi-control-plane --lib -- operator < /dev/null`. Expected: FAIL.
- [ ] **Step 3: Implement** the routes and counter. Both routes go outside session CSRF. The metrics route never touches tenant-scoped data beyond Task 1's aggregates.
- [ ] **Step 4: Run** Step 2's command. Expected: `4 passed`. Then run `./tools/cargo.sh clippy --locked -p abi-control-plane --all-targets -- -D warnings` and `bash tools/check_rust_sizes.sh`. Both exit 0.
- [ ] **Step 5: Document** the routes and the `operator_token_sha256` config key in `crates/abi-control-plane/README.md`. Commit `feat(control-plane): health and operator metrics endpoints`.

### Task 3: Operator runbooks with executable drills (C39)

**Files:**
- Create: `crates/abi-control-plane/RUNBOOKS.md`; `crates/abi-control-plane/src/service/tests/runbook_drills.rs`
- Modify: `crates/abi-control-plane/src/service/tests.rs` (register the module); `crates/abi-control-plane/README.md` (link)

**Interfaces:**
- Consumes: existing service test support (router construction, pairing, revocation, node authentication) and Task 2's `/healthz`.

Runbooks: (1) node compromise / tenant revocation; (2) authority signing-key compromise; (3) backup and restore; (4) upgrade and rollback of the binary; (5) emergency shutdown; (6) metrics and health checks. Each section gives preconditions, exact commands, the verification signal, and the drill test that exercises it. Key rotation is not supported today (README lists it as open). Runbook 2 says so and documents the supported procedure: stop, `init` a new authority, re-pair every node, with old node certificates refused. It does not invent a rotation command.

- [ ] **Step 1: Write the drill tests.** `drill_node_revocation_refuses_authentication`: pair, revoke, and the next `/v1/nodes/authenticate` → 401. `drill_restore_preserves_revocation_and_deletion`: revoke a node and delete an account; copy the closed database file; reopen from the copy; the revoked node is still refused and the deleted account is absent. `drill_new_authority_refuses_old_node_certificates`: a certificate signed by authority A is refused by a freshly initialized authority B. `drill_graceful_shutdown_completes_inflight_health`: a server using `with_graceful_shutdown` on a oneshot answers one `/healthz`, then stops after the signal within 5 s.
- [ ] **Step 2: Run** `./tools/cargo.sh test --locked -p abi-control-plane --lib -- runbook_drills < /dev/null`. Record the result: a drill that passes on first run is a characterization of existing behaviour. A failure is a defect, handled under superpowers:systematic-debugging with a RED→GREEN fix.
- [ ] **Step 3: Write `RUNBOOKS.md`**, citing each drill by name. State the limits: local drills only; staging/separate-host execution (C36/C41) stays external.
- [ ] **Step 4: Run** the docs checks `python3 -m unittest discover -s tools/tests -p 'test_docs*.py'` and `git diff --check`. Commit `docs(control-plane): operator runbooks with executable drills`.

### Task 4: Governed Discord/Twilio message executor (C11)

**Files:**
- Create: `crates/abi-agent-host/src/governed_message.rs`; `crates/abi-agent-host/tests/governed_message.rs`; `crates/abi-agent-host/tests/governed_message/source.json`
- Create: `crates/abi-connectors/src/governed_sender.rs`; tests inline under `#[cfg(test)]`
- Modify: `crates/abi-agent-host/src/lib.rs` (export); `crates/abi-connectors/Cargo.toml` (+ `abi-agent-host = { workspace = true }`); `crates/abi-connectors/src/lib.rs`

**Interfaces:**
- Produces in `abi-agent-host`:
  - `pub enum MessageChannel { DiscordChannel, TwilioSms }`
  - `pub struct ApprovedMessage { pub channel: MessageChannel, pub account: String, pub recipient: String, pub body: String }`
  - `pub struct MessageReceipt { pub provider_id: String }`
  - `pub enum SendError { Refused(&'static str), Uncertain(&'static str) }`
  - `pub trait MessageSender { fn send(&mut self, message: &ApprovedMessage) -> Result<MessageReceipt, SendError>; }`
  - `pub struct GovernedMessageExecutor<S: MessageSender>`, with `pub fn bind(source: &CompiledSource, approved: ApprovedMessage, sender: S, clock) -> Result<Self, ToolExecutionError>` and `impl GovernedLiveExecutor` (tool name `message_send`, adapter id `abi.message`, api version `1`). `pub fn sends(&self) -> u32`.
- Produces in `abi-connectors`: `pub struct ConnectorMessageSender<'t> { transport: &'t dyn Transport, discord: Option<DiscordClient>, twilio: Option<TwilioClient> }` with `impl MessageSender`, mapping to `send_message_live`.

Binding rules (from A4: "exact approval binds account, recipient, content/action, scope and expiry"): `execute` refuses with no sender call when any of these hold: `call.input` JSON ≠ `{"channel","account","recipient","body"}` equal to the bound `ApprovedMessage`; the `prepared`/`request` binding checks of `GovernedProcessExecutor::execute` fail (reuse them: same commitment and policy-digest predicates); `context.should_stop()`; or the clock is at/after source expiry. On `SendError::Uncertain`, return a `ToolExecutionError` and never call `send` again on that executor. The lifecycle already records the outcome as uncertain/unresolved. A second `execute` on the same executor after any attempt refuses with `message_already_attempted`.

- [ ] **Step 1: Write failing tests** with a counting fake `MessageSender`, mirroring the compiled-source setup of `crates/abi-agent-host/tests/platform_process.rs`: `approved_message_sends_exactly_once_and_returns_receipt`; `recipient_or_content_mismatch_refuses_before_send` (sends == 0 for each of recipient, body, account and channel changed); `ambiguous_send_is_not_retried` (fake returns `Uncertain`; result is Err; sends == 1; a second execute refuses and sends stays 1); `expired_source_or_stop_refuses_before_send`. In `abi-connectors`: `connector_sender_builds_exact_discord_and_twilio_requests` using `RecordingTransport` (asserts method, URL path, recipient and body bytes), and `transport_error_maps_to_uncertain`.
- [ ] **Step 2: Run** `./tools/cargo.sh test --locked -p abi-agent-host --test governed_message < /dev/null` and `./tools/cargo.sh test --locked -p abi-connectors --lib -- governed_sender < /dev/null`. Expected: FAIL.
- [ ] **Step 3: Implement** the executor and the sender.
- [ ] **Step 4: Run** Step 2's commands. Expected: all pass. Then run clippy for both crates and the size check, and confirm the `abi-cli` goldens are unchanged: `./tools/cargo.sh test --locked -p abi-cli --test golden < /dev/null`.
- [ ] **Step 5: Commit** `feat(agent-host): governed message executor for Discord/Twilio`. Live-account acceptance stays external (C11 E part).

### Task 5: Governed desktop application activation (C10, macOS)

**Files:**
- Create: `crates/abi-agent-host/src/governed_desktop.rs`; `crates/abi-agent-host/src/governed_desktop/macos.rs`; `crates/abi-agent-host/tests/governed_desktop.rs`; `crates/abi-agent-host/tests/governed_desktop/source.json`
- Modify: `crates/abi-agent-host/src/lib.rs`

**Interfaces:**
- Produces:
  - `pub trait DesktopBackend { fn frontmost_bundle_id(&mut self) -> Result<String, DesktopError>; fn activate(&mut self, bundle_id: &str) -> Result<(), DesktopError>; }`
  - `pub enum DesktopError { Unsupported, PermissionDenied, Failed }`
  - `pub struct MacDesktop` (`#[cfg(target_os = "macos")]`), which runs the pinned executables `/usr/bin/lsappinfo` (`front`, then `info -only bundleid`) and `/usr/bin/open -b <id>` with a bounded wait, plus no-op unsupported stubs elsewhere
  - `pub struct GovernedDesktopExecutor<B: DesktopBackend>`: `bind(source, approved_bundle_id: String, backend, clock)`, `impl GovernedLiveExecutor` (tool name `desktop_activate`, adapter id `abi.desktop`, api version `1`). Its trusted postcondition `desktop.frontmost_matches` is verified by reading `frontmost_bundle_id()` after activation (at most 20 polls, 100 ms apart).
- Bundle IDs are validated as reverse-DNS `[A-Za-z0-9.-]{3,255}` with at least one dot. Anything else refuses.

- [ ] **Step 1: Write failing tests** with a fake backend: `approved_application_activates_and_verifies_frontmost`; `different_bundle_in_call_refuses_before_activation`; `backend_unsupported_or_permission_denied_refuses` (no activation; distinct error codes `desktop_unsupported` / `desktop_permission_denied`); `frontmost_mismatch_after_activation_is_uncertain_not_retried` (activate called once). Add `#[ignore] native_mac_activation_of_finder`, run by hand, which activates `com.apple.finder` through `MacDesktop`.
- [ ] **Step 2: Run** `./tools/cargo.sh test --locked -p abi-agent-host --test governed_desktop < /dev/null`. Expected: FAIL.
- [ ] **Step 3: Implement.** Process execution uses `std::process::Command` with an absolute path, a cleared environment and a bounded wait. Never use a shell.
- [ ] **Step 4: Run** Step 2's command (all non-ignored pass), clippy and the size check. Run the ignored native test once with `-- --ignored native_mac`, and record its result in the commit body and the ledger. A pass there is local Mac evidence only.
- [ ] **Step 5: Commit** `feat(agent-host): governed macOS application activation`. Screen capture, click and text entry stay in the A4 backlog.

### Task 6: Mac package, install, upgrade, rollback and removal (C32, unsigned)

**Files:**
- Create: `tools/package/build_macos.sh`; `tools/package/install.sh`; `tools/tests/test_package_install.py`
- Modify: `CLAUDE.md` and `AGENTS.md` (policy-suite list 14 → 15, plus package commands); `docs/verification/2026-10-27-c01-c44-execution-ledger.md` (later, in Task 7)

**Interfaces:**
- `build_macos.sh OUT_DIR`: builds `--locked --release -p abi-cli -p abi-mcp -p abi-control-plane` through `./tools/cargo.sh` and stages `abi-<version>-<git short rev>-aarch64-apple-darwin/{bin/abi,bin/abi-mcp,bin/abi-control-plane,lib/*.dylib,MANIFEST.json}`. `MANIFEST.json` holds `{"name":"abi","version","revision","target","rustc","files":{relpath: sha256}}`. Output is a `.tar.gz` plus a `.sha256`. It refuses if the tree is dirty. Dylibs are copied by `cp X X.tmp && mv` (no in-place rewrite).
- `install.sh install|rollback|remove|status PREFIX [ARCHIVE]`: verifies the archive `.sha256` and every manifest hash, extracts into `PREFIX/versions/.staging-<id>`, renames it to `PREFIX/versions/<id>`, and atomically repoints `PREFIX/current` (by `ln -s` to a temporary name, then `mv -h` over the old one). It records `PREFIX/previous`. `rollback` swaps `current` and `previous`. `remove` deletes only `PREFIX/versions`, `current` and `previous` after a confirmation flag `--yes`. Leftover `.staging-*` directories are removed at the start of the next install.

- [ ] **Step 1: Write failing tests** in `test_package_install.py`. They use synthetic archives built in a temp dir by a helper that writes a fake `bin/abi` script and a matching `MANIFEST.json`; no cargo build. Tests: `test_install_then_status_reports_version`; `test_upgrade_keeps_previous_and_rollback_restores_it`; `test_checksum_mismatch_refuses_and_leaves_current`; `test_interrupted_upgrade_keeps_previous_current` (simulate interruption by leaving `.staging-x`; `current` still resolves to the old version, and the next install cleans the staging dir); `test_remove_requires_yes_and_removes_only_owned_paths` (a sibling file in `PREFIX` survives).
- [ ] **Step 2: Run** `python3 -m unittest tools/tests/test_package_install.py -v`. Expected: FAIL (script missing).
- [ ] **Step 3: Implement** both scripts (POSIX `sh`, `set -eu`; `shasum -a 256`).
- [ ] **Step 4: Run** Step 2 (pass) and the full policy discovery `python3 -m unittest discover -s tools/tests -p 'test_*.py'` (OK). Run `build_macos.sh .work/finish-20261027/pkg` once from a clean tree, then install into `.work/finish-20261027/prefix` and run `PREFIX/current/bin/abi backends`. Record the result in the commit body. If an instruction-file test enumerates the suites, update it with the new suite.
- [ ] **Step 5: Commit** `feat(tools): unsigned macOS package with install, upgrade, rollback and removal`. Signing, notarization and Linux/Windows packages stay external.

### Task 7: Ledger, matrix and todo update; bound gate

**Files:**
- Modify: `docs/verification/2026-10-27-c01-c44-execution-ledger.md` (rows C10, C11, C32, C38, C39), `docs/verification/2026-10-23-completion-matrix.md` (same rows plus an execution-record bullet), `tasks/todo.md` (one dated entry)
- Create: `docs/verification/2026-10-27-finish-gate-receipt.md`

- [ ] **Step 1:** Run the bound full gate on the branch head, using fresh `--shared` clones of ABI HEAD and WDBX HEAD under `.work/finish-20261027/final/`, with the same invocation and capture procedure as the A1 receipt. Expected: `exit=0`, `check: all green`, `dependency-scan: RUN`, no `SKIP`. Disclose load-sensitive reruns.
- [ ] **Step 2:** Set each of the five rows `M`→`U`, citing that row's test files and the receipt, and keep each row's stated limits: live accounts, signing, Linux/Windows, staging and separate hosts remain external. Mirror the change in the matrix and confirm the parity diff shows only those rows.
- [ ] **Step 3:** Run the docs checks, commit `docs(verification): record finish-ABI slices and bound gate`, and fast-forward `task/finish-abi-20261027` into local `main`.
