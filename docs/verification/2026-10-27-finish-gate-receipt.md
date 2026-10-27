# Finish-ABI bound gate receipt — 2026-10-27

Plan: [`../plans/2026-10-27-finish-abi.md`](../plans/2026-10-27-finish-abi.md), Task 7.

## Source set

The gate ran on clean `git clone --shared` copies under the ignored
`.work/finish-20261027/final/{abi,wdbx}`. Pre- and post-gate captures are
identical (`abi-stable`, `wdbx-stable`).

- ABI: branch `task/finish-abi-20261027` at
  `b8ca1eca0fda7102697e98a84e49d99328e112b4`. The only later commit, `c3ea94d`,
  edits `CLAUDE.md`/`AGENTS.md`. After it, all Python policy suites were rerun
  and passed (OK, 8 skips).
- WDBX: `52a2143179e1ec0bdbf23576d0399a64128713a7` (clean). Object IDs:
  - `crates`: `ad1d9ecbcc96762cc7e326055447183f43e6a96b`
  - `Cargo.lock`: `703fbcfc7d375ec0af730b0298b28e2ec210270d`
  - `Cargo.toml`: `4e8cae4657687b81e72101cc4698ab6c018bcf26`
  - `rust-toolchain.toml`: `65ee9b915478b31c122a9199eaefca9450ec8a4c`

  WDBX moved from `9d386dd` (the A1 receipt) to this revision through its owner's
  commits. The CI pin `2969d71` is unchanged.

## Command and result

```sh
cd .work/finish-20261027/final/abi
PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR="$PWD/../target" env -u ABI_WDBX_PERSIST ABI_WDBX_PATH=:memory: ./tools/check.sh < /dev/null
```

- `EXIT 0`, `check: all green`.
- Rust: 1,281 executions passed, 0 failed, 1 ignored. The ignored test is
  `native_mac_activation_of_finder`, which is run by hand. It passed once during
  Task 5 on this Mac.
- Python policy: `Ran 200 tests`, `OK (skipped=8)`. This includes the new
  `test_package_install` suite.
- `dependency-scan: RUN cargo-audit (cargo-audit 0.22.2)`, with zero `SKIP` lines.
- `benchmark regression gate: PASS`. CUDA: no `nvcc` on this host.
- Host load average was about 40–110 during the run. No reruns were needed.
- Log SHA-256: `c8a623a213daee8997ec2228d700c00e6e1a1562c888e2acab953bba4522d317`
  (local `.work/finish-20261027/final/gate.log`).

## Rows moved `M` → `U` (implemented, awaiting qualification)

| Row | Evidence | Remaining before Q |
| --- | --- | --- |
| C38 | `crates/abi-control-plane/src/metrics.rs`, `src/operator_http.rs`, `src/authority/metrics_tests.rs`, `src/service/tests/operator.rs` | alerting/export, audit retention, staging operation |
| C39 | `crates/abi-control-plane/RUNBOOKS.md`, `src/service/tests/runbook_drills.rs` | staging/separate-host execution; key rotation does not exist (runbook documents re-init + re-pair) |
| C11 | `crates/abi-agent-host/src/governed_message.rs`, `tests/governed_message.rs`, `tests/adapter_receipts.rs`, `crates/abi-connectors/src/governed_sender.rs` | live Discord/Twilio accounts (E); Abbey lifecycle join; evidence is provider acceptance, not delivery |
| C10 | `crates/abi-agent-host/src/governed_desktop.rs`, `src/governed_desktop/macos.rs`, `tests/governed_desktop.rs` | only application activation exists; capture/click/text are missing; macOS privacy-permission detection; non-macOS stub never compiled |
| C32 | `tools/package/build_macos.sh`, `tools/package/install.sh`, `tools/tests/test_package_install.py` | signing/notarization, Linux/Windows packages, installed full journey (C33) |

## Verdict

This is local Mac source evidence for the five slices above, bound to the
recorded ABI/WDBX pair. No row is qualified (`Q`) by this receipt. WDBX's own
gate and the Abbey gates remain owner-held.
