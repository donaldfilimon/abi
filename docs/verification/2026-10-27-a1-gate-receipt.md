# A1 bound gate receipt — 2026-10-27

Plan: `docs/plans/2026-10-26-a1-remainder-a2-relay-requalification.md`, Task 1
(and, because no source changed after Task 7b, the Task 8 Step 2 gate). Ledger:
[`2026-10-27-c01-c44-execution-ledger.md`](2026-10-27-c01-c44-execution-ledger.md).

## Source set

The gate ran on local `git clone --shared` copies under the ignored
`.work/finish-20261027/base/{abi,wdbx}`, so an active WDBX writer could not move
the consumed sources during the run. These are not the canonical checkout paths.

- ABI `HEAD`: `1dd7f0bf73c644b75d4b65b9ce2574bef1872cf3`. Status before and after:
  clean (no entries).
- WDBX `HEAD`: `9d386ddbe57c9e8423dc4755f231e1387a0cd1f7`. Object IDs:
  - `crates`: `c75e7e64da03df7c220dca3459816b20ebe7e7f6`
  - `Cargo.lock`: `703fbcfc7d375ec0af730b0298b28e2ec210270d`
  - `Cargo.toml`: `4e8cae4657687b81e72101cc4698ab6c018bcf26`
  - `rust-toolchain.toml`: `65ee9b915478b31c122a9199eaefca9450ec8a4c`
- Consumed-path status in the clone: empty, before and after. Other dirty paths
  in the clone: none.
- Pre- and post-gate captures are identical (`abi-stable`, `wdbx-stable`).
- The canonical `../wdbx` sits at the same `HEAD`, with another owner's
  uncommitted edits. Those edits were not consumed:
  `crates/abi-wdbx/src/model/contract/tokenizer.rs`,
  `crates/abi-wdbx/src/model/package/components/tokenizer.rs`,
  `crates/abi-wdbx/tests/fixtures/model-v1/n02-fixtures.md`,
  `crates/abi-wdbx/tests/model_package.rs`, `docs/contracts/model/tokenizer-v1.md`,
  `tools/tests/test_wdbx_model_reference.py`, and untracked
  `crates/abi-wdbx/tests/fixtures/model-v1/tokenizer-profile-negatives-v1.json`.
- WDBX has moved from the matrix-recorded
  `96c4b131faed45eadcaef215a31af53776282f81` to `9d386dd`.
- The CI pin `2969d71a2c3a81a060dcfc6cf53b4617f13c8b94` is unchanged.
- Abbey (observed only; owner gates not run): `0ad2fef80f0c1ec38dd1367882541e6a2cdec661`,
  12 dirty entries.

## Toolchain

`rustc 1.99.1 (58cf8f9c3 2026-10-13)`, `cargo 1.99.1 (5f94df478 2026-08-27)`,
host `aarch64-apple-darwin`, `cargo-audit 0.22.2`.

## Command and result

```sh
cd .work/finish-20261027/base/abi
CARGO_TARGET_DIR="$PWD/../target" env -u ABI_WDBX_PERSIST ABI_WDBX_PATH=:memory: ./tools/check.sh < /dev/null
```

- `EXIT 0` and `check: all green`.
- Rust: 1,248 test executions passed, 0 failed, 0 ignored. The last recorded
  count was 1,241 at `aaf2929`; the A2 tests account for the difference.
- Python policy: `Ran 186 tests`, `OK (skipped=8)`.
- `dependency-scan: RUN cargo-audit (cargo-audit 0.22.2)`, with the two existing
  scoped exceptions and zero `SKIP` lines.
- `benchmark regression gate: PASS`.
- CUDA: `CUDA feature compilation unavailable: nvcc is not installed`.
- Child-process panics printed by the `reconnect_process::recovery_child`
  fault-injection fixtures are expected. Every parent test passed.
- Host load average during the run was about 80–110, from other sessions'
  builds. No test needed a rerun.
- Log SHA-256: `9b0ef1827685c7c16984419f3142e43e233c71f75ca75ebdba9cd3f712601877`
  (local `.work/finish-20261027/base/gate.log`).

## Verdict

This is local source evidence for C02, bound to the recorded ABI/WDBX pair. It
is not WDBX's own gate, not Abbey's gates, and not a frozen release cohort. C42
stays external.
