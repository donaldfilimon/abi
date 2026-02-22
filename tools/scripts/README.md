# tools/scripts/

Native Zig replacements for former shell-based CI quality gates.

## Scripts

| Script | Purpose | Called by |
| --- | --- | --- |
| `check_feature_catalog.zig` | Verify feature catalog parity across config/build surfaces | `zig build check-consistency` |
| `check_gpu_policy_consistency.zig` | Verify build-time and runtime GPU policy tables agree | `zig build check-consistency` |
| `check_import_rules.zig` | Enforce no circular `@import("abi")` usage in feature modules | `zig build check-imports` |
| `check_ralph_gate.zig` | Require live Ralph gate input and score threshold | `zig build ralph-gate` |
| `check_test_baseline_consistency.zig` | Verify baseline numbers in docs remain synchronized | `zig build check-consistency` |
| `check_zig_016_patterns.zig` | Scan for deprecated pre-0.16 Zig patterns | `zig build check-consistency` |
| `check_zig_version_consistency.zig` | Verify `.zigversion` and docs match pinned Zig version | `zig build check-consistency` |
| `cli_full_preflight.zig` | Check required env vars and tools before full CLI matrix run | `run_cli_full_matrix.py` |
| `run_cli_full_matrix.py` | Run exhaustive CLI matrix with isolation and reporting | `zig build cli-tests-full` |
| `run_cli_full_matrix_ci.sh` | CI/local wrapper that injects stub env and tolerates blocked checks | `run_cli_full_matrix.py` |
| `toolchain_doctor.zig` | Diagnose PATH precedence and active Zig mismatch | `zig build toolchain-doctor` |
| `validate_test_counts.zig` | Run tests and verify count baselines | `zig build validate-baseline` |

## Support Files

| File | Purpose |
| --- | --- |
| `baseline.zig` | Canonical metadata: pinned Zig version, main/feature test baseline counts |
| `cli_full_env.template` | Template `.env` file for full CLI matrix credentials (copy and fill locally) |
| `cli_full_env.stub` | Safe placeholder `.env` for local or CI-like matrix runs |
| `util.zig` | Shared utility functions imported by other scripts |

## Recommended CI/Local Invocation

For local/offline matrix runs where external credentials or connectivity are unavailable:

```bash
tools/scripts/run_cli_full_matrix_ci.sh --timeout-scale 1.5
```

The CI/local wrapper:
- Loads `tools/scripts/cli_full_env.stub` as environment input.
- Runs preflight in non-blocking mode, classifying missing prerequisites as blocked rows.
- Preserves the same JSON/Markdown report outputs under `/tmp`.
- Uses short PTY probe windows to avoid long hangs in interactive vectors.

Equivalent build-style invocation:

```bash
zig build cli-tests-full \\
  -Dcli-full-env-file=tools/scripts/cli_full_env.stub \\
  -Dcli-full-allow-blocked=true
```

Nested-only matrix run:

```bash
python3 tools/scripts/run_cli_full_matrix.py \\
  --repo /Users/donaldfilimon/abi \\
  --id-prefix nested. \\
  --allow-blocked \\
  --env-file tools/scripts/cli_full_env.stub \\
  --pty-probe-window 8
```

Keep isolated temp workspace for debugging failed vectors:

```bash
python3 tools/scripts/run_cli_full_matrix.py \\
  --repo /Users/donaldfilimon/abi \\
  --allow-blocked \\
  --env-file tools/scripts/cli_full_env.stub \\
  --keep-temp
```

Build shortcut for nested vectors:

```bash
zig build cli-tests-nested \\
  -Dcli-full-env-file=tools/scripts/cli_full_env.stub \\
  -Dcli-full-allow-blocked=true
```

Update `baseline.zig` only after re-validating baselines in CI/local gates.
