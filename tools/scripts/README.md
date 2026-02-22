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
| `run_cli_full_matrix.py` | Run exhaustive CLI command matrix with isolation and reporting | `zig build cli-tests-full` |
| `toolchain_doctor.zig` | Diagnose PATH precedence and active Zig mismatch | `zig build toolchain-doctor` |
| `validate_test_counts.zig` | Run tests and verify count baselines | `zig build validate-baseline` |

## Support Files

| File | Purpose |
| --- | --- |
| `baseline.zig` | Canonical metadata: pinned Zig version, main/feature test baseline counts |
| `cli_full_env.template` | Template `.env` file for full CLI matrix credentials (copy and fill locally) |
| `util.zig` | Shared utility functions imported by other scripts |

Update `baseline.zig` only after re-validating baselines in CI/local gates.
