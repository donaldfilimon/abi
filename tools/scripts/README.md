# tools/scripts/

Native Zig replacements for former shell-based CI quality gates.

## Scripts

| Script | Purpose | Called by |
| --- | --- | --- |
| `check_feature_catalog.zig` | Verify feature catalog parity across config/build surfaces | `zig build check-consistency` |
| `check_import_rules.zig` | Enforce no circular `@import("abi")` usage in feature modules | `zig build check-imports` |
| `check_test_baseline_consistency.zig` | Verify baseline numbers in docs remain synchronized | `zig build check-consistency` |
| `check_zig_version_consistency.zig` | Verify `.zigversion` and docs match pinned Zig version | `zig build check-consistency` |
| `check_zig_016_patterns.zig` | Scan for deprecated pre-0.16 Zig patterns | `zig build check-consistency` |
| `check_ralph_gate.zig` | Require live Ralph gate input and score threshold | `zig build ralph-gate` |
| `validate_test_counts.zig` | Run tests and verify count baselines | `zig build validate-baseline` |
| `toolchain_doctor.zig` | Diagnose PATH precedence and active Zig mismatch | `zig build toolchain-doctor` |

## Baseline Source of Truth

`baseline.zig` is the canonical metadata file for:
- pinned Zig version
- main test baseline counts
- feature test baseline counts

Update `baseline.zig` only after re-validating baselines in CI/local gates.
