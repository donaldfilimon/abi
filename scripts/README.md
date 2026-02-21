# scripts/

CI quality-gate and development helper scripts.

## CI Scripts (called from `.github/workflows/ci.yml` and `zig build`)

| Script | Purpose | Called by |
|--------|---------|----------|
| `check_feature_catalog.sh` | Verify feature catalog matches build options, config, and flag combos | `zig build check-consistency`, CI |
| `check_import_rules.sh` | Enforce no circular `@import("abi")` in feature modules | `zig build check-imports`, CI |
| `check_test_baseline_consistency.sh` | Verify baseline numbers match across all doc files | `zig build check-consistency`, CI |
| `check_zig_version_consistency.sh` | Verify `.zigversion` matches `build.zig` and docs | `zig build check-consistency`, CI |
| `check_zig_016_patterns.sh` | Scan for deprecated pre-0.16 Zig patterns | `zig build check-consistency`, CI |
| `check_ralph_gate.sh` | Validate Ralph scoring report (when present) | `zig build ralph-gate`, CI |
| `validate_test_counts.sh` | Run tests and verify counts match `project_baseline.env` | `zig build validate-baseline` |

## Development Helpers

| Script | Purpose |
|--------|---------|
| `toolchain_doctor.sh` | Diagnose PATH precedence and active Zig mismatch |

## Configuration

| File | Purpose |
|------|---------|
| `project_baseline.env` | Canonical test baseline numbers (source of truth for CI) |
| `ralph_prompts_upgrade.json` | Prompt set for Ralph upgrade analysis runs |
