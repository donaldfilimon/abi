#!/bin/bash
# Validate feature flag consistency across build system files.
#
# Counts feat_* declarations in:
#   - build/options.zig  (CanonicalFlags struct)
#   - build/flags.zig    (FlagCombo struct + validation_matrix entries)
#   - src/core/feature_catalog.zig (unique compile_flag_field values)
#
# Reports mismatches that indicate documentation or code drift.
# Compatible with macOS (BSD grep) and Linux (GNU grep).
set -euo pipefail

PROJECT_DIR="${1:-.}"

# ── Count feat_* fields inside a Zig struct ──────────────────────────────
count_struct_feat_fields() {
    local file="$1" struct_name="$2"
    awk -v name="$struct_name" '
        $0 ~ name " = struct" { inside=1; next }
        inside && /^};/ { inside=0 }
        inside && /feat_[a-z_]+.*: bool/ { count++ }
        END { print count+0 }
    ' "$file"
}

# ── Count feat_* fields in CanonicalFlags (options.zig) ──────────────────
OPTIONS_FILE="$PROJECT_DIR/build/options.zig"
if [[ ! -f "$OPTIONS_FILE" ]]; then
    echo "ERROR: $OPTIONS_FILE not found" >&2; exit 1
fi
OPTIONS_COUNT=$(count_struct_feat_fields "$OPTIONS_FILE" "CanonicalFlags")

# ── Count feat_* fields in FlagCombo (flags.zig) ────────────────────────
FLAGS_FILE="$PROJECT_DIR/build/flags.zig"
if [[ ! -f "$FLAGS_FILE" ]]; then
    echo "ERROR: $FLAGS_FILE not found" >&2; exit 1
fi
COMBO_FIELDS=$(count_struct_feat_fields "$FLAGS_FILE" "FlagCombo")

# Count validation_matrix entries (lines containing .name = "...")
MATRIX_COUNT=$(grep -c '\.name = "' "$FLAGS_FILE" 2>/dev/null || echo 0)

# ── Count UNIQUE catalog flags (feature_catalog.zig) ────────────────────
CATALOG_FILE="$PROJECT_DIR/src/core/feature_catalog.zig"
if [[ ! -f "$CATALOG_FILE" ]]; then
    echo "ERROR: $CATALOG_FILE not found" >&2; exit 1
fi
# Extract unique compile_flag_field values (some features share a flag)
CATALOG_TOTAL=$(grep -c '\.compile_flag_field =' "$CATALOG_FILE" 2>/dev/null || echo 0)
CATALOG_UNIQUE=$(grep '\.compile_flag_field =' "$CATALOG_FILE" | sed 's/.*"\(feat_[a-z_]*\)".*/\1/' | sort -u | wc -l | tr -d ' ')

# ── Internal-only flags (not in catalog) ─────────────────────────────────
# Extract from the internal_allowed_flags array declaration
INTERNAL_COUNT=$(grep 'internal_allowed_flags.*=.*\[' "$OPTIONS_FILE" | grep -o '"feat_[a-z_]*"' | wc -l | tr -d ' ')

# ── Report ───────────────────────────────────────────────────────────────
echo "=== ABI Feature Flag Sync Report ==="
echo ""
echo "  build/options.zig   CanonicalFlags fields:  $OPTIONS_COUNT"
echo "  build/flags.zig     FlagCombo fields:        $COMBO_FIELDS"
echo "  feature_catalog.zig total entries:            $CATALOG_TOTAL"
echo "  feature_catalog.zig unique flags:             $CATALOG_UNIQUE"
echo "  internal-only flags (not in catalog):         $INTERNAL_COUNT"
echo "  build/flags.zig     validation_matrix rows:   $MATRIX_COUNT"
echo ""

ERRORS=0

# CanonicalFlags should equal unique catalog flags + internal-only flags
EXPECTED_OPTIONS=$((CATALOG_UNIQUE + INTERNAL_COUNT))
if [[ "$OPTIONS_COUNT" -ne "$EXPECTED_OPTIONS" ]]; then
    echo "MISMATCH: CanonicalFlags has $OPTIONS_COUNT flags, expected $EXPECTED_OPTIONS (unique_catalog=$CATALOG_UNIQUE + internal=$INTERNAL_COUNT)"
    ERRORS=$((ERRORS + 1))
else
    echo "OK: CanonicalFlags ($OPTIONS_COUNT) = unique catalog ($CATALOG_UNIQUE) + internal ($INTERNAL_COUNT)"
fi

# FlagCombo should match CanonicalFlags (same set of flags)
if [[ "$COMBO_FIELDS" -ne "$OPTIONS_COUNT" ]]; then
    echo "MISMATCH: FlagCombo has $COMBO_FIELDS feat_* fields, CanonicalFlags has $OPTIONS_COUNT"
    ERRORS=$((ERRORS + 1))
else
    echo "OK: FlagCombo ($COMBO_FIELDS) matches CanonicalFlags ($OPTIONS_COUNT)"
fi

echo ""
if [[ "$ERRORS" -eq 0 ]]; then
    echo "PASSED: All flag counts are consistent."
    echo "  Total flags: $OPTIONS_COUNT ($CATALOG_UNIQUE catalog + $INTERNAL_COUNT internal)"
    echo "  Catalog entries: $CATALOG_TOTAL (some features share flags)"
    echo "  Validation combos: $MATRIX_COUNT"
else
    echo "FAILED: $ERRORS mismatch(es) found."
    exit 1
fi
