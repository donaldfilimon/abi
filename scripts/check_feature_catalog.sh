#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

catalog_file="src/core/feature_catalog.zig"
if [[ ! -f "$catalog_file" ]]; then
    echo "ERROR: feature catalog missing: $catalog_file"
    exit 1
fi

required_files=(
    "src/core/config/mod.zig"
    "src/core/registry/types.zig"
    "src/core/framework.zig"
    "build/options.zig"
    "build/flags.zig"
    "src/services/tests/parity/mod.zig"
)

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

catalog_entries_features="$tmp_dir/catalog_features.txt"
catalog_entries_flags="$tmp_dir/catalog_flags.txt"
catalog_entries_parity="$tmp_dir/catalog_parity.txt"
catalog_enum_features="$tmp_dir/catalog_enum.txt"
tmp_config_features="$tmp_dir/config_features.txt"
tmp_build_options_flags="$tmp_dir/options_flags.txt"
tmp_flag_combo_flags="$tmp_dir/flag_combo_flags.txt"
catalog_features_sorted="$tmp_dir/catalog_features_sorted.txt"
catalog_flags_sorted="$tmp_dir/catalog_flags_sorted.txt"
catalog_flags_unique="$tmp_dir/catalog_flags_unique.txt"
catalog_enum_sorted="$tmp_dir/catalog_enum_sorted.txt"
config_features_sorted="$tmp_dir/config_features_sorted.txt"
build_options_flags_sorted="$tmp_dir/options_flags_sorted.txt"
flag_combo_flags_sorted="$tmp_dir/flag_combo_flags_sorted.txt"
errors=0

# Internal flags intentionally kept for compatibility and derived behavior.
internal_allowed_flags=(
    "enable_explore"
    "enable_vision"
)

is_allowed_internal() {
    local flag="$1"
    local candidate
    for candidate in "${internal_allowed_flags[@]}"; do
        if [[ "$candidate" == "$flag" ]]; then
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Extract metadata from feature_catalog.
# ---------------------------------------------------------------------------
sed -n '/pub const all = /,/^[[:space:]]*};/p' "$catalog_file" \
    | grep -oE '\.feature[[:space:]]*=[[:space:]]*\.([A-Za-z_][A-Za-z0-9_]*)' \
    | sed -E 's/.*\.feature[[:space:]]*=[[:space:]]*\.([A-Za-z_][A-Za-z0-9_]*)/\1/' \
    > "$catalog_entries_features" \
    || true

sed -n '/pub const all = /,/^[[:space:]]*};/p' "$catalog_file" \
    | grep -oE '\.compile_flag_field[[:space:]]*=[[:space:]]*"([A-Za-z0-9_]+)"' \
    | sed -E 's/.*"([A-Za-z0-9_]+)"/\1/' \
    > "$catalog_entries_flags" \
    || true

sed -n '/pub const all = /,/^[[:space:]]*};/p' "$catalog_file" \
    | grep -oE '\.parity_spec[[:space:]]*=[[:space:]]*\.([A-Za-z_][A-Za-z0-9_]*)' \
    | sed -E 's/.*\.parity_spec[[:space:]]*=[[:space:]]*\.([A-Za-z_][A-Za-z0-9_]*)/\1/' \
    > "$catalog_entries_parity" \
    || true

sed -n '/pub const Feature = enum {/,/^[[:space:]]*};/p' "src/core/feature_catalog.zig" \
    | sed -n '2,$p' \
    | grep -E '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*,' \
    | sed -E 's/^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*),?/\1/' \
    > "$catalog_enum_features" \
    || true

# ---------------------------------------------------------------------------
# Extract feature surfaces from consumers.
# ---------------------------------------------------------------------------
awk '
    /pub const BuildOptions = struct {/,/^[[:space:]]*};/ {
        if ($0 ~ /enable_/) {
            line = $0
            sub(/^[[:space:]]*/, "", line)
            sub(/:.*/, "", line)
            if (line ~ /^enable_[A-Za-z0-9_]+$/) print line
        }
    }
' build/options.zig > "$tmp_build_options_flags" || true

awk '
    /pub const FlagCombo = struct {/,/^[[:space:]]*};/ {
        if ($0 ~ /enable_/) {
            line = $0
            sub(/^[[:space:]]*/, "", line)
            sub(/:.*/, "", line)
            if (line ~ /^enable_[A-Za-z0-9_]+$/) print line
        }
    }
' build/flags.zig > "$tmp_flag_combo_flags" || true

if rg -q "pub const Feature = feature_catalog.Feature" "src/core/config/mod.zig"; then
    cat "$catalog_entries_features" > "$tmp_config_features"
else
    sed -n '/pub const Feature = enum {/,/^[[:space:]]*};/p' "src/core/config/mod.zig" \
        | sed -n '2,$p' \
        | grep -E '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*,' \
        | sed -E 's/^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*),?/\1/' \
        > "$tmp_config_features" \
        || true
fi

# Normalize/derive deterministic ordering and uniqueness.
sort "$catalog_entries_features" > "$catalog_features_sorted"
sort "$catalog_entries_flags" > "$catalog_flags_sorted"
sort -u "$catalog_entries_flags" > "$catalog_flags_unique"
sort "$catalog_enum_features" > "$catalog_enum_sorted"
sort "$tmp_config_features" > "$config_features_sorted"
sort -u "$tmp_build_options_flags" > "$build_options_flags_sorted"
sort -u "$tmp_flag_combo_flags" > "$flag_combo_flags_sorted"

# ---------------------------------------------------------------------------
# Static consumers must still import feature_catalog.
# ---------------------------------------------------------------------------
for f in "${required_files[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required feature-catalog consumer file missing: $f"
        errors=$((errors + 1))
        continue
    fi
    if ! rg -q "feature_catalog" "$f"; then
        echo "ERROR: $f does not reference feature_catalog"
        errors=$((errors + 1))
    fi
done

catalog_feature_count=$(wc -l < "$catalog_entries_features" | tr -d ' ')
catalog_entry_count=$(wc -l < "$catalog_entries_flags" | tr -d ' ')
catalog_parity_count=$(wc -l < "$catalog_entries_parity" | tr -d ' ')
catalog_unique_flag_count=$(wc -l < "$catalog_flags_unique" | tr -d ' ')
catalog_enum_count=$(wc -l < "$catalog_enum_features" | tr -d ' ')
config_feature_count=$(wc -l < "$tmp_config_features" | tr -d ' ')
build_options_count=$(wc -l < "$tmp_build_options_flags" | tr -d ' ')
combo_flag_count=$(wc -l < "$tmp_flag_combo_flags" | tr -d ' ')

if (( catalog_feature_count == 0 )); then
    echo "ERROR: no catalog features parsed"
    errors=$((errors + 1))
fi
if (( catalog_entry_count == 0 )); then
    echo "ERROR: no catalog metadata entries parsed"
    errors=$((errors + 1))
fi
if (( catalog_parity_count == 0 )); then
    echo "ERROR: no catalog parity spec entries parsed"
    errors=$((errors + 1))
fi

if ! cmp -s "$catalog_entries_features" "$catalog_enum_features"; then
    echo "ERROR: feature_catalog.Feature enum does not match all[] feature order"
    errors=$((errors + 1))
fi
if ! cmp -s "$catalog_entries_features" "$tmp_config_features"; then
    echo "ERROR: src/core/config/mod.zig Feature enum does not match catalog feature order"
    errors=$((errors + 1))
fi

if (( catalog_feature_count != catalog_enum_count )); then
    echo "ERROR: catalog feature enum cardinality differs (${catalog_feature_count} vs ${catalog_enum_count})"
    errors=$((errors + 1))
fi
if (( catalog_feature_count != config_feature_count )); then
    echo "ERROR: config feature enum cardinality differs (${catalog_feature_count} vs ${config_feature_count})"
    errors=$((errors + 1))
fi
if (( catalog_feature_count != catalog_parity_count )); then
    echo "ERROR: catalog parity spec cardinality differs (${catalog_feature_count} vs ${catalog_parity_count})"
    errors=$((errors + 1))
fi

catalog_feature_dupes="$(sort "$catalog_entries_features" | uniq -d || true)"
if [[ -n "$catalog_feature_dupes" ]]; then
    echo "ERROR: duplicate feature(s) in feature_catalog metadata:"
    echo "$catalog_feature_dupes"
    errors=$((errors + 1))
fi

flag_dupes="$(sort "$catalog_entries_flags" | uniq -d || true)"
if [[ -n "$flag_dupes" ]]; then
    echo "INFO: duplicate compile flags in feature_catalog (expected for derived/inherited toggles):"
    echo "$flag_dupes"
fi

if (( catalog_unique_flag_count == 0 )); then
    echo "ERROR: no catalog compile flags parsed"
    errors=$((errors + 1))
fi

# Canonical compile flags in catalog must be present on both feature surfaces.
while IFS= read -r flag; do
    [[ -z "$flag" ]] && continue
    if ! grep -Fxq "$flag" "$build_options_flags_sorted"; then
        echo "ERROR: BuildOptions missing catalog flag '$flag'"
        errors=$((errors + 1))
    fi
    if ! grep -Fxq "$flag" "$flag_combo_flags_sorted"; then
        echo "ERROR: FlagCombo missing catalog flag '$flag'"
        errors=$((errors + 1))
    fi
done < "$catalog_flags_unique"

# Non-catalog flags are allowed only for known internal toggles.
while IFS= read -r flag; do
    [[ -z "$flag" ]] && continue
    if ! grep -Fxq "$flag" "$catalog_flags_unique"; then
        if ! is_allowed_internal "$flag"; then
            echo "ERROR: BuildOptions contains unknown flag '$flag' not derived from catalog"
            errors=$((errors + 1))
        fi
    fi
done < "$build_options_flags_sorted"
while IFS= read -r flag; do
    [[ -z "$flag" ]] && continue
    if ! grep -Fxq "$flag" "$catalog_flags_unique"; then
        if ! is_allowed_internal "$flag"; then
            echo "ERROR: FlagCombo contains unknown flag '$flag' not derived from catalog"
            errors=$((errors + 1))
        fi
    fi
done < "$flag_combo_flags_sorted"

if (( errors > 0 )); then
    echo "FAILED: Feature catalog audit found $errors issue(s)"
    exit 1
fi

echo "OK: Feature catalog audit passed"
