#!/usr/bin/env bash

ABI_TOOLCHAIN_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ABI_TOOLCHAIN_REPO_ROOT="$(git -C "$ABI_TOOLCHAIN_LIB_DIR" rev-parse --show-toplevel 2>/dev/null || printf '%s\n' "$ABI_TOOLCHAIN_LIB_DIR")"

abi_toolchain_repo_root() {
  printf '%s\n' "$ABI_TOOLCHAIN_REPO_ROOT"
}

abi_toolchain_expected_version() {
  if [[ -n "${ABI_TOOLCHAIN_EXPECTED_VERSION:-}" ]]; then
    printf '%s\n' "$ABI_TOOLCHAIN_EXPECTED_VERSION"
    return 0
  fi

  local version_file version
  version_file="$ABI_TOOLCHAIN_REPO_ROOT/.zigversion"
  if [[ ! -f "$version_file" ]]; then
    echo "ERROR: missing .zigversion at $version_file" >&2
    return 1
  fi

  version="$(awk 'NR == 1 { gsub(/\r/, ""); sub(/[[:space:]]+$/, ""); print; exit }' "$version_file")"
  if [[ -z "$version" ]]; then
    echo "ERROR: empty .zigversion at $version_file" >&2
    return 1
  fi

  ABI_TOOLCHAIN_EXPECTED_VERSION="$version"
  printf '%s\n' "$ABI_TOOLCHAIN_EXPECTED_VERSION"
}

abi_toolchain_commit_from_version() {
  local version="${1:?expected Zig version}"
  if [[ "$version" != *+* ]]; then
    echo "ERROR: Zig version '$version' does not include a commit suffix" >&2
    return 1
  fi
  printf '%s\n' "${version##*+}"
}

abi_toolchain_cache_root() {
  if [[ -n "${ABI_HOST_ZIG_CACHE_DIR:-}" ]]; then
    printf '%s\n' "$ABI_HOST_ZIG_CACHE_DIR"
    return 0
  fi
  if [[ -z "${HOME:-}" ]]; then
    echo "ERROR: HOME is unset and ABI_HOST_ZIG_CACHE_DIR was not provided" >&2
    return 1
  fi
  printf '%s/.cache/abi-host-zig\n' "$HOME"
}

abi_toolchain_source_dir() {
  if [[ -n "${ABI_ZIG_SOURCE_DIR:-}" ]]; then
    printf '%s\n' "$ABI_ZIG_SOURCE_DIR"
    return 0
  fi
  if [[ -z "${HOME:-}" ]]; then
    echo "ERROR: HOME is unset and ABI_ZIG_SOURCE_DIR was not provided" >&2
    return 1
  fi
  printf '%s/zig\n' "$HOME"
}

abi_toolchain_canonical_host_zig() {
  local version="${1:-}"
  if [[ -z "$version" ]]; then
    version="$(abi_toolchain_expected_version)" || return 1
  fi
  printf '%s/%s/bin/zig\n' "$(abi_toolchain_cache_root)" "$version"
}

abi_toolchain_realpath() {
  local target="${1:?expected path}"
  if command -v realpath >/dev/null 2>&1; then
    realpath "$target"
    return
  fi
  if [[ ! -e "$target" ]]; then
    return 1
  fi
  local dir base
  dir="$(cd "$(dirname "$target")" && pwd -P)" || return 1
  base="$(basename "$target")"
  printf '%s/%s\n' "$dir" "$base"
}

abi_toolchain_resolve_candidate() {
  local candidate="${1:?expected Zig candidate}"
  local resolved
  if [[ "$candidate" == */* ]]; then
    [[ -x "$candidate" ]] || return 1
    abi_toolchain_realpath "$candidate" 2>/dev/null || printf '%s\n' "$candidate"
    return 0
  fi

  resolved="$(command -v "$candidate" 2>/dev/null)" || return 1
  abi_toolchain_realpath "$resolved" 2>/dev/null || printf '%s\n' "$resolved"
}

abi_toolchain_binary_version() {
  local zig_bin="${1:?expected Zig binary}"
  "$zig_bin" version 2>/dev/null | tr -d '\r'
}

abi_toolchain_emit_inspection() {
  local expected_version cache_root cache_path
  expected_version="$(abi_toolchain_expected_version)" || return 1
  cache_root="$(abi_toolchain_cache_root)" || return 1
  cache_path="$(abi_toolchain_canonical_host_zig "$expected_version")" || return 1

  local cache_exists=0 cache_version="" cache_matches_expected=0
  if [[ -e "$cache_path" ]]; then
    cache_exists=1
    if [[ -x "$cache_path" ]]; then
      cache_version="$(abi_toolchain_binary_version "$cache_path" || true)"
      if [[ "$cache_version" == "$expected_version" ]]; then
        cache_matches_expected=1
      fi
    fi
  fi

  local selected_status="no_zig_found"
  local selected_source="none"
  local selected_path=""
  local selected_version=""
  local selected_matches_expected=0
  local selected_env_name=""

  if [[ -n "${ABI_HOST_ZIG:-}" ]]; then
    selected_source="abi_host_zig"
    selected_env_name="ABI_HOST_ZIG"
    selected_path="$ABI_HOST_ZIG"
    if selected_path="$(abi_toolchain_resolve_candidate "$ABI_HOST_ZIG" 2>/dev/null)"; then
      selected_version="$(abi_toolchain_binary_version "$selected_path" || true)"
      if [[ "$selected_version" == "$expected_version" ]]; then
        selected_status="ok"
        selected_matches_expected=1
      else
        selected_status="abi_host_zig_mismatch"
      fi
    else
      selected_path="$ABI_HOST_ZIG"
      selected_status="abi_host_zig_missing"
    fi
  elif [[ -n "${ZIG_REAL:-}" ]]; then
    selected_source="zig_real"
    selected_env_name="ZIG_REAL"
    selected_path="$ZIG_REAL"
    if selected_path="$(abi_toolchain_resolve_candidate "$ZIG_REAL" 2>/dev/null)"; then
      selected_version="$(abi_toolchain_binary_version "$selected_path" || true)"
      selected_status="ok"
      if [[ "$selected_version" == "$expected_version" ]]; then
        selected_matches_expected=1
      fi
    else
      selected_path="$ZIG_REAL"
      selected_status="zig_real_missing"
    fi
  elif [[ -n "${ZIG:-}" ]]; then
    selected_source="zig_env"
    selected_env_name="ZIG"
    selected_path="$ZIG"
    if selected_path="$(abi_toolchain_resolve_candidate "$ZIG" 2>/dev/null)"; then
      selected_version="$(abi_toolchain_binary_version "$selected_path" || true)"
      selected_status="ok"
      if [[ "$selected_version" == "$expected_version" ]]; then
        selected_matches_expected=1
      fi
    else
      selected_path="$ZIG"
      selected_status="zig_missing"
    fi
  elif [[ "$cache_exists" == "1" ]]; then
    selected_source="cache"
    selected_path="$cache_path"
    selected_version="$cache_version"
    if [[ "$cache_matches_expected" == "1" ]]; then
      selected_status="ok"
      selected_matches_expected=1
    else
      selected_status="cache_stale"
    fi
  else
    if selected_path="$(abi_toolchain_resolve_candidate zig 2>/dev/null)"; then
      selected_source="path"
      selected_version="$(abi_toolchain_binary_version "$selected_path" || true)"
      selected_status="ok"
      if [[ "$selected_version" == "$expected_version" ]]; then
        selected_matches_expected=1
      fi
    fi
  fi

  printf 'expected_version=%s\n' "$expected_version"
  printf 'cache_root=%s\n' "$cache_root"
  printf 'cache_path=%s\n' "$cache_path"
  printf 'cache_exists=%s\n' "$cache_exists"
  printf 'cache_version=%s\n' "$cache_version"
  printf 'cache_matches_expected=%s\n' "$cache_matches_expected"
  printf 'selected_status=%s\n' "$selected_status"
  printf 'selected_source=%s\n' "$selected_source"
  printf 'selected_env_name=%s\n' "$selected_env_name"
  printf 'selected_path=%s\n' "$selected_path"
  printf 'selected_version=%s\n' "$selected_version"
  printf 'selected_matches_expected=%s\n' "$selected_matches_expected"
}

abi_toolchain_resolve_active_zig() {
  local line key value
  local expected_version="" cache_path="" cache_exists="" cache_matches_expected=""
  local selected_status="" selected_source="" selected_path="" selected_version="" selected_matches_expected=""

  while IFS= read -r line; do
    key="${line%%=*}"
    value="${line#*=}"
    case "$key" in
      expected_version) expected_version="$value" ;;
      cache_path) cache_path="$value" ;;
      cache_exists) cache_exists="$value" ;;
      cache_matches_expected) cache_matches_expected="$value" ;;
      selected_status) selected_status="$value" ;;
      selected_source) selected_source="$value" ;;
      selected_path) selected_path="$value" ;;
      selected_version) selected_version="$value" ;;
      selected_matches_expected) selected_matches_expected="$value" ;;
    esac
  done < <(abi_toolchain_emit_inspection)

  case "$selected_status" in
    ok)
      if [[ -n "$selected_path" ]]; then
        printf '%s\n' "$selected_path"
        return 0
      fi
      ;;
    abi_host_zig_missing)
      echo "ERROR: ABI_HOST_ZIG points to '$selected_path', but that binary is not executable." >&2
      return 1
      ;;
    abi_host_zig_mismatch)
      echo "ERROR: ABI_HOST_ZIG resolved to '$selected_path' ($selected_version), expected $expected_version." >&2
      echo "Rebuild the pinned compiler with ./tools/scripts/bootstrap_host_zig.sh or point ABI_HOST_ZIG at a matching Zig." >&2
      return 1
      ;;
    zig_real_missing)
      echo "ERROR: ZIG_REAL points to '$selected_path', but that binary is not executable." >&2
      return 1
      ;;
    zig_missing)
      echo "ERROR: ZIG points to '$selected_path', but that binary is not executable." >&2
      return 1
      ;;
    cache_stale)
      echo "ERROR: cached host-built Zig at '$cache_path' is stale (found '$selected_version', expected '$expected_version')." >&2
      echo "Rebuild it with ./tools/scripts/bootstrap_host_zig.sh." >&2
      return 1
      ;;
  esac

  echo "ERROR: no usable Zig compiler was found. Run ./tools/scripts/bootstrap_host_zig.sh or put Zig $expected_version on PATH." >&2
  return 1
}
