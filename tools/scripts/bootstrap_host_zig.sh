#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/zig_toolchain.sh"

find_compiler_rt() {
  local stderr_file="$1"
  local from_stderr
  from_stderr="$(grep -oE '/[^ )]*libcompiler_rt\.a' "$stderr_file" | head -1 || true)"
  if [[ -n "$from_stderr" && -f "$from_stderr" ]]; then
    printf '%s\n' "$from_stderr"
  fi
}

run_zig_build_with_relinked_runner() {
  local zig_bin="$1"
  local work_dir="$2"
  shift 2

  local stderr_file
  stderr_file="$(mktemp)"
  trap 'rm -f "$stderr_file"' RETURN

  (
    cd "$work_dir"

    if "$zig_bin" build "$@" 2>"$stderr_file"; then
      exit 0
    fi

    local build_o
    build_o="$(grep -oE '\.zig-cache/o/[a-f0-9]+/build_zcu\.o' "$stderr_file" | head -1 || true)"
    if [[ -z "$build_o" ]]; then
      build_o="$(find .zig-cache/o -name 'build_zcu.o' -newer "$stderr_file" -print -quit 2>/dev/null || true)"
    fi

    if [[ -z "$build_o" || ! -f "$build_o" ]]; then
      cat "$stderr_file" >&2
      exit 1
    fi

    local build_dir build_bin compiler_rt zig_lib_dir
    build_dir="$(dirname "$build_o")"
    build_bin="$build_dir/build"
    compiler_rt="$(find_compiler_rt "$stderr_file")"

    local rt_args=()
    if [[ -n "$compiler_rt" ]]; then
      rt_args=("$compiler_rt")
    fi

    echo "[bootstrap_host_zig] relinking stage3 build runner with Apple ld..." >&2
    echo "[bootstrap_host_zig]   obj: $build_o" >&2
    echo "[bootstrap_host_zig]   sdk: $SYSROOT" >&2

    /usr/bin/ld -dynamic \
      -platform_version macos "$MACOS_VER" "$MACOS_VER" \
      -syslibroot "$SYSROOT" \
      -e _main \
      -o "$build_bin" \
      "$build_o" \
      -lSystem \
      "${rt_args[@]}"

    zig_lib_dir=""
    local runner_args=()
    local arg_index=1
    while [[ $arg_index -le $# ]]; do
      if [[ "${!arg_index}" == "--zig-lib-dir" ]]; then
        local next_index=$((arg_index + 1))
        if [[ $next_index -le $# ]]; then
          zig_lib_dir="${!next_index}"
        fi
        arg_index=$((arg_index + 2))
        continue
      fi
      runner_args+=("${!arg_index}")
      arg_index=$((arg_index + 1))
    done
    if [[ -z "$zig_lib_dir" ]]; then
      zig_lib_dir="$work_dir/lib"
    fi

    exec "$build_bin" "$zig_bin" "$zig_lib_dir" "$work_dir" "$work_dir/.zig-cache" "${HOME}/.cache/zig" "${runner_args[@]}"
  )
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: ./tools/scripts/bootstrap_host_zig.sh

Bootstraps the Zig commit pinned by .zigversion into the canonical ABI host-Zig
cache directory so Darwin hosts can run full local zig build gates.

Environment:
  ABI_ZIG_SOURCE_DIR       Zig source checkout to build from (default: $HOME/zig)
  ABI_HOST_ZIG_CACHE_DIR   Install/cache root (default: $HOME/.cache/abi-host-zig)
EOF
  exit 0
fi

repo_root="$(abi_toolchain_repo_root)"
cd "$repo_root"

expected_version="$(abi_toolchain_expected_version)"
commit_hint="$(abi_toolchain_commit_from_version "$expected_version")"
source_dir="$(abi_toolchain_source_dir)"
cache_root="$(abi_toolchain_cache_root)"
worktree_dir="$cache_root/worktrees/$expected_version"
build_dir="$cache_root/build/$expected_version"
install_dir="$cache_root/$expected_version"
SYSROOT="${SDKROOT:-$(xcrun --show-sdk-path 2>/dev/null || echo /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk)}"
MACOS_VER="$(sw_vers -productVersion 2>/dev/null || echo 26.0)"

if [[ ! -d "$source_dir" ]]; then
  echo "ERROR: Zig source checkout not found at '$source_dir'." >&2
  echo "Set ABI_ZIG_SOURCE_DIR or create a checkout there before bootstrapping." >&2
  exit 1
fi

if ! git -C "$source_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: '$source_dir' is not a Zig git checkout." >&2
  exit 1
fi

source_dir="$(cd "$source_dir" && pwd -P)"
mkdir -p "$cache_root/worktrees" "$cache_root/build"

full_commit="$(git -C "$source_dir" rev-parse --verify "${commit_hint}^{commit}" 2>/dev/null || true)"
if [[ -z "$full_commit" ]]; then
  if git -C "$source_dir" remote get-url origin >/dev/null 2>&1; then
    echo "[bootstrap_host_zig] Commit $commit_hint not found locally; fetching origin..." >&2
    git -C "$source_dir" fetch --tags origin >&2
    full_commit="$(git -C "$source_dir" rev-parse --verify "${commit_hint}^{commit}" 2>/dev/null || true)"
  fi
fi

if [[ -z "$full_commit" ]]; then
  echo "ERROR: could not resolve Zig commit '$commit_hint' from .zigversion ($expected_version)." >&2
  echo "Ensure '$source_dir' has that commit locally, then rerun this helper." >&2
  exit 1
fi

echo "[bootstrap_host_zig] source checkout: $source_dir" >&2
echo "[bootstrap_host_zig] pinned Zig:      $expected_version" >&2
echo "[bootstrap_host_zig] source commit:   $full_commit" >&2
echo "[bootstrap_host_zig] install prefix:  $install_dir" >&2

git -C "$source_dir" worktree prune >&2 || true
if git -C "$source_dir" worktree list --porcelain | grep -Fqx "worktree $worktree_dir"; then
  git -C "$source_dir" worktree remove --force "$worktree_dir" >&2
fi

rm -rf "$worktree_dir" "$build_dir" "$install_dir"
git -C "$source_dir" worktree add --detach "$worktree_dir" "$full_commit" >&2

jobs="$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

cmake_args=(
  -S "$worktree_dir"
  -B "$build_dir"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="$install_dir"
)
prefix_entries=()

if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
  link_flags=()
  include_flags=()
  for formula in "" llvm@21 zstd zlib; do
    if [[ -n "$formula" ]]; then
      prefix="$(brew --prefix "$formula" 2>/dev/null || true)"
    else
      prefix="$(brew --prefix 2>/dev/null || true)"
    fi
    [[ -n "$prefix" && -d "$prefix" ]] || continue
    prefix_entries+=("$prefix")
    [[ -d "$prefix/lib" ]] && link_flags+=("-L$prefix/lib")
    [[ -d "$prefix/include" ]] && include_flags+=("-I$prefix/include")
  done

  if (( ${#prefix_entries[@]} > 0 )); then
    prefix_path="$(IFS=';'; printf '%s' "${prefix_entries[*]}")"
    cmake_args+=("-DCMAKE_PREFIX_PATH=$prefix_path")
  fi

  if (( ${#link_flags[@]} > 0 )); then
    joined_link_flags="${link_flags[*]}"
    export LDFLAGS="${joined_link_flags} ${LDFLAGS:-}"
    cmake_args+=("-DCMAKE_EXE_LINKER_FLAGS=$joined_link_flags")
    cmake_args+=("-DCMAKE_SHARED_LINKER_FLAGS=$joined_link_flags")
  fi

  if (( ${#include_flags[@]} > 0 )); then
    joined_include_flags="${include_flags[*]}"
    export CPPFLAGS="${joined_include_flags} ${CPPFLAGS:-}"
    cmake_args+=("-DCMAKE_C_FLAGS=$joined_include_flags")
    cmake_args+=("-DCMAKE_CXX_FLAGS=$joined_include_flags")
  fi
fi

cmake "${cmake_args[@]}"

cmake --build "$build_dir" --target zig2 --parallel "$jobs"

zig2_bin="$build_dir/zig2"
if [[ ! -x "$zig2_bin" ]]; then
  echo "ERROR: bootstrap build did not produce '$zig2_bin'." >&2
  exit 1
fi

stage3_args=(
  --prefix "$install_dir"
  --zig-lib-dir "$worktree_dir/lib"
  --sysroot "$SYSROOT"
  "-Dversion-string=$expected_version"
  -Dtarget=native
  -Dcpu=native
  -Denable-llvm
  "-Dconfig_h=$build_dir/config.h"
  -Dno-langref
  -Doptimize=ReleaseFast
  -Dstrip
)

for prefix in "${prefix_entries[@]}"; do
  stage3_args+=(--search-prefix "$prefix")
done

if ! run_zig_build_with_relinked_runner "$zig2_bin" "$worktree_dir" "${stage3_args[@]}"; then
  echo "[bootstrap_host_zig] ERROR: failed to produce the pinned host-built Zig for $expected_version." >&2
  echo "[bootstrap_host_zig] The helper got through bootstrap setup but the final Zig self-build is still blocked on this Darwin host." >&2
  echo "[bootstrap_host_zig] Fallback evidence remains: ./tools/scripts/run_build.sh typecheck --summary all" >&2
  exit 1
fi

zig_bin="$install_dir/bin/zig"
if [[ ! -x "$zig_bin" ]]; then
  echo "ERROR: bootstrap completed without producing '$zig_bin'." >&2
  exit 1
fi

actual_version="$(abi_toolchain_binary_version "$zig_bin" || true)"
if [[ "$actual_version" != "$expected_version" ]]; then
  echo "ERROR: bootstrapped Zig version mismatch." >&2
  echo "Expected: $expected_version" >&2
  echo "Actual:   ${actual_version:-<unreadable>}" >&2
  exit 1
fi

cat <<EOF
Bootstrapped pinned Zig $actual_version
Binary: $zig_bin

Use it for direct zig build validation on macOS:
  export PATH="$install_dir/bin:\$PATH"
  hash -r
  zig build toolchain-doctor
  zig build full-check
  zig build check-docs

Optional helper-script override:
  export ABI_HOST_ZIG="$zig_bin"
EOF
