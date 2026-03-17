#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/zig_toolchain.sh"

# Create a wrapper script that intercepts zig invocations and handles Darwin linker failures
create_darwin_zig_wrapper() {
    local real_zig="$1"
    local wrapper_path="$2"
    local sysroot="${3:-${SDKROOT:-$(xcrun --show-sdk-path 2>/dev/null || echo /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk)}}"
    local macos_ver="${4:-$(sw_vers -productVersion 2>/dev/null || echo 26.0)}"

    cat > "$wrapper_path" << 'WRAPPER_EOF'
#!/bin/bash
set -euo pipefail

REAL_ZIG="__REAL_ZIG__"
SYSROOT="__SYSROOT__"
MACOS_VER="__MACOS_VER__"

find_compiler_rt() {
    local from_stderr
    from_stderr="$(grep -oE '/[^ )]*libcompiler_rt\.a' "$STDERR_FILE" 2>/dev/null | head -1 || true)"
    if [[ -n "$from_stderr" && -f "$from_stderr" ]]; then
        echo "$from_stderr"
    fi
}

find_object_file() {
    local pattern="$1"
    local from_stderr
    from_stderr="$(grep -oE '\.zig-cache/o/[a-f0-9]+/'"$pattern" "$STDERR_FILE" 2>/dev/null | head -1 || true)"
    if [[ -n "$from_stderr" && -f "$from_stderr" ]]; then echo "$from_stderr"; return; fi
    find .zig-cache/o -name "$pattern" -newer "$STDERR_FILE" 2>/dev/null | head -1 || true
}

relink_with_apple_ld() {
    local obj="$1" output="$2"
    local compiler_rt; compiler_rt="$(find_compiler_rt)"

    local rt_args=()
    if [[ -n "$compiler_rt" ]]; then
        rt_args=("$compiler_rt")
    fi

    echo "[darwin-wrapper] Relinking $(basename "$output") with Apple ld..." >&2
    /usr/bin/ld -dynamic \
        -platform_version macos "$MACOS_VER" "$MACOS_VER" \
        -syslibroot "$SYSROOT" \
        -e _main \
        -o "$output" \
        "$obj" \
        -lSystem \
        "${rt_args[@]}" || {
        echo "[darwin-wrapper] Apple ld also failed" >&2
        return 1
    }
}

STDERR_FILE="$(mktemp)"
trap 'rm -f "$STDERR_FILE"' EXIT

if "$REAL_ZIG" "$@" 2>"$STDERR_FILE"; then
    exit 0
else
    ZIG_EXIT=$?
fi

# Check if this is a linker failure we can fix
if ! grep -qE '(undefined.*_arc4random_buf|undefined.*__availability_version|using LLD to link|MachO|lld-link)' "$STDERR_FILE" 2>/dev/null; then
    cat "$STDERR_FILE" >&2
    exit $ZIG_EXIT
fi

# Handle zig build (build runner)
if [[ "${1:-}" == "build" ]]; then
    BUILD_O="$(find_object_file 'build_zcu.o')"
    if [[ -n "$BUILD_O" && -f "$BUILD_O" ]]; then
        BUILD_DIR="$(dirname "$BUILD_O")"
        BUILD_BIN="$BUILD_DIR/build"
        relink_with_apple_ld "$BUILD_O" "$BUILD_BIN" || exit 1

        ZIG_LIB_DIR="$("$REAL_ZIG" env 2>/dev/null | grep '\.lib_dir' | sed 's/.*= *"\(.*\)".*/\1/' || true)"
        if [[ -z "$ZIG_LIB_DIR" ]]; then
            ZIG_LIB_DIR="$(dirname "$(dirname "$REAL_ZIG")")/lib"
        fi

        shift
        exec "$BUILD_BIN" "$REAL_ZIG" "$ZIG_LIB_DIR" "$(pwd)" ".zig-cache" "${HOME}/.cache/zig" "$@"
    fi
fi

# Handle zig build-exe / test / etc - try to find and relink the output
OUTPUT_O="$(find_object_file '*.o')"
if [[ -n "$OUTPUT_O" && -f "$OUTPUT_O" ]]; then
    # Guess output name from zig command or use basename of .o
    OUTPUT_BIN="${OUTPUT_O%.o}"
    if [[ -z "$OUTPUT_BIN" || "$OUTPUT_BIN" == "$OUTPUT_O" ]]; then
        OUTPUT_BIN="$(dirname "$OUTPUT_O")/output"
    fi
    if relink_with_apple_ld "$OUTPUT_O" "$OUTPUT_BIN" 2>/dev/null; then
        echo "[darwin-wrapper] Successfully relinked $OUTPUT_BIN" >&2
        exit 0
    fi
fi

cat "$STDERR_FILE" >&2
exit $ZIG_EXIT
WRAPPER_EOF

    chmod +x "$wrapper_path"
    sed -i.bak "s|__REAL_ZIG__|$real_zig|g; s|__SYSROOT__|$sysroot|g; s|__MACOS_VER__|$macos_ver|g" "$wrapper_path"
    rm -f "$wrapper_path.bak"
}

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

# On Darwin 26+, use a wrapper to handle linker issues during stage3 build
zig_cmd="$zig2_bin"
if [[ "$(uname -s)" == "Darwin" ]]; then
  macos_major="${MACOS_VER%%.*}"
  if [[ "$macos_major" -ge 26 ]]; then
    wrapper_path="$cache_root/zig2_wrapper_$expected_version"
    echo "[bootstrap_host_zig] Creating Darwin wrapper for stage3 build..." >&2
    create_darwin_zig_wrapper "$zig2_bin" "$wrapper_path" "$SYSROOT" "$MACOS_VER"
    zig_cmd="$wrapper_path"
  fi
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

if ! run_zig_build_with_relinked_runner "$zig_cmd" "$worktree_dir" "${stage3_args[@]}"; then
  echo "[bootstrap_host_zig] Build failed, attempting manual link fallback..." >&2

  # On Darwin 26+, try to manually link the zig executable
  if [[ "$(uname -s)" == "Darwin" && "${MACOS_VER%%.*}" -ge 26 ]]; then
    echo "[bootstrap_host_zig] Searching for zig build artifacts..." >&2

    # Look in multiple locations for the zig executable output
    local zig_exe_path=""
    local search_paths=(
      "$worktree_dir/zig-out/bin/zig"
      "$worktree_dir/.zig-cache/o/"*/zig
      "$worktree_dir/.zig-cache/o/"*/bin/zig
      "$cache_root/$expected_version/bin/zig"
    )

    for candidate in "${search_paths[@]}"; do
      if [[ -f "$candidate" && -x "$candidate" ]]; then
        zig_exe_path="$candidate"
        echo "[bootstrap_host_zig] Found existing zig binary: $zig_exe_path" >&2
        break
      fi
    done

    # If no binary found, look for object files to link
    if [[ -z "$zig_exe_path" ]]; then
      local obj_dirs=()
      while IFS= read -r dir; do
        obj_dirs+=("$dir")
      done < <(find "$worktree_dir/.zig-cache/o" -name "*.o" -newer "$worktree_dir/build.zig" -exec dirname {} \; 2>/dev/null | sort -u)

      for obj_dir in "${obj_dirs[@]}"; do
        local main_o=""
        # Look for main entry point objects
        for pattern in zig.o main.o build.o; do
          if [[ -f "$obj_dir/$pattern" ]]; then
            main_o="$obj_dir/$pattern"
            break
          fi
        done

        if [[ -n "$main_o" ]]; then
          echo "[bootstrap_host_zig] Found object directory: $obj_dir (main: $(basename "$main_o"))" >&2

          # Collect all object files in this directory
          local all_objects=()
          while IFS= read -r obj; do
            all_objects+=("$obj")
          done < <(find "$obj_dir" -maxdepth 1 -name "*.o" -type f 2>/dev/null)

          if [[ ${#all_objects[@]} -gt 0 ]]; then
            echo "[bootstrap_host_zig] Collected ${#all_objects[@]} object files" >&2

            # Find compiler_rt
            local comp_rt=""
            for rt_path in "$worktree_dir"/.zig-cache/o/*/libcompiler_rt.a "$build_dir"/libcompiler_rt.a; do
              if [[ -f "$rt_path" ]]; then
                comp_rt="$rt_path"
                echo "[bootstrap_host_zig] Using compiler_rt: $comp_rt" >&2
                break
              fi
            done

            local rt_args=()
            [[ -n "$comp_rt" ]] && rt_args=("$comp_rt")

            mkdir -p "$install_dir/bin"

            echo "[bootstrap_host_zig] Linking zig executable with Apple ld..." >&2
            if /usr/bin/ld -dynamic \
                -platform_version macos "$MACOS_VER" "$MACOS_VER" \
                -syslibroot "$SYSROOT" \
                -e _main \
                -o "$install_dir/bin/zig" \
                "${all_objects[@]}" \
                -lSystem \
                "${rt_args[@]}" 2>/dev/null; then

              echo "[bootstrap_host_zig] Manual link succeeded!" >&2
              chmod +x "$install_dir/bin/zig"
              zig_exe_path="$install_dir/bin/zig"
              break
            else
              echo "[bootstrap_host_zig] Link failed for $obj_dir, trying next..." >&2
            fi
          fi
        fi
      done
    fi

    # If we found or created a binary, copy it to install location
    if [[ -n "$zig_exe_path" && -f "$zig_exe_path" && "$zig_exe_path" != "$install_dir/bin/zig" ]]; then
      mkdir -p "$install_dir/bin"
      cp "$zig_exe_path" "$install_dir/bin/zig"
      chmod +x "$install_dir/bin/zig"
      echo "[bootstrap_host_zig] Copied zig binary to install location" >&2
    fi

    if [[ ! -f "$install_dir/bin/zig" ]]; then
      echo "[bootstrap_host_zig] Could not produce zig binary through manual linking." >&2
    fi
  fi
fi

# Verify the zig binary was produced

zig_bin="$install_dir/bin/zig"
if [[ ! -x "$zig_bin" ]]; then
  echo "ERROR: bootstrap completed without producing '$zig_bin'." >&2
  echo "[bootstrap_host_zig] The helper got through bootstrap setup but the final Zig self-build is still blocked on this Darwin host." >&2
  echo "[bootstrap_host_zig] Fallback evidence remains: ./tools/scripts/run_build.sh typecheck --summary all" >&2
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
