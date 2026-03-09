#!/usr/bin/env bash
# .cel toolchain configuration — single source of truth for upstream pin
#
# CEL (Custom Environment Linker) provides a patched Zig compiler for
# macOS 26+ (Tahoe) that fixes the upstream linker incompatibility.
#
# Consistency contract:
#   - ZIG_VERSION must match .zigversion and build.zig.zon minimum_zig_version
#   - ZIG_UPSTREAM_COMMIT must match the commit hash suffix in ZIG_VERSION
#   - Patches in patches/ must apply cleanly against ZIG_UPSTREAM_COMMIT
#
# Migration: run ./tools/scripts/cel_migrate.sh for guided setup

ZIG_UPSTREAM_REPO="https://codeberg.org/zig/zig.git"
ZIG_UPSTREAM_COMMIT="738d2be9d" # matches .zigversion 0.16.0-dev.2650+738d2be9d
ZIG_VERSION="0.16.0-dev.2650+738d2be9d"
CEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Platform requirements for CEL
CEL_MIN_MACOS_MAJOR=26  # CEL is required on macOS 26+ (Tahoe)

# Build configuration
CEL_BUILD_TYPE="Release"
CEL_PREFER_STATIC_LLVM=true

# Export for child scripts
export ZIG_UPSTREAM_REPO ZIG_UPSTREAM_COMMIT ZIG_VERSION CEL_DIR
