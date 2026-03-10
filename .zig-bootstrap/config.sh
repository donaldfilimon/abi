#!/usr/bin/env bash
# .zig-bootstrap toolchain configuration — single source of truth for upstream pin
#
# CEL (Custom Environment Linker) provides a patched Zig compiler for
# macOS 26+ (Tahoe) that fixes the upstream linker incompatibility.
#
# Consistency contract:
#   - ZIG_VERSION must match .zigversion and build.zig.zon minimum_zig_version
#   - ZIG_UPSTREAM_COMMIT must match the commit hash suffix in ZIG_VERSION
#   - Patches in patches/ must apply cleanly against ZIG_UPSTREAM_COMMIT
#
# Migration: run ./tools/scripts/bootstrap_migrate.sh for guided setup

ZIG_UPSTREAM_REPO="https://github.com/ziglang/zig.git"
ZIG_UPSTREAM_COMMIT="738d2be9d" # matches .zigversion 0.16.0-dev.1503+738d2be9d
ZIG_VERSION="0.16.0-dev.1503+738d2be9d"
ZLS_UPSTREAM_REPO="https://github.com/zigtools/zls.git"
ZLS_UPSTREAM_COMMIT=""  # Empty = use latest; set to a short hash for reproducible builds
BOOTSTRAP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Platform requirements for bootstrap Zig
BOOTSTRAP_MIN_MACOS_MAJOR=26  # CEL is required on macOS 26+ (Tahoe)

# Build configuration
BOOTSTRAP_BUILD_TYPE="Release"
BOOTSTRAP_PREFER_STATIC_LLVM=true

# Export for child scripts
export ZIG_UPSTREAM_REPO ZIG_UPSTREAM_COMMIT ZIG_VERSION ZLS_UPSTREAM_REPO ZLS_UPSTREAM_COMMIT BOOTSTRAP_DIR
