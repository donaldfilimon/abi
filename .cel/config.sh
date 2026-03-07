#!/usr/bin/env bash
# .cel toolchain configuration — single source of truth for upstream pin
ZIG_UPSTREAM_REPO="https://github.com/ziglang/zig.git"
ZIG_UPSTREAM_COMMIT="74f361a5c" # matches .zigversion 0.16.0-dev.2650+74f361a5c
ZIG_VERSION="0.16.0-dev.2650+74f361a5c"
CEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
