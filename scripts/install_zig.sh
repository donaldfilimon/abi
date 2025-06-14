#!/usr/bin/env bash
set -euo pipefail
VERSION=0.14.1
URL="https://ziglang.org/download/${VERSION}/zig-$(uname -m)-linux-${VERSION}.tar.xz"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
curl -L "$URL" -o "$TMP_DIR/zig.tar.xz"
mkdir -p "$TMP_DIR/extract"
tar -xf "$TMP_DIR/zig.tar.xz" -C "$TMP_DIR/extract"
sudo rm -rf /usr/local/zig
sudo mkdir -p /usr/local/zig
sudo cp -r "$TMP_DIR/extract"/*/ /usr/local/zig/
sudo ln -sf /usr/local/zig/zig /usr/local/bin/zig
