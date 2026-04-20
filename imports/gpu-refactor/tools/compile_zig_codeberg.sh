#!/bin/bash
set -euo pipefail

# Auto compile Zig from the Codeberg mirror master branch
# Linked against LLVM, Clang, and LLD

ZIG_REPO="https://codeberg.org/ziglang/zig.git"
CLONE_DIR="${HOME}/.cache/zig-codeberg"

echo "Checking Homebrew LLVM installation..."
LLVM_PREFIX=$(brew --prefix llvm)
if [ ! -d "$LLVM_PREFIX" ]; then
  echo "Error: LLVM not found. Please run 'brew install llvm'."
  exit 1
fi

echo "Setting up Zig clone directory at $CLONE_DIR..."
mkdir -p "$CLONE_DIR"
cd "$CLONE_DIR"

if [ ! -d "zig" ]; then
  echo "Cloning Zig from Codeberg..."
  git clone "$ZIG_REPO" zig
fi

cd zig
echo "Fetching latest master..."
git checkout master
git pull origin master

echo "Configuring CMake build..."
mkdir -p build && cd build

cmake .. \
  -DCMAKE_C_COMPILER="$LLVM_PREFIX/bin/clang" \
  -DCMAKE_CXX_COMPILER="$LLVM_PREFIX/bin/clang++" \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_PREFIX_PATH="$LLVM_PREFIX" \
  -DZIG_STATIC_LLVM=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -GNinja

echo "Compiling Zig with Ninja..."
ninja -j$(sysctl -n hw.ncpu)

echo "Compilation complete! The zig binary is located at:"
echo "$CLONE_DIR/zig/build/zig"

echo "Creating symlink in ~/.local/bin/zig..."
mkdir -p ~/.local/bin
ln -sf "$CLONE_DIR/zig/build/zig" ~/.local/bin/zig
echo "Done! Make sure ~/.local/bin is in your PATH."
