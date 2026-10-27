#!/bin/sh
# Build an UNSIGNED ABI package for Apple-silicon macOS.
#
#   tools/package/build_macos.sh OUT_DIR
#
# Builds abi, abi-mcp and abi-control-plane in release mode through
# ./tools/cargo.sh (locked), then writes into OUT_DIR:
#
#   abi-<version>-<rev>-aarch64-apple-darwin/{bin/*,MANIFEST.json}
#   abi-<version>-<rev>-aarch64-apple-darwin.tar.gz
#   abi-<version>-<rev>-aarch64-apple-darwin.tar.gz.sha256
#
# The release binaries load libabi_fm_shim.dylib and libabi_metal_dot.dylib
# through `@loader_path/` install names (set by the abi-connectors and abi-gpu
# build scripts), so the dylibs ship in bin/ beside the executables. Rewriting
# install names would invalidate the linker's ad-hoc signature.
#
# The tree must have no tracked changes (untracked or ignored scratch is
# allowed), so the package revision describes what was built. Files are copied
# to a temporary name and renamed, never rewritten in place. Signing,
# notarization and Linux/Windows packages are out of scope.
set -eu

TARGET=aarch64-apple-darwin

die() {
  printf 'build_macos.sh: %s\n' "$*" >&2
  exit 1
}

if [ $# -ne 1 ] || [ -z "$1" ]; then
  printf 'usage: build_macos.sh OUT_DIR\n' >&2
  exit 2
fi
out_arg=$1

root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"

[ "$(uname -s)/$(uname -m)" = "Darwin/arm64" ] || die "this package targets $TARGET; run it on Apple-silicon macOS"

tracked_changes() {
  git status --porcelain --untracked-files=no
}
[ -z "$(tracked_changes)" ] || die "the tree has tracked changes; commit or stash them first"

version=$(sed -n '/^\[workspace\.package\]/,/^\[/s/^version = "\([^"]*\)"$/\1/p' Cargo.toml | head -n 1)
[ -n "$version" ] || die "could not read [workspace.package] version from Cargo.toml"
revision=$(git rev-parse --short HEAD)
rustc_version=$("$(rustup which rustc)" --version)
id="abi-$version-$revision-$TARGET"

mkdir -p "$out_arg"
out=$(cd "$out_arg" && pwd)
stage="$out/$id"
for existing in "$stage" "$out/$id.tar.gz" "$out/$id.tar.gz.sha256"; do
  [ ! -e "$existing" ] || die "$existing already exists; choose another OUT_DIR"
done

./tools/cargo.sh build --locked --release -p abi-cli -p abi-mcp -p abi-control-plane

[ -z "$(tracked_changes)" ] || die "the build changed tracked files"
[ "$(git rev-parse --short HEAD)" = "$revision" ] || die "HEAD moved during the build"

case ${CARGO_TARGET_DIR:-} in
  '') release="$root/target/release" ;;
  /*) release="$CARGO_TARGET_DIR/release" ;;
  *) release="$root/$CARGO_TARGET_DIR/release" ;;
esac

copy() {
  cp "$1" "$2.tmp"
  mv -f "$2.tmp" "$2"
}

mkdir -p "$stage/bin"
dylibs=""
for binary in abi abi-mcp abi-control-plane; do
  [ -x "$release/$binary" ] || die "missing release binary: $release/$binary"
  copy "$release/$binary" "$stage/bin/$binary"
  # Every non-system dependency must be an @loader_path dylib we ship.
  foreign=$(otool -L "$release/$binary" | sed -n '2,$p' | sed 's/^[[:space:]]*//; s/ (.*$//' |
    grep -Ev '^(/usr/lib/|/System/Library/|@loader_path/)' || true)
  [ -z "$foreign" ] || die "$binary links an unshippable library: $foreign"
  dylibs="$dylibs $(otool -L "$release/$binary" | sed -n 's/^[[:space:]]*@loader_path\/\([^ ]*\) (.*$/\1/p')"
done
for dylib in $(printf '%s\n' $dylibs | LC_ALL=C sort -u); do
  [ -f "$release/$dylib" ] || die "missing dylib: $release/$dylib"
  copy "$release/$dylib" "$stage/bin/$dylib"
done

files=$(cd "$stage" && find . -type f ! -name MANIFEST.json | sed 's|^\./||' | LC_ALL=C sort)
count=$(printf '%s\n' "$files" | wc -l | tr -d ' ')
{
  printf '{\n'
  printf '  "name": "abi",\n'
  printf '  "version": "%s",\n' "$version"
  printf '  "revision": "%s",\n' "$revision"
  printf '  "target": "%s",\n' "$TARGET"
  printf '  "rustc": "%s",\n' "$rustc_version"
  printf '  "files": {\n'
  index=0
  for relpath in $files; do
    index=$((index + 1))
    digest=$(shasum -a 256 "$stage/$relpath" | cut -d ' ' -f 1)
    comma=","
    [ "$index" -lt "$count" ] || comma=""
    printf '    "%s": "%s"%s\n' "$relpath" "$digest" "$comma"
  done
  printf '  }\n'
  printf '}\n'
} >"$stage/MANIFEST.json.tmp"
mv -f "$stage/MANIFEST.json.tmp" "$stage/MANIFEST.json"

COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -czf "$out/$id.tar.gz.tmp" -C "$out" "$id"
mv -f "$out/$id.tar.gz.tmp" "$out/$id.tar.gz"
(cd "$out" && shasum -a 256 "$id.tar.gz" >"$id.tar.gz.sha256.tmp" && mv -f "$id.tar.gz.sha256.tmp" "$id.tar.gz.sha256")

printf 'built %s/%s.tar.gz (unsigned)\n' "$out" "$id"
