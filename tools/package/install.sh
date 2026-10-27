#!/bin/sh
# Install, upgrade, roll back, remove and report an unsigned ABI macOS package.
#
#   install.sh install  PREFIX ARCHIVE   verify ARCHIVE.sha256 and every MANIFEST.json
#                                        hash, then make it PREFIX/current
#   install.sh rollback PREFIX           swap PREFIX/current and PREFIX/previous
#   install.sh remove   PREFIX --yes     delete versions/, current and previous only,
#                                        and only when every one of them is ABI's
#   install.sh status   PREFIX           print the current and previous versions
#
# Layout: PREFIX/versions/<id>/{bin/,MANIFEST.json}, with PREFIX/current and
# PREFIX/previous as relative symlinks (versions/<id>), so the prefix can move.
# A version is extracted into PREFIX/versions/.staging-<id>, verified there,
# and renamed into place. `previous` is then pointed at the old current
# version, and only after that is `current` pointed at the new one. Each link
# is replaced atomically (a new symlink renamed over the old one with `mv -h`),
# so an interruption at any point leaves `current` on a complete version.
# Rerunning the same install converges: when versions/<id> already exists and
# is not current, the archive is verified again and must carry a byte-identical
# MANIFEST.json, and the existing copy (itself re-verified) is made current.
# The same path makes any installed version current again, not only `previous`.
# Leftover .staging-* directories are removed at the start of the next install.
#
# MANIFEST.json is parsed line by line, without a JSON library: each "files"
# entry is exactly `    "relpath": "<64 hex sha256>"` on its own line, the
# layout tools/package/build_macos.sh writes. Anything else is refused.
#
# This installs an UNSIGNED, un-notarized local build. It is not a signed
# distribution, and nothing here touches ~/.abi or any user data.
set -eu

TARGET=aarch64-apple-darwin

usage() {
  printf 'usage: install.sh install PREFIX ARCHIVE | rollback PREFIX | remove PREFIX --yes | status PREFIX\n' >&2
  exit 2
}

die() {
  printf 'install.sh: %s\n' "$*" >&2
  exit 1
}

[ $# -ge 2 ] || usage
command=$1
prefix=$2
case $prefix in
  '' | / | //) die "refusing prefix '$prefix'" ;;
esac

staging=""
link_tmp=""
cleanup() {
  if [ -n "$staging" ] && [ -e "$staging" ]; then
    rm -rf "$staging"
  fi
  if [ -n "$link_tmp" ]; then
    rm -f "$link_tmp"
  fi
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

# Print a symlink's relative target, or nothing when it is absent.
link_target() {
  if [ -L "$1" ]; then
    readlink "$1"
  fi
}

# Atomically point PREFIX/<name> at <target>.
repoint() {
  link_tmp="$prefix/.$1.tmp.$$"
  rm -f "$link_tmp"
  ln -s "$2" "$link_tmp"
  mv -f -h "$link_tmp" "$prefix/$1"
  link_tmp=""
}

manifest_field() {
  sed -n "s/^  \"$2\": \"\\([^\"]*\\)\",\$/\\1/p" "$1/MANIFEST.json" | head -n 1
}

valid_id() {
  printf '%s\n' "$1" | grep -Eq '^abi-[A-Za-z0-9._+]+-[A-Za-z0-9]+-[A-Za-z0-9_]+-[A-Za-z0-9_]+-[A-Za-z0-9_]+$'
}

# Verify that DIR holds exactly the files its MANIFEST.json lists, each with
# the listed SHA-256, and that the manifest describes package <id>.
verify_manifest() {
  dir=$1
  id=$2
  [ -f "$dir/MANIFEST.json" ] || die "missing MANIFEST.json in $id"
  grep -qx '  "name": "abi",' "$dir/MANIFEST.json" || die "MANIFEST.json is not an abi manifest"
  version=$(manifest_field "$dir" version)
  revision=$(manifest_field "$dir" revision)
  target=$(manifest_field "$dir" target)
  [ "$target" = "$TARGET" ] || die "package target '$target' is not $TARGET"
  [ "abi-$version-$revision-$target" = "$id" ] || die "MANIFEST.json describes abi-$version-$revision-$target, not $id"

  entries=$(sed -n 's/^    "\([^"]*\)": "\([0-9a-f]\{64\}\)",\{0,1\}$/\2 \1/p' "$dir/MANIFEST.json")
  [ -n "$entries" ] || die "MANIFEST.json lists no files"
  listed=$(printf '%s\n' "$entries" | wc -l | tr -d ' ')
  unique=$(printf '%s\n' "$entries" | cut -d ' ' -f 2- | LC_ALL=C sort -u | wc -l | tr -d ' ')
  [ "$listed" = "$unique" ] || die "MANIFEST.json lists a file twice"

  while read -r digest relpath; do
    case $relpath in
      '' | /* | ../* | */../* | */.. | .. | MANIFEST.json) die "MANIFEST.json lists an unsafe path '$relpath'" ;;
    esac
    [ -f "$dir/$relpath" ] && [ ! -L "$dir/$relpath" ] || die "manifest file missing: $relpath"
    actual=$(shasum -a 256 "$dir/$relpath" | cut -d ' ' -f 1)
    [ "$actual" = "$digest" ] || die "manifest hash mismatch: $relpath"
  done <<EOF
$entries
EOF

  # Nothing but listed regular files (and directories) may ship.
  others=$(find "$dir" ! -type f ! -type d | head -n 1)
  [ -z "$others" ] || die "package contains a non-regular file: ${others#"$dir"/}"
  present=$(find "$dir" -type f ! -path "$dir/MANIFEST.json" | wc -l | tr -d ' ')
  [ "$present" = "$listed" ] || die "package holds $present files but MANIFEST.json lists $listed"
}

describe() {
  target=$(link_target "$prefix/$1")
  if [ -z "$target" ]; then
    printf '%s: none\n' "$1"
    return 0
  fi
  id=${target##*/}
  if [ ! -f "$prefix/$target/MANIFEST.json" ]; then
    printf '%s: %s (missing)\n' "$1" "$id"
    return 1
  fi
  printf '%s: %s (version %s)\n' "$1" "$id" "$(manifest_field "$prefix/$target" version)"
}

# Print archive entries (read from stdin) that would land outside <id>/.
unsafe_entries() {
  while IFS= read -r entry; do
    case $entry in
      "$1" | "$1/") ;;
      "$1"/*)
        case $entry in
          */../* | */..) printf '%s\n' "$entry" ;;
        esac
        ;;
      *) printf '%s\n' "$entry" ;;
    esac
  done
}

do_install() {
  [ $# -eq 3 ] || usage
  archive=$3
  [ -f "$archive" ] || die "archive not found: $archive"
  [ -f "$archive.sha256" ] || die "checksum file not found: $archive.sha256"
  expected=$(cut -d ' ' -f 1 <"$archive.sha256" | head -n 1)
  printf '%s\n' "$expected" | grep -Eq '^[0-9a-f]{64}$' || die "malformed checksum file: $archive.sha256"
  actual=$(shasum -a 256 "$archive" | cut -d ' ' -f 1)
  [ "$actual" = "$expected" ] || die "checksum mismatch: $archive"

  id=$(basename "$archive" .tar.gz)
  valid_id "$id" || die "archive name is not abi-<version>-<revision>-$TARGET.tar.gz: $archive"
  tar -tzf "$archive" >/dev/null || die "unreadable archive: $archive"
  bad_entry=$(tar -tzf "$archive" | unsafe_entries "$id" | head -n 1)
  [ -z "$bad_entry" ] || die "archive entry outside $id/: $bad_entry"

  mkdir -p "$prefix/versions"
  for leftover in "$prefix"/versions/.staging-*; do
    [ -e "$leftover" ] || continue
    rm -rf "$leftover"
  done
  old=$(link_target "$prefix/current")
  [ "$old" != "versions/$id" ] || die "$id is already installed and current (use rollback or remove)"

  staging="$prefix/versions/.staging-$id"
  mkdir "$staging"
  tar -xzf "$archive" -C "$staging" --strip-components 1 --no-same-owner
  verify_manifest "$staging" "$id"
  if [ -e "$prefix/versions/$id" ] || [ -L "$prefix/versions/$id" ]; then
    # Installed before (an interrupted install, or an older version): reuse it
    # only when it is the same package and still verifies.
    [ -d "$prefix/versions/$id" ] && [ ! -L "$prefix/versions/$id" ] || die "versions/$id is not a directory"
    cmp -s "$staging/MANIFEST.json" "$prefix/versions/$id/MANIFEST.json" || die "versions/$id holds a different package than $archive"
    verify_manifest "$prefix/versions/$id" "$id"
    rm -rf "$staging"
    staging=""
    action=activated
  else
    mv "$staging" "$prefix/versions/$id"
    staging=""
    action=installed
  fi

  # previous first, then current: an interruption between the two leaves the
  # old version current, and rerunning this install completes the switch.
  if [ -n "$old" ]; then
    repoint previous "$old"
  fi
  repoint current "versions/$id"
  printf '%s %s\n' "$action" "$id"
}

do_rollback() {
  [ $# -eq 2 ] || usage
  previous=$(link_target "$prefix/previous")
  [ -n "$previous" ] || die "no previous version to roll back to"
  [ -d "$prefix/$previous" ] || die "previous version is missing: $previous"
  verify_manifest "$prefix/$previous" "${previous##*/}"
  current=$(link_target "$prefix/current")
  repoint current "$previous"
  if [ -n "$current" ]; then
    repoint previous "$current"
  fi
  printf 'rolled back to %s\n' "${previous##*/}"
}

# True when versions/<entry> is one this installer created: a leftover
# .staging-<id>, or a directory <id> whose MANIFEST.json names package abi.
owned_entry() {
  path="$prefix/versions/$1"
  [ -d "$path" ] && [ ! -L "$path" ] || return 1
  case $1 in
    .staging-*) valid_id "${1#.staging-}" ;;
    *) valid_id "$1" && [ -f "$path/MANIFEST.json" ] && grep -qx '  "name": "abi",' "$path/MANIFEST.json" ;;
  esac
}

do_remove() {
  [ $# -eq 3 ] && [ "$3" = "--yes" ] || die "remove deletes $prefix/versions, current and previous; rerun with --yes"
  # Check everything before deleting anything: refuse a prefix ABI does not own.
  if [ -e "$prefix/versions" ] || [ -L "$prefix/versions" ]; then
    [ -d "$prefix/versions" ] && [ ! -L "$prefix/versions" ] || die "refusing: $prefix/versions is not an ABI versions directory"
    for entry in "$prefix"/versions/* "$prefix"/versions/.[!.]* "$prefix"/versions/..?*; do
      [ -e "$entry" ] || [ -L "$entry" ] || continue
      owned_entry "${entry##*/}" || die "refusing: $entry was not installed by install.sh"
    done
  fi
  for link in current previous; do
    if [ -L "$prefix/$link" ]; then
      target=$(readlink "$prefix/$link")
      case $target in
        versions/*) valid_id "${target#versions/}" || die "refusing: $prefix/$link points outside versions/: $target" ;;
        *) die "refusing: $prefix/$link points outside versions/: $target" ;;
      esac
    elif [ -e "$prefix/$link" ]; then
      die "refusing: $prefix/$link is not an ABI symlink"
    fi
  done

  rm -f "$prefix/current" "$prefix/previous"
  if [ -d "$prefix/versions" ]; then
    for entry in "$prefix"/versions/* "$prefix"/versions/.[!.]* "$prefix"/versions/..?*; do
      [ -e "$entry" ] || continue
      rm -rf "$entry"
    done
    rmdir "$prefix/versions"
  fi
  printf 'removed abi from %s\n' "$prefix"
}

do_status() {
  [ $# -eq 2 ] || usage
  status=0
  describe current || status=1
  describe previous || status=1
  return "$status"
}

case $command in
  install) do_install "$@" ;;
  rollback) do_rollback "$@" ;;
  remove) do_remove "$@" ;;
  status) do_status "$@" ;;
  *) usage ;;
esac
