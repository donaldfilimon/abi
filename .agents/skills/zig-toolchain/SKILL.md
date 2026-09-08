---
name: zig-toolchain
description: Use before running any zig command on this Mac, and when asked which Zig is active, what version a repo targets, where the stdlib source lives, how to read Zig docs, what `zig std` serves, why two zig binaries disagree, or where to put build and test scratch. Covers zvm layout, the dangling `current` symlink, the Homebrew nightly cask beside zvm, and the iCloud scratch rule.
---

# Zig toolchain on this Mac

Zig here is **master/nightly**, installed through zvm. Measured 2026-09-06.

## What is actually installed

```
zig version        -> 0.17.0-dev.2018+ab30a0b9a
zig env .zig_exe   -> /Users/donaldfilimon/.zvm/master/zig
zig env .std_dir   -> /Users/donaldfilimon/.zvm/master/lib/std
zig env .target    -> aarch64-macos.27.0...27.0-none
global cache       -> ~/.cache/zig
```

`~/.zvm/bin` is a **symlink to `~/.zvm/master`**, not a directory of shims. Every
`zig` call resolves through it, so deleting `~/.zvm/master` breaks `zig` outright.

`zvm list` reports exactly one installed version: `master`. zvm itself is at
`~/.zvm/self/zvm`.

## Two traps in the layout

**`~/.zvm/current` is a dangling symlink.** It points at
`~/.zvm/0.17.0-dev.1442+972627084`, a directory that no longer exists, so
`~/.zvm/current/zig` fails with "no such file or directory". It is stale zvm
bookkeeping from 2026-07-24, not the active toolchain, and reading it to answer
"which Zig is active" gives the wrong answer or an error. `~/.zvm/bin` is the
live path.

**There is a second zig on PATH.** `/opt/homebrew/bin/zig` is a Homebrew
`zig@nightly` cask. On 2026-09-06 both binaries were the same build
(`0.17.0-dev.2018+ab30a0b9a`), so the duplication is currently invisible, but the
two update on independent schedules and will drift. zvm's copy wins because
`~/.zvm/bin` sits earlier on PATH. When a build behaves differently than a
version string suggests, resolve the binary first:

```bash
whence -pa zig        # zsh; `command -v -a` is a bash-ism zsh rejects
```

Pin the binary explicitly in scripts and in anything you report as evidence:
`/Users/donaldfilimon/.zvm/bin/zig`.

## Reading the standard library

The stdlib source at `~/.zvm/master/lib/std/` is the authority for what this
toolchain does. Read it with grep and sed. It is 100 top-level entries.

`zig std` starts a local docs server, but it is a **browser** tool: it serves an
autodoc WASM app (`main.js`, `main.wasm`) plus `sources.tar`, and the tar
contains only `std/` source, the same files already on disk. For an agent it adds
a server and a WASM decode step to reach material it can already read. Read
`lib/std` directly. Use `zig std` only when a human wants the rendered docs:

```bash
zig std --port 9899 --no-open-browser    # `/` alone 404s; the app is at main.js
```

## The language reference ships with the toolchain

`~/.zvm/master/doc/langref.html` is the full language reference, 960 KB, and its
version stamp matches `zig version` exactly (`0.17.0-dev.2018+ab30a0b9a`). It is
the authority for the builtin list (124 documented, 128 recognized by the
compiler) and for **deprecation**, which the compiler will not tell you about.

That split matters. Compiling proves **removal**; the langref proves
**deprecation**. `@intFromEnum` compiles and passes today while the langref says
"Deprecated. Use @backingInt or @bitCast instead", so a green test is not
evidence a builtin is current. Check both.

Master diverges from published tutorials and from model pretraining fast. Treat
any Zig idiom recalled rather than read as unverified until it compiles, and
check the langref before calling it current.

## Scratch paths

Build and test scratch goes under `/private/tmp`, never in an iCloud-managed
path (`~/Documents`, `~/Desktop`, `~/Library/Mobile Documents`) and never at the
home root. iCloud paths stall git and can break codesigning of test artifacts.

`/private/tmp` is wiped on reboot, so nothing unique may live there. Verification
snippets are disposable by design; real work belongs in a repository.

## Verifying a claim about Zig

```bash
mkdir -p /private/tmp/zig-check && cd /private/tmp/zig-check
cat > s.zig <<'EOF'
const std = @import("std");
test "claim" { try std.testing.expect(true); }
EOF
/Users/donaldfilimon/.zvm/bin/zig test s.zig
```

`zig ast-check <file>` catches syntax and simple semantic errors without a full
build, which is the cheapest first pass on a snippet.

## Command surface

`build` `fetch` `init` `build-exe` `build-lib` `build-obj` `test` `test-obj`
`run` `ast-check` `fmt` `reduce` `translate-c` `ar` `cc` `c++` `dlltool` `lib`
`objcopy` `objdump` `ranlib` `rc` `env` `help` `std` `libc` `targets`.

`zig cc` and `zig c++` are drop-in cross compilers for C and C++, usable outside
any Zig project. `zig targets` lists every supported compilation target.

## Related

Language and library detail lives in the sibling skills: [[zig-types]],
[[zig-errors]], [[zig-memory]], [[zig-comptime]], [[zig-builtins]], [[zig-std]],
[[zig-io]], [[zig-build]], [[zig-testing]], [[zig-c-interop]].
