---
name: swift
description: >-
  This skill should be used when the user runs /swift, or asks about fixing Swift
  builds on this Mac: Xcode vs swiftly/TOOLCHAINS, which compiler a Swift tree
  under ~/dev/active pins (Gama, GamaStudio, Gama qt/, String, Mixed,
  accountforge, FlipperCompanion, SocialPilot, DeviceConnector, LiveContainer),
  SwiftData macro errors, repository-selected Swift 6.5-dev snapshots, external
  SwiftPM scratch paths, ~Copyable verification (-typecheck false passes),
  Swift Testing #expect on ~Copyable values, --filter matching nothing and
  exiting 0, or probing main-snapshot language features. Do not use for
  Rust/abi or general Swift language tutorials.
---

# Swift on this Mac

Procedural rules for Swift work on this machine. General Swift language
knowledge is assumed; this skill encodes the machine- and repo-specific
constraints that otherwise produce false SwiftData errors, codesign failures,
wrong-compiler builds, and false-green tests.

The repository's own `AGENTS.md`/`CLAUDE.md` and gate script are the
authority inside a tree. This skill routes you to the right compiler and gate;
it does not replace them.

## When not to use

- Rust / `~/dev/active/abi` work: use ABI project skills instead.
- Pure Swift language or concurrency Q&A with no toolchain or repository context.

## Hard toolchain rules

1. **Always `unset TOOLCHAINS`** (or prefix `env -u TOOLCHAINS`) before any
   Swift invocation, in every tree. A stray value silently swaps the compiler
   behind a bare `xcrun swift`.
2. **PATH `swift` is a swiftly shim.** `~/.swiftly/bin` is first on PATH and
   shims `swift`, `swiftc` and `strings`. Plain `swift` resolves through the
   nearest `.swift-version`, else swiftly's default, which is a main snapshot,
   not Xcode. Never trust a bare `swift` unless the repo's gate does exactly
   that on purpose.
3. **Default compiler is Xcode's**, invoked explicitly:

   ```bash
   /usr/bin/xcrun --toolchain default swift …
   # or: /Users/donaldfilimon/.grok/skills/swift/scripts/xcode-swift.sh …
   ```

   A snapshot compiler against the macOS 27 SDK produces nonsense SwiftData
   macro errors (`@Query`, `\.modelContext`) and codesign failures.
4. **Read the pin before choosing a command:** `.swift-version`,
   `Toolchains.toml`, `Package.swift` tools version, and the gate script. A
   repository-selected compiler, scratch path, or wrapper beats every generic
   command here.
5. **Keep build output outside the checkout** (`--scratch-path` or
   `--build-path` under `/private/tmp`), and give each session its own path
   when a peer may be running the same gate. For `swift run`, put the path
   flag **before** the product name.
6. **Change toolchains only through `swiftly`**, then verify `swiftly list`,
   `swift --version`, `swiftly run swift --version +main-snapshot-2026-08-21`,
   and `strings /bin/ls | head -1`. `~/Library/Developer/Toolchains` is
   swiftly-owned, not free space. Do not "fix" `SWIFT_PROJECT_BIN`.

## Which compiler and gate, per tree

Re-measure before trusting a row: `cat <tree>/.swift-version` and read the
gate's opening comment. Snapshot as of 2026-09-29.

| Tree (`~/dev/active/…`) | Compiler | Gate (authority) |
|------|------|------|
| `Gama` | **Snapshot pin** `main-snapshot-2026-08-21` (6.5-dev, id `org.swift.65202608211a`) via `.swift-version`; `swiftly run swift …` from the root. Manifest stays `swift-tools-version: 6.4` on purpose. | `./scripts/check-apple.sh` fast; `./scripts/check.sh` full matrix. `check-apple-platforms.sh` alone requires Xcode default to report 6.4. |
| `Gama/GamaStudio` | Same snapshot pin; depends on Gama by path | `cd GamaStudio && ./tools/check.sh` (not in Gama's gate or CI: run it after Apple-host or layout changes) |
| `Gama/qt` | **Exception inside the exception:** Xcode default 6.4 plus Homebrew Qt 6 | `cd qt && env -u TOOLCHAINS ./Scripts/check.sh`, verdict `check.sh: PASSED` |
| `String` | Snapshot pin `main-snapshot-2026-08-21`; Xcode 6.4 is a secondary route | `swiftly run swift build/test +main-snapshot-2026-08-21 --scratch-path /private/tmp/… -Xswiftc -warnings-as-errors` (both routes, build and test). `./script/build_and_run.sh --verify` builds and launches but **runs no tests**. |
| `Mixed` | Xcode project, Xcode default (`env -u TOOLCHAINS`) | `./script/build_and_run.sh all` (see `--help`) |
| `accountforge` | `.swift-version` 6.4.0 | `./tools/check.sh` |
| `FlipperCompanion` | `.swift-version` 6.4.0; Xcode project | `./check.sh` |
| `SocialPilot` | Xcode default via `xcrun --toolchain default` | `./check.sh` |
| `DeviceConnector` | `.swift-version` 6.4.0 | see `BUILD.md` |
| `LiveContainer` | Xcode project | `Scripts/check.sh`, verdict `check.sh: PASSED` |

None of these trees is under iCloud any more (Gama, String and Mixed cut over
on 2026-09-24). Their parked iCloud originals are recovery copies: never
develop there, and load `home-ops:icloud-git-safety` before any git operation
on an iCloud path.

## Probe modern language and SDK features

Do not infer availability from a proposal title, a main-branch interface, or
syntax highlighting. Compile the smallest representative source with the
repository-selected compiler and every supported route the change affects. A
type-check pass is not runtime, cross-SDK, ABI, or hosted-CI proof. Inspect the
installed public `.swiftinterface` when prose and the compiler disagree.

These spellings are available in the Gama snapshot, each for a narrow purpose:

| Spelling | Use and boundary |
| --- | --- |
| `Module::Declaration` | Selects a module explicitly when a local declaration could shadow its name; useful in macro-generated source. |
| `~Sendable` | Suppresses implicit `Sendable` inference to record intentional non-Sendability. It does not replace isolation design or an unavailable conformance a project uses for a named diagnostic. |
| `@diagnose(...)` | Changes one named diagnostic for a documented compatibility exception, with a removal condition. Never to hide portability, ownership, or concurrency failures. |
| `anyAppleOS` | Availability that truly applies to every Apple OS. Prefer `canImport(AppKit)`/`canImport(UIKit)` when framework capability is the real requirement. |
| `@c(name)` | C entry point. Not a drop-in for `@_cdecl`: `@_cdecl` emits C and Swift-convention symbols, `@c` only the C one, so migrating needs a versioned ABI and a consumer audit. |

Main-snapshot syntax is not a reason to adopt an API. Confirm it is
implemented rather than experimental, solves a current requirement, and passes
every supported toolchain and target gate.

## Move-only code: `-typecheck` gives FALSE PASSES

Measured 2026-08-28 on the 6.5-dev snapshot and Xcode 6.4 (27 probes,
identical). `swiftc -typecheck` exits 0 on definitively illegal `~Copyable`
code: move-only enforcement runs in SIL, after type checking.

```bash
# WRONG - reports success on illegal code
xcrun --toolchain <id> swiftc -typecheck -swift-version 6 probe.swift   # exit 0

# RIGHT - actually enforces ownership
xcrun --toolchain <id> swiftc -c -swift-version 6 -o /dev/null probe.swift
#   error: 'a' consumed more than once
```

A compile-fail fixture for a noncopyable contract must use `-c`, or it passes
and proves nothing.

Related facts from the same probes: `self` is immutable inside a `~Copyable`
deinit (consume into a local and mutate that); copying out of `.pointee` for a
noncopyable Pointee is illegal while in-place mutation, borrowing reads,
`move()` and `assumingMemoryBound` are legal; a struct storing a noncopyable
value needs an explicit `: ~Copyable`; a global `~Copyable` var can be mutated
but never consumed.

## Swift Testing traps

- `#expect` cannot read a bare stored property off a noncopyable value.
  `#expect(host.needsFrame)` expands to `__checkPropertyAccess`, which
  requires `Copyable`, and the diagnostic names that helper, not the property.
  Bind first: `let dirty = host.needsFrame; #expect(dirty)`. Comparisons
  (`#expect(host.ids == [...])`) use another overload and are fine.
- `--filter` matches the source identifier (struct or function name), not the
  `@Suite` display name and not the filename. A non-matching filter prints a
  warning and **exits 0**: confirm the reported test count.
- Do not add XCTest to a tree that is Swift Testing only (Gama, String).

## Failure signatures

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Nonsense errors on `@Query` / `\.modelContext` | Snapshot compiler (swiftly shim or `TOOLCHAINS`) | `unset TOOLCHAINS`; `/usr/bin/xcrun --toolchain default swift` |
| Gama/String check script fails on the version line | Built with Xcode 6.4 instead of the pin | `swiftly run swift …` from the repo root |
| "resource fork, Finder information, or similar detritus not allowed" | xattrs on in-tree build output | Scratch path under `/private/tmp` |
| `swiftc` aborts `couldNotFindTmpDir` | The `TMPDIR` passed does not exist | `mkdir -p` it first |
| Build path ignored by `swift run` | Flag order | Path flag before the product name |
| A gate collides with another session's run | Shared fixed scratch path (Gama: `/private/tmp/gama-framework-swiftpm`) | Use the repo's per-gate scratch variable with a unique root |

## Archived Swift trees

The Swift `AbbeyBot` (desktop + Vapor server + CLI) and `AbbeyCompanion` trees
are no longer on disk: they were moved to the Trash on 2026-09-28 with the
archive trees. Their history is on GitHub (`donaldfilimon/AbbeyBot`,
`donaldfilimon/AbbeyCompanion`) and in `~/at-risk-bundles/2026-09-28-archives/`
(`MANIFEST.tsv`). Restoring either is Donald's call. They share no code with
the active Rust `abbey-bot` or `abbey`. See `references/abbeybot.md`.

## Additional resources

Central root: `/Users/donaldfilimon/.grok/skills/swift/` (edit here; copies
under `~/.claude/skills/` and abi's `.agents/`/`.claude/` mirrors are sync
targets and get overwritten).

- `references/toolchain.md`: toolchain diagnosis and route verification.
- `references/abbeybot.md`: pointer to the archived Swift AbbeyBot.
- `scripts/xcode-swift.sh`: `unset TOOLCHAINS` + `xcrun --toolchain default swift` passthrough.
