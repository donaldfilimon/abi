# Swift toolchain reference (this Mac)

## Preferred invocation

Read repository instructions and pin files before choosing a compiler. A
repository-selected toolchain, external scratch path, or validation wrapper
overrides the generic Xcode commands below.

```bash
unset TOOLCHAINS || true
/usr/bin/xcrun --toolchain default swift --version
/usr/bin/xcrun --toolchain default swift build --scratch-path /private/tmp/MyPkg.build
```

Snapshot-pinned trees (Gama, GamaStudio, String) select
`main-snapshot-2026-08-21` through `.swift-version`:

```bash
cd ~/dev/active/Gama
unset TOOLCHAINS
swiftly run swift --version        # must report 6.5-dev
swiftly run swift test --scratch-path "/private/tmp/gama-test-$USER"
```

## Verify which compiler you actually have

```bash
whence -pa swift                                 # ~/.swiftly/bin/swift first: a shim
swiftly list                                     # installed toolchains and swiftly's default
/usr/bin/xcrun --toolchain default swift --version   # Xcode's compiler (6.4 today)
xcode-select -p                                  # which Xcode (may be Xcode-beta)
```

swiftly's default is a main snapshot, so a bare `swift` outside a tree with a
`.swift-version` is **not** Xcode. `xcrun --toolchain <id>` is unaffected by
`TOOLCHAINS`; a bare `xcrun swift` is not.

Before adopting a main-only language feature or a new SDK API, compile a
minimal probe with the selected compiler and each supported secondary route.
Inspect the installed public `.swiftinterface` when prose and compiler
behavior disagree. A successful type-check establishes only that
compiler/SDK combination: not ABI, runtime behavior, another platform,
packaging, accessibility, or hosted CI.

## Failure signatures

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Nonsense errors on `@Query` / `\.modelContext` | Snapshot compiler via swiftly shim or `TOOLCHAINS` | `unset TOOLCHAINS`; `/usr/bin/xcrun --toolchain default` |
| Codesign / "detritus not allowed" failures | xattrs on in-tree build output | Scratch path under `/private/tmp` |
| `swift run Product --build-path …` ignores the path | Flag order | Path flag **before** the product name |
| Version-line failure in a Gama/String script | Wrong compiler for the pin | `swiftly run swift …` from the repo root |
| `couldNotFindTmpDir` | `TMPDIR` points at a missing directory | `mkdir -p "$TMPDIR"` |

## SwiftPM hygiene

- Preserve `Package.resolved` pins unless intentionally bumping dependencies.
- Swift 6 language mode is the norm across these trees; treat strict
  concurrency (actors, `Sendable`) as required.
- Keep `swift-tools-version` where the repo pins it (Gama deliberately stays at
  6.4 so Xcode's integrated SwiftPM can resolve platform gates).
