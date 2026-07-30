# Swift toolchain reference (this Mac)

## Preferred invocation

```bash
unset TOOLCHAINS || true
/usr/bin/xcrun --toolchain default swift --version
/usr/bin/xcrun --toolchain default swift build --build-path "${TMPDIR}/MyPkg.build"
```

Confirm the selected toolchain is the **Xcode** default, not a snapshot:

```bash
xcrun --find swift
# expect something under /Applications/Xcode*.app/... or /Library/Developer/CommandLineTools
which -a swift
# PATH may still show ~/.swiftly/bin/swift — do not use it for AbbeyBot/SwiftData
```

## Failure signatures

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Nonsense errors on `@Query` / `\.modelContext` | swiftly / DEVELOPMENT-SNAPSHOT on PATH or `TOOLCHAINS` set | `unset TOOLCHAINS`; use `/usr/bin/xcrun --toolchain default` |
| Codesign / ad-hoc sign failures under `~/Desktop` or Downloads | Finder/iCloud xattrs on `.build` | `--build-path "${TMPDIR}/…"` |
| `swift run Product --build-path …` ignores build path | Flag order | `--build-path` **before** product name |
| Zig/network confusion in home CLAUDE.md | Wrong project | AbbeyBot is Swift; ABI Zig lives in `~/abi` |

## SPM hygiene

- Prefer existing `Package.resolved` pins unless intentionally bumping deps.
- DiscordBM `from: "1.16.0"` (Voice still deferred upstream).
- Vapor/Fluent stack is server-only; keep SwiftData out of `AbbeyCore`.

## macOS 27 / Swift 6.4

- Package platforms: `.macOS(.v27)`.
- Every target: `.swiftLanguageMode(.v6)`.
- Treat strict concurrency (actors, `Sendable`) as required, not optional.

## Desktop vs Linux server builds

- Desktop app: Xcode toolchain on macOS only.
- AbbeyServer Docker image: `swift:6.0-bookworm` (Linux). Do not assume macOS-only APIs inside `AbbeyServer` / `AbbeyCore` paths used by the server.
