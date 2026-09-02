---
name: swift
description: >-
  This skill should be used when the user runs /swift, or asks about fixing Swift
  builds on this Mac, Xcode vs swiftly/TOOLCHAINS, SwiftData macro errors,
  AbbeyBot/AbbeyServer, the archived AbbeyCompanion, DiscordBM, or
  Swift 6.4 / macOS 27 SPM for those packages. Do not use for Rust/abi or general
  Swift language tutorials.
version: 0.1.0
---

# Swift (toolchain + AbbeyBot)

Procedural rules for Swift work on this Mac and for the unified AbbeyBot package.
General Swift language knowledge is assumed; this skill encodes **machine- and repo-specific** constraints that otherwise cause false SwiftData errors and codesign failures.

## When not to use

- Rust / `~/dev/active/abi` work — use ABI project skills instead.
- Pure Swift language or concurrency Q&A with no Mac toolchain or AbbeyBot context.

## Hard toolchain rules

1. Always `unset TOOLCHAINS` (or `unset TOOLCHAINS || true`) before any Swift invocation.
   This one is universal — it holds in every tree, including the exceptions below.
2. **Default** (AbbeyBot, AbbeyCompanion, CoreAIAssistant, Invasion3D, Mixed):
   invoke Swift via Xcode.

```bash
/usr/bin/xcrun --toolchain default swift …
```

   **⚠️ TWO TREES ARE EXCEPTIONS AND PIN THEIR OWN SNAPSHOT. Using Xcode's
   default there is the WRONG COMPILER**, and the check scripts will reject it:

| Tree | Pin | Everyday invocation |
|------|-----|---------------------|
| `~/Desktop/Gama` | `.swift-version` = `main-snapshot-2026-08-21` (Swift 6.5-dev, id `org.swift.65202608211a`) | `swiftly run swift <build\|run\|test>` from the repo root |
| `~/Desktop/String` | same snapshot via `.swift-version`; Xcode 6.4 is a **secondary** route, not the primary | `swiftly run swift build +main-snapshot-2026-08-21 --scratch-path <outside-iCloud> -Xswiftc -warnings-as-errors` |

   Gama's own `CLAUDE.md` states it is the machine-wide exception to the
   "Xcode default toolchain" rule; its `check-*.sh` scripts verify
   `Swift version 6.5` and fail loudly on mismatch. String runs
   warnings-as-errors on **both** build and test, on both routes.
   Both trees are iCloud/FileProvider-managed: `swift test` needs a
   `--scratch-path` outside the checkout or codesigning fails, and `git status`
   can stall for 60+ seconds.

3. Never trust PATH `swift` when it resolves to **swiftly** / `DEVELOPMENT-SNAPSHOT` — that mix breaks SwiftData macros (`@Query`, `\.modelContext`) against the macOS 27 SDK.
4. Prefer a tree's own `Scripts/` wrappers when they exist — but `ls` that
   directory first rather than trusting this file; the script set differs per
   tree.
5. Keep SwiftPM build output outside the checkout. This avoids stale build
   products and extended-attribute codesign failures regardless of where the
   source tree lives:

| Tree | Build path |
|------|------------|
| AbbeyBot | `--build-path "${TMPDIR}/AbbeyBot.build"` (server: `AbbeyBot.server.build`) |
| AbbeyCompanion | `--build-path "${TMPDIR}/AbbeyCompanion.build"` |

For `swift run`, put `--build-path` **before** the product name.

Helper (absolute path):

```bash
/Users/donaldfilimon/.grok/skills/swift/scripts/xcode-swift.sh --version
```

## Move-only code: `-typecheck` gives FALSE PASSES

Measured 2026-08-28 on both the 6.5-dev snapshot and Xcode 6.4, 27 probes,
identical results. **`swiftc -typecheck` returns EXIT 0 on definitively illegal
`~Copyable` code.** Move-only enforcement runs in SIL, *after* type checking, so
a plain use-after-consume typechecks clean:

```bash
# WRONG - reports success on illegal code
xcrun --toolchain <id> swiftc -typecheck -swift-version 6 probe.swift   # exit 0

# RIGHT - actually enforces ownership
xcrun --toolchain <id> swiftc -c -swift-version 6 -o /dev/null probe.swift
#   error: 'a' consumed more than once
```

Anyone verifying a noncopyable migration, or writing a compile-fail fixture for
one, must use `-c`. A `-typecheck` fixture that "must fail to compile" will pass
and prove nothing.

Related ownership facts established by the same probes: `self` is immutable
inside a `~Copyable` deinit (no in-place mutation, no `inout` of a stored
property — but consuming into a local and mutating the local is fine); copying
out of `.pointee` for a noncopyable Pointee is illegal while in-place mutation,
borrowing reads, `move()`, and `assumingMemoryBound` are all legal; a struct
does **not** become noncopyable by inference and needs an explicit `: ~Copyable`
when it stores one; and a global `~Copyable` var can be mutated but never
consumed.

## Orient: which tree?

> **Current as of 2026-08-22:** the canonical Swift AbbeyBot tree is
> `/Users/donaldfilimon/dev/active/AbbeyBot`. It is separate from the Rust
> Discord bot at `/Users/donaldfilimon/dev/active/abbey-bot`.

| Path | Role | Present? |
|------|------|----------|
| `/Users/donaldfilimon/dev/active/AbbeyBot` | Canonical Swift dual product (`AbbeyBot` desktop + `AbbeyServer` + `abbey` CLI); its own git repo and remote. | **active** |
| `/Users/donaldfilimon/dev/archive/AbbeyCompanion` | Retired companion predecessor; wrappers are `Scripts/check.sh`, `run.sh`, `smoke.sh`, `lib.sh`. | archived |
| `/Users/donaldfilimon/Downloads/AbbeyCompanion 4` | Former superseded companion copy | **gone** |

Read `AGENTS.md` in the active tree before changing architecture. Keep `CLAUDE.md` a thin redirect to `AGENTS.md`.

## AbbeyBot quick map

| Layer | Path | Notes |
|-------|------|-------|
| Core | `Sources/AbbeyCore/` | No SwiftData; DiscordBM, personas, `IngestScorer`, `ReputationMath`, `DiscordCopy` |
| Desktop | `Sources/AbbeyBotApp/` | SwiftUI + SwiftData; `AbbeyEngine` |
| Server | `Sources/AbbeyServer/` | Vapor + Fluent; `BotRuntime`; Leaf + `/api/*` |
| Tests | `Tests/AbbeyCoreTests/` | AbbeyCore unit tests |
| Verify | `Scripts/verify-all.sh` | Static/security, package graphs, desktop, server, CLI, web, and related smoke gates; read the script for the current exact sequence |

Platforms: `.macOS(.v27)`. Language mode: `.v6` / tools-version `6.4`.

### Mirrored engines

Desktop `AbbeyEngine` and server `BotRuntime` are **persistence adapters** around shared AbbeyCore. When changing ingest, slash-command copy, persona resolve, or scoring:

1. Prefer editing AbbeyCore helpers first.
2. Check **both** engines' `ingestMessage` / `makeInteractionRouter` / `makeMessageIngress` for drift.

### Server env (non-secrets)

See `.env.example`: `DISCORD_BOT_TOKEN`, `DISCORD_DEV_GUILD_ID`, `DATABASE_URL`, optional `ABBEY_API_TOKEN` for `POST /api/ingest`.

`DATABASE_URL`: `postgres://…`, `sqlite://memory` / `sqlite::memory:`, or `sqlite:///path`.

### Verify gates

These are two different gates covering two different projects. Do not treat one
as evidence for the other.

**AbbeyBot gate — active and runnable locally:**

```bash
cd /Users/donaldfilimon/dev/active/AbbeyBot
unset TOOLCHAINS
bash Scripts/verify-all.sh
```

**AbbeyCompanion gate — a DIFFERENT project, present and runnable.** A green result
here says nothing about AbbeyBot:

```bash
cd /Users/donaldfilimon/dev/archive/AbbeyCompanion
unset TOOLCHAINS
bash Scripts/check.sh
```

Claim-honest: green local smoke does not prove live Discord (needs Message
Content intent, credentials, and manual observation). Follow `AGENTS.md` and
the repository ledgers for the current voice and dashboard acceptance boundary.

## Git / process (AbbeyBot)

- Prefer `cursor/` branches from `main`; FF-merge when finishing; never force-push `main`.
- Conventional Commits.
- `origin` is `https://github.com/donaldfilimon/AbbeyBot.git`; inspect branch,
  ancestry, and user authorization before pushing or opening a PR.

## Additional resources

### Reference files

Load as needed from the central skill (or synced copy):

- **`references/toolchain.md`** — toolchain diagnosis, codesign/xattr, common failures
- **`references/abbeybot.md`** — API surface, architecture, smoke expectations, deferred scope

Central root: `/Users/donaldfilimon/.grok/skills/swift/`

### Scripts

- **`/Users/donaldfilimon/.grok/skills/swift/scripts/xcode-swift.sh`** — `unset TOOLCHAINS` + `xcrun --toolchain default swift` passthrough
