---
name: swift
description: >-
  This skill should be used when the user runs /swift, or asks about fixing Swift
  builds on this Mac, Xcode vs swiftly/TOOLCHAINS, SwiftData macro errors,
  AbbeyCompanion (or a restored AbbeyBot/AbbeyServer), DiscordBM, or
  Swift 6.4 / macOS 27 SPM for those packages. Do not use for Rust/abi or general
  Swift language tutorials.
version: 0.1.0
---

# Swift (toolchain + AbbeyBot)

Procedural rules for Swift work on this Mac and for the unified AbbeyBot package.
General Swift language knowledge is assumed; this skill encodes **machine- and repo-specific** constraints that otherwise cause false SwiftData errors and codesign failures.

## When not to use

- Rust / `~/abi` work — use ABI project skills instead.
- Pure Swift language or concurrency Q&A with no Mac toolchain or AbbeyBot context.

## Hard toolchain rules

1. Always `unset TOOLCHAINS` (or `unset TOOLCHAINS || true`) before any Swift invocation.
2. Always invoke Swift via Xcode:

```bash
/usr/bin/xcrun --toolchain default swift …
```

3. Never trust PATH `swift` when it resolves to **swiftly** / `DEVELOPMENT-SNAPSHOT` — that mix breaks SwiftData macros (`@Query`, `\.modelContext`) against the macOS 27 SDK.
4. Prefer a tree's own `Scripts/` wrappers when they exist — but `ls` that
   directory first rather than trusting this file; the script set differs per
   tree, and the AbbeyBot wrappers this skill used to name are unreachable
   (see "Orient: which tree?").
5. Build **off** Desktop/Downloads when the tree lives under iCloud-synced folders:

| Tree | Build path |
|------|------------|
| AbbeyBot | `--build-path "${TMPDIR}/AbbeyBot.build"` (server: `AbbeyBot.server.build`) |
| AbbeyCompanion | `--build-path "${TMPDIR}/AbbeyCompanion.build"` |

For `swift run`, put `--build-path` **before** the product name.

Helper (absolute path):

```bash
/Users/donaldfilimon/.grok/skills/swift/scripts/xcode-swift.sh --version
```

## Orient: which tree?

> **The AbbeyBot tree is not on this Mac (verified 2026-08-09).** `~/Desktop` is
> empty, and home `CLAUDE.md` records it is not in `Archive/`, `.Trash`, or on
> the external SSD either. **Ask the user where it went before acting on any
> AbbeyBot instruction below** — do not silently substitute another tree.

| Path | Role | Present? |
|------|------|----------|
| `/Users/donaldfilimon/Downloads/Code/AbbeyCompanion` | Only surviving Swift/Abbey tree; its own git repo. Wrappers are `Scripts/check.sh`, `run.sh`, `smoke.sh`, `lib.sh` — there is **no** `run-smoke.sh` or `verify-all.sh` here. | yes |
| `/Users/donaldfilimon/Desktop/AbbeyBot` | Former dual product (`AbbeyBot` desktop + `AbbeyServer`) | **gone** |
| `/Users/donaldfilimon/Downloads/AbbeyCompanion 4` | Former superseded companion copy | **gone** |

Read `AGENTS.md` in the active tree before changing architecture. Keep `CLAUDE.md` a thin redirect to `AGENTS.md`.

## AbbeyBot quick map

| Layer | Path | Notes |
|-------|------|-------|
| Core | `Sources/AbbeyCore/` | No SwiftData; DiscordBM, personas, `IngestScorer`, `ReputationMath`, `DiscordCopy` |
| Desktop | `Sources/AbbeyBotApp/` | SwiftUI + SwiftData; `AbbeyEngine` |
| Server | `Sources/AbbeyServer/` | Vapor + Fluent; `BotRuntime`; Leaf + `/api/*` |
| Tests | `Tests/AbbeyCoreTests/` | AbbeyCore unit tests |
| Verify | `Scripts/verify-all.sh` (in the AbbeyBot tree — currently unreachable) | Desktop + server smoke |

Platforms: `.macOS(.v27)`. Language mode: `.v6` / tools-version `6.4`.

### Mirrored engines

Desktop `AbbeyEngine` and server `BotRuntime` are **persistence adapters** around shared AbbeyCore. When changing ingest, slash-command copy, persona resolve, or scoring:

1. Prefer editing AbbeyCore helpers first.
2. Check **both** engines' `ingestMessage` / `makeInteractionRouter` / `makeMessageIngress` for drift.

### Server env (non-secrets)

See `.env.example`: `DISCORD_BOT_TOKEN`, `DISCORD_DEV_GUILD_ID`, `DATABASE_URL`, optional `ABBEY_API_TOKEN` for `POST /api/ingest`.

`DATABASE_URL`: `postgres://…`, `sqlite://memory` / `sqlite::memory:`, or `sqlite:///path`.

### Verify gates

```bash
# Unreachable: /Users/donaldfilimon/Desktop/AbbeyBot no longer exists. Retained
# only to describe the gate's shape if that tree is restored. The AbbeyCompanion
# equivalent (verified present) is:
cd /Users/donaldfilimon/Downloads/Code/AbbeyCompanion
unset TOOLCHAINS
bash Scripts/check.sh
```

Claim-honest: green smoke ≠ live Discord (needs Message Content intent + token). Voice and React SPA remain deferred.

## Git / process (AbbeyBot)

- Prefer `cursor/` branches from `main`; FF-merge when finishing; never force-push `main`.
- Conventional Commits.
- No remote unless the user asks to create/push one.

## Additional resources

### Reference files

Load as needed from the central skill (or synced copy):

- **`references/toolchain.md`** — toolchain diagnosis, codesign/xattr, common failures
- **`references/abbeybot.md`** — API surface, architecture, smoke expectations, deferred scope

Central root: `/Users/donaldfilimon/.grok/skills/swift/`

### Scripts

- **`/Users/donaldfilimon/.grok/skills/swift/scripts/xcode-swift.sh`** — `unset TOOLCHAINS` + `xcrun --toolchain default swift` passthrough
