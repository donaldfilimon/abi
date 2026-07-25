---
name: swift
description: >-
  This skill should be used when the user runs /swift, or asks about fixing Swift
  builds on this Mac, Xcode vs swiftly/TOOLCHAINS, SwiftData macro errors,
  AbbeyBot/AbbeyServer/AbbeyCompanion, DiscordBM, verify-all / AbbeyBot smoke, or
  Swift 6.4 / macOS 27 SPM for those packages. Do not use for Zig/abi or general
  Swift language tutorials.
version: 0.1.0
---

# Swift (toolchain + AbbeyBot)

Procedural rules for Swift work on this Mac and for the unified AbbeyBot package.
General Swift language knowledge is assumed; this skill encodes **machine- and repo-specific** constraints that otherwise cause false SwiftData errors and codesign failures.

## When not to use

- Zig / `~/abi` work — use ABI project skills instead.
- Pure Swift language or concurrency Q&A with no Mac toolchain or AbbeyBot context.

## Hard toolchain rules

1. Always `unset TOOLCHAINS` (or `unset TOOLCHAINS || true`) before any Swift invocation.
2. Always invoke Swift via Xcode:

```bash
/usr/bin/xcrun --toolchain default swift …
```

3. Never trust PATH `swift` when it resolves to **swiftly** / `DEVELOPMENT-SNAPSHOT` — that mix breaks SwiftData macros (`@Query`, `\.modelContext`) against the macOS 27 SDK.
4. Prefer project wrappers when present: `Scripts/run.sh`, `Scripts/run-smoke.sh`, `Scripts/run-server.sh`, `Scripts/run-server-smoke.sh`, `Scripts/verify-all.sh` (AbbeyBot only for the last three).
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

| Path | Role |
|------|------|
| `/Users/donaldfilimon/Desktop/AbbeyBot` | **Active** — dual product (`AbbeyBot` desktop + `AbbeyServer`) |
| `/Users/donaldfilimon/Downloads/AbbeyCompanion 4` | Superseded local-only companion; smoke via `run-smoke.sh` / `run.sh` only; point new work at AbbeyBot |

Read `AGENTS.md` in the active tree before changing architecture. Keep `CLAUDE.md` a thin redirect to `AGENTS.md`.

## AbbeyBot quick map

| Layer | Path | Notes |
|-------|------|-------|
| Core | `Sources/AbbeyCore/` | No SwiftData; DiscordBM, personas, `IngestScorer`, `ReputationMath`, `DiscordCopy` |
| Desktop | `Sources/AbbeyBotApp/` | SwiftUI + SwiftData; `AbbeyEngine` |
| Server | `Sources/AbbeyServer/` | Vapor + Fluent; `BotRuntime`; Leaf + `/api/*` |
| Tests | `Tests/AbbeyCoreTests/` | AbbeyCore unit tests |
| Verify | `Scripts/verify-all.sh` | Desktop + server smoke |

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
cd /Users/donaldfilimon/Desktop/AbbeyBot
unset TOOLCHAINS
bash Scripts/verify-all.sh
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
