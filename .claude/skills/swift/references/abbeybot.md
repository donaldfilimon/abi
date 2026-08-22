# AbbeyBot reference

Canonical Swift tree:
`/Users/donaldfilimon/dev/active/AbbeyBot` (verified 2026-08-22).

Do not confuse it with either of these separate projects:

- `/Users/donaldfilimon/dev/active/abbey-bot` — active Rust Discord bot.
- `/Users/donaldfilimon/dev/archive/AbbeyCompanion` — retired Swift companion
  predecessor.

Read the live tree's `AGENTS.md` before changing architecture or repeating a
capability claim. This reference is an orientation aid; repository source,
tests, manifests, and the gate remain authoritative.

## Products

| Product | Executable / target | Persistence | UI |
| --- | --- | --- | --- |
| Desktop | `AbbeyBot` / `AbbeyBotApp` | SwiftData | SwiftUI |
| Server | `AbbeyServer` | Fluent with Postgres or SQLite fallback | Leaf, JSON API, dashboard SPA |
| CLI | `abbey` / `AbbeyCLI` through the `AbbeyBot` executable | Local SQLite by default | Terminal REPL |

Shared libraries: `AbbeyCore` for personas, inference, Discord integration, and
learning primitives; `AbbeyServerKit` for the Fluent-backed runtime shared by
server and CLI.

## Scripts

| Script | Purpose |
| --- | --- |
| `Scripts/run.sh` | Launch desktop through the Xcode toolchain wrapper. |
| `Scripts/run-smoke.sh` | Desktop build, tests, and launch smoke. |
| `Scripts/run-server.sh` | Launch the headless server. |
| `Scripts/run-server-smoke.sh` | Boot the server and verify HTTP/database behavior. |
| `Scripts/run-cli-smoke.sh` | Verify the terminal client path. |
| `Scripts/run-web-smoke.sh` | Verify the dashboard web package. |
| `Scripts/check-static-security.sh` | Diff, conflict, secret, and static security hygiene. |
| `Scripts/check-server-snapshot.sh` | CI-aligned server graph under a Swift snapshot toolchain. |
| `Scripts/check-server-linux.sh` | Cross-compile the server graph with the configured Static Linux SDK. |
| `Scripts/verify-all.sh` | Gate of record; read it for the current exact sequence. |

Build directories live under `${TMPDIR}/AbbeyBot*.build`. Always
`unset TOOLCHAINS` and prefer repository wrappers because the desktop target
requires the Xcode toolchain; a swiftly development snapshot breaks SwiftData
macro/toolchain compatibility.

## Architecture boundaries

- Prefer shared behavior in `AbbeyCore` or `AbbeyServerKit` rather than forking
  desktop `AbbeyEngine`, server `BotRuntime`, and CLI behavior.
- Desktop uses SwiftData; server/CLI use Fluent. They are persistence adapters,
  not independent persona systems.
- Discord, Twitch, sync, learning, and voice claims change over time. Use the
  current `AGENTS.md`, `README.md`, source, tests, and `tasks/` ledgers; do not
  revive the 2026-08-09 "tree missing" or "React dashboard deferred" snapshot.
- Green local tests are not live Discord evidence. Credentials, intents,
  participant consent, and observed external behavior are separate acceptance
  layers.

## Git and provenance

The repository has `origin` at
`https://github.com/donaldfilimon/AbbeyBot.git`. Inspect the current branch,
working tree, ancestry, and requested authority before pushing or opening a PR.

Historical notes that mention `/Users/donaldfilimon/Desktop/AbbeyBot` may remain
in dated ledgers as provenance. Current instructions and agent profiles must use
`/Users/donaldfilimon/dev/active/AbbeyBot`.
