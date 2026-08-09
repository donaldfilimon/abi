# AbbeyBot reference (archived — tree not present)

> **This tree is missing from this Mac (verified 2026-08-09).**
> `/Users/donaldfilimon/Desktop/AbbeyBot` does not exist, and home `CLAUDE.md`
> records it is not in `Archive/`, `.Trash`, or on the external SSD. Every path
> below is therefore **historical**: it describes the architecture if the tree
> is restored, and none of the scripts can be run today. Ask the user where the
> tree went before acting on anything here — do not substitute another tree.
>
> The only surviving Swift/Abbey tree is
> `/Users/donaldfilimon/Downloads/Code/AbbeyCompanion`, whose wrappers are
> `Scripts/check.sh`, `run.sh`, `smoke.sh`, `lib.sh`.

Canonical tree (historical): `/Users/donaldfilimon/Desktop/AbbeyBot`.

## Products

| Product | Executable | Persistence | UI |
|---------|------------|-------------|-----|
| Desktop | `AbbeyBot` (`AbbeyBotApp`) | SwiftData | SwiftUI |
| Server | `AbbeyServer` | Fluent (Postgres / sqlite) | Leaf + JSON `/api/*` |

Shared library: `AbbeyCore` (personas, inference router, DiscordBM bridge, DQN / ingest scoring).

## Scripts

| Script | Purpose |
|--------|---------|
| `/Users/donaldfilimon/Desktop/AbbeyBot/Scripts/run.sh` | Launch desktop |
| `/Users/donaldfilimon/Desktop/AbbeyBot/Scripts/run-smoke.sh` | Debug + release build, AbbeyCoreTests, app launch |
| `/Users/donaldfilimon/Desktop/AbbeyBot/Scripts/run-server.sh` | Launch server (sources `.env` if present) |
| `/Users/donaldfilimon/Desktop/AbbeyBot/Scripts/run-server-smoke.sh` | Build server, boot, assert health/status/ingest/messages |
| `/Users/donaldfilimon/Desktop/AbbeyBot/Scripts/verify-all.sh` | Both smoke gates |

Build dirs: `${TMPDIR}/AbbeyBot.build`, `${TMPDIR}/AbbeyBot.server.build`.

## HTTP API (AbbeyServer)

- `GET /health` — liveness + Discord state
- `GET /api/metrics` — session counters
- `GET /api/status` — dialect + table counts + metrics
- `POST /api/ingest` — JSON ingest; when `ABBEY_API_TOKEN` is set, require `Authorization: Bearer …` or `X-Abbey-Token`
- List: `/api/users`, `/api/channels`, `/api/messages`, `/api/reputation`, `/api/interactions`, `/api/equity` (`?limit=`, default 100, max 500)
- `GET /` — Leaf status page

## Shared helpers (prefer editing these)

- `IngestScorer` — sentiment → DQN step
- `ReputationMath` — EMA / penalty / composite keys
- `ReplyCooldownTracker`, `ChannelSummaryBuilder`
- `PersonaName`, `MemoryFactText`
- `DiscordCopy` — slash-command user-facing strings
- `IntentClassifier.classify` — strict-by-default (emoji/empty → `.unknown`)

Adapters: `SocialBrain` / `AbbeyScheduler` (SwiftData) vs `FluentSocialBrain` / `FluentScheduler` (Fluent).

## Discord live path (desktop)

1. Developer Portal bot + **Message Content** intent
2. Discord sidebar → token (Keychain) → Connect
3. Optional dev guild → Register slash commands
4. Operating mode `liveDiscord`; optional auto-connect

Slash commands: `ask`, `rep`, `remember`, `forget`, `context`, `persona`.

## Deferred (do not claim shipped)

- React dashboard SPA
- Discord Voice
- SwiftData ↔ Postgres sync

## Companion (Downloads)

`/Users/donaldfilimon/Downloads/AbbeyCompanion 4` is superseded. Prefer AbbeyBot for new features. Companion smoke remains valid for the local-only app only.
