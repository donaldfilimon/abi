# AbbeyBot reference

Canonical tree: `/Users/donaldfilimon/Desktop/AbbeyBot`.

## Products

| Product | Executable | Persistence | UI |
|---------|------------|-------------|-----|
| Desktop | `AbbeyBot` (`AbbeyBotApp`) | SwiftData | SwiftUI |
| Server | `AbbeyServer` | Fluent (Postgres / sqlite) | Leaf + JSON `/api/*` |

Shared library: `AbbeyCore` (personas, inference router, DiscordBM bridge, DQN / ingest scoring).

## Scripts

| Script | Purpose |
|--------|---------|
| `Scripts/run.sh` | Launch desktop |
| `Scripts/run-smoke.sh` | Debug + release build, AbbeyCoreTests, app launch |
| `Scripts/run-server.sh` | Launch server (sources `.env` if present) |
| `Scripts/run-server-smoke.sh` | Build server, boot, assert health/status/ingest/messages |
| `Scripts/verify-all.sh` | Both smoke gates |

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
