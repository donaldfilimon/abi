# AbbeyBot (Swift) reference: archived

The Swift AbbeyBot tree (desktop `AbbeyBot` app, Vapor `AbbeyServer`, `abbey`
CLI; SwiftUI + SwiftData desktop, Fluent server, shared `AbbeyCore`) is **not on
disk**. It lived at `~/dev/active/AbbeyBot`, moved to
`~/Archive/experimental-2026-09-18/AbbeyBot` on 2026-09-18, and went to the
Trash with that archive tree on 2026-09-28. The retired `AbbeyCompanion`
predecessor followed the same path from `~/dev/archive/`.

Recovery sources:

- GitHub: `donaldfilimon/AbbeyBot`, `donaldfilimon/AbbeyCompanion`.
- `~/at-risk-bundles/2026-09-28-archives/MANIFEST.tsv` (incremental bundles of
  local-only commits; they need the GitHub history to restore).

Restoring either tree is Donald's call. If one is restored, read its own
`AGENTS.md` and gate (`Scripts/verify-all.sh` for AbbeyBot) rather than trusting
any earlier description of its architecture.

Do not confuse it with the active Rust projects `~/dev/active/abbey-bot`
(Discord bot) and `~/dev/active/abbey` (CLI/TUI). They share no code with it.
