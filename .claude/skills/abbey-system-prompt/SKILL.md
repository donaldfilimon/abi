---
name: abbey-system-prompt
description: >-
  Maintain the Abbey / Aviva / Abi Discord system prompts in the Rust abbey-bot
  (`src/ask.rs`) and keep the hosted Grok bots (Grok Bot Abbey, Aviva Grok)
  in step with them. Use when the user says "improve grok bots", "abbey prompt",
  "persona prompt", "ask.rs", "system prompt", "tip floor", asks to change how
  Abbey/Aviva/Abi speak on Discord, or after an abbey-bot PR merges and the
  hosted bot's tip floor needs bumping. Runs as /abbey-system-prompt.
---

# Abbey system prompt

Two different bots share one voice, and they must not share one text.

| Bot | Where the copy lives | Carries a tip floor? |
|---|---|---|
| Discord Abbey (Rust `abbey-bot`, live launchd service) | `~/dev/active/abbey-bot/src/ask.rs` | **No, by design** |
| Hosted Grok Bot Abbey / Aviva Grok (grok.com) | the bot's instructions on grok.com; no local copy, no connector | **Yes** |

The hosted bots run inside a Grok goal loop that maintains the repository, so
their instructions legitimately carry operator vocabulary: a tip floor (the
last merged PR and commit), `Current / Partial / Proposed / Blocked` claim
labels, "empty goal means standby", cross-lane handoff rules. The Discord bot
feeds the model no ledger and no repo state, so that vocabulary is noise to a
loopback model, and a pinned tip floor rots the moment the next PR merges
(`#162` sat in the Discord prompt while `main` was at `#170`). A test in
`ask.rs` now bans it from every persona prompt. Never copy hosted instructions
into `ask.rs` verbatim, and never copy `ask.rs` back as the hosted text.

## Discord side: `src/ask.rs`

Four functions own the copy; everything else is shaping.

- `contract_description(persona)`: the persona's operating description.
- `contract_character(persona)`: the first-person character line.
- `system_prompt(persona)`: `"You are {persona}. "` + description +
  character + fixed Discord framing. Used by `/persona ask`, `/roleplay`,
  `/summarize`, the welcome message, and media captions (`engine.rs`,
  `commands_brain/media.rs`), so a change here reaches all of them.
- `degraded_reply(persona)`: the no-backend honesty copy, pinned verbatim.

Routing stays frozen in `persona.rs` (golden contracts, wyhash). Prompt copy
is the only thing this skill edits.

### What the tests pin (change copy and assert together)

- `You are {persona}. ` prefix on every prompt.
- Aviva's first sentence, verbatim: `Focused response mode optimized for
  speed, clarity, candor, and technical precision.` and the em dash in
  `honest\u{2014}not`.
- Abbey's character line starts `I\u{2019}ll`; **no ASCII `'` anywhere** in
  Abbey's description or character (`\u{2019}` for apostrophes, `\u{201c}` /
  `\u{201d}` for quotes). Phrases pinned: `lead with the answer`, `when
  I\u{2019}m not sure`, `local and consent-aware`, `can\u{2019}t verify`,
  `hand NSFW roleplay to Aviva through /roleplay`, `deep runtime claims to
  Abi`, `local-first Discord companion`, `not OpenAI Realtime`,
  `ABBEY_VOICE_MODE=local`, `unverified as unverified`, `music mirroring is
  not listen consent`.
- `WDBX is substrate` in Abi; `hand to Abbey` in Aviva; `/roleplay` named in
  all three.
- Banned in every prompt: `Tip floor`, `a0ad563`, `Partial / Proposed /
  Blocked`, `Current` as a claim label, `empty goal`, `Cross-lane`, and the
  private product name.
- `degraded_reply` is compared as a whole string; edit the test in the same
  commit.

### Facts the copy must stay true to

- Text comes from a loopback OpenAI-compatible server (`ABBEY_BOT_LLM_ENDPOINT`,
  Ollama or mlx-lm on 127.0.0.1). Voice, when `ABBEY_VOICE_MODE=local`, is
  on-device mlx-audio, SFW, not OpenAI Realtime. Listening needs explicit
  consent; music mirroring is not listen consent.
- Adult roleplay is `/roleplay` only: bot DMs, or NSFW guild channels where an
  operator ran `/admin nsfw on` (`roleplay_gate.rs` is the admission matrix).
  The copy points at the command, not at a persona, because SFW channels
  refuse regardless of persona.
- `/roleplay` forces `Persona::Aviva` and receives `system_prompt(Aviva)`, so
  Aviva's description must admit that lane without becoming erotic HQ for
  ordinary `/persona ask` routing.
- Never name features the bot does not have (no Realtime, no distributed
  runtime, no AGI claims).

### Procedure

1. The shared checkout is edited by other agents (a Grok goal loop has been
   caught writing it). Never `git checkout` there. Work in a worktree beside
   the repo: `git -C ~/dev/active/abbey-bot worktree add
   ../abbey-bot-wt-<topic>-<date> -b content/<topic>-<date> origin/main`.
2. Edit copy and the pinning tests together. Keep `persona.rs` untouched.
3. `cargo fmt --all`, then `cargo test --locked ask` (single crate: no `-p`,
   no `--workspace`), then the real gate: `./check.sh > log 2>&1; echo
   "EXIT: $?"`, exit read from the log, `--locked` untouched. Drop
   `~/.swiftly/bin` from `PATH` first.
4. Add a dated line to `tasks/goals.md` (every merge here does).
5. Commit, push the branch, open the PR. Hosted `Gate` will complete in
   seconds with 0 steps: that is the account billing lock, UNMEASURABLE, not
   a failure; say so in the PR body and do not chase it.
6. Merge, then remove the worktree and delete the branch
   (`git worktree remove`, `git branch -d`, `git worktree prune`). Do not
   fast-forward the shared checkout's `main` yourself, and do not redeploy the
   launchd service: merge is not deploy, and both are Donald's calls.

## Hosted side: grok.com bots

No connector reaches grok.com from this machine. The update path is: Donald
pastes the bot's current instructions, you return revised text, Donald pastes
it back. Offer the browser pane only as the alternative.

After each abbey-bot merge, the one routine edit is the **tip floor**: bump it
to the merged PR number and `main` SHA (`git -C ~/dev/active/abbey-bot
log -1 --format=%h origin/main` after a fetch). Everything else in the hosted
text changes only when the voice or the honesty rules change, and then the
Discord copy changes in the same pass so the two do not drift in substance.

Keep the two texts different in kind: the hosted bot may say "tip floor
#N / <sha>+" and speak in claim labels; the Discord bot may not.

## Related

- `~/claude-account-skills/skills/discord-abbey/` is the claude.ai account
  skill covering both Discord bots' architecture; it is uploaded separately
  through claude.ai Settings → Capabilities → Skills and is not synced by
  `/sync-clis`. Link to it, do not duplicate it.
- `dev/active/abbey-bot/AGENTS.md` is the operational authority for the live
  service, the worktree rule, and the gate.
