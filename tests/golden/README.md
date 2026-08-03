# Golden fixtures — captured from the Zig implementation

These files were produced by the **Zig** `abi` and `abi-mcp` binaries while
`./build.sh check` was still green, at commit `919dad8` on branch
`rust-rewrite`, with Zig `0.17.0-dev.1442+972627084` (matching `.zigversion`).

They exist because of a sequencing constraint. `build.zig` wires the entire
module graph through `src/root.zig`, and the test steps (`test-contracts`,
`test-feature-contracts`, `check-parity`) walk all of it. So the first Zig
directory that gets deleted breaks `zig build check`, and it stays broken until
the last one goes. From that moment there is no running Zig implementation to
compare against.

Capturing the frozen surfaces *first* turns "the Rust CLI has 13 commands with
the right names" — which source-reading can establish — into "the Rust CLI emits
byte-identical help output", which it cannot.

Regenerate only if the Zig tree is still intact:

```bash
TERM=dumb NO_COLOR=1 ./zig-out/bin/abi help --json > tests/golden/help.json
```

## CLI

| File | Produced by |
|---|---|
| `help.txt` | `abi help` |
| `help.json` | `abi help --json` — the whole frozen command surface, 18 KB |
| `completion.{bash,zsh,fish}` | `abi help --completion <shell>` |
| `help-<command>.txt` | `abi help <command>`, one per top-level command |

The 13 top-level commands: `help`, `complete`, `train`, `agent`, `backends`,
`plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, `scheduler`, `nn`. Plus
the `--tui` → `tui` shortcut, which `help.json` records under `shortcuts`.

Captured with `TERM=dumb NO_COLOR=1`; `help.txt` still contains ANSI SGR escapes,
so the Zig renderer does not consult either variable. The Rust port must emit the
same bytes — including the escapes — or explicitly decide to change that and
re-record.

## MCP

| File | Produced by |
|---|---|
| `mcp-initialize.json` | `initialize` response — protocol version, capabilities, `serverInfo` |
| `mcp-tools-list.json` | `tools/list` response — all 12 tools with full `inputSchema` |

The frozen 12, in the order the server returns them (order is part of the
fixture): `ai_run`, `ai_complete`, `ai_learn`, `ai_train`, `wdbx_query`,
`scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`,
`plugin_list`, `wdbx_stats`, `plugin_run`.

Note this is *not* the declaration order in `src/mcp/handlers.zig`, and
`wdbx_stats` comes after `plugin_list` rather than next to `wdbx_query`. Ported
code must reproduce the emitted order, not the tidier one.

## WDBX on-disk format

`wdbx-format.md` specifies it; `wdbx-sample.seg.jsonl` and `wdbx-sample.manifest`
are **synthetic** fixtures in that format.

Synthetic on purpose. The live store at `~/.abi/` holds ~300 segments and ~180 MB
of the user's actual completions and embeddings; committing a slice of that would
publish their data into a public repository. The fixtures cover every record type
and edge case instead, and `wdbx-format.md` records the census taken from the real
store so the sample can be checked as representative.

## MCP tool call responses

`mcp-tool-calls.jsonl` — all 12 tools invoked with **empty arguments**, one line
per tool as `{"tool":…,"response":…}`. Nine return a validation error, and those
error strings are themselves part of the contract: `Missing input`,
`Missing profile`, `Missing query`, `Missing connector service`,
`Missing plugin name`.

Each tool was invoked in its own server process. Batching them into one stdin
stream does not work — the server stops after the first tool that errors, which
silently truncated the first capture attempt to two lines.

`mcp-tool-calls-args.jsonl` — the success paths, with arguments. Note
`connector_test` returns the OpenAI-compatible *local synthesized* response, so
it cross-checks `abi-connectors`' `payload::openai_local_response`.

## WDBX runtime output

- `wdbx-db-verify.txt` — `abi wdbx db verify ~/.abi/wdbx` against the real store.
- `wdbx-stats.txt` — `abi wdbx` subcommand listing (the bare `stats` form prints
  usage; the real numbers come from `db verify` and the `wdbx_stats` MCP tool).
- `backends.txt` — `abi backends`, including its ANSI colour codes.

### Cross-implementation agreement

`wdbx-db-verify.txt` is the strongest verification the port has so far. Zig and
the Rust reader were pointed at the same store and agree on every field:

```
Zig : epoch=306 kv=262 vectors=663 blocks=330 spatial=0 temporal_nodes=3 temporal_edges=2 chain_valid=true
Rust:           kv=262 vectors=663 blocks=330 spatial=0 temporal_nodes=3 temporal_edges=2 chain verifies
```

The original cross-implementation capture is historical and is not directly
reproducible after the Zig tree was removed. Do not point tests or automation at
`~/.abi/wdbx`. New compatibility captures must first copy an operator-approved
store into a disposable scratch directory and run only against that copy.

**Historical caveat on these numbers.** Capturing the `ai_complete` / `ai_learn` / `ai_train`
success paths above *wrote to the live store*; those tools reported
`persisted=true`. The store went from epoch 300 to 306, gaining 6 kv entries, 9
vectors, 3 blocks, and the first 3 temporal nodes / 2 temporal edges it had ever
held. Appends only, and the chain still verifies, but it means the counts here
are later than the ones in the `abi-wdbx` commit message, and that a golden
capture of an AI tool is not a read-only operation.

Silver lining: those temporal records are the only real ones in existence, so
they are what first exercised the Rust temporal-record path against actual data
rather than a fixture.
