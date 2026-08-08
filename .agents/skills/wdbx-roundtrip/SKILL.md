---
name: wdbx-roundtrip
description: Build the abi CLI and drive a full WDBX persistence round-trip on a scratch segment — db init → block insert → query → db verify — proving the on-disk checkpoint + WAL chain stays valid. Use to smoke-test WDBX persistence/durability after touching the store, checkpoint, or WAL code, or to demo the block lifecycle.
---

# wdbx-roundtrip — drive abi's WDBX persistence lifecycle

Driver: **`.agents/skills/wdbx-roundtrip/roundtrip.sh`** (paths relative to repo root).
Builds the CLI and runs the four-step store lifecycle against a `mktemp` scratch
v2 base path (created and removed by the driver). Evidence is the `RESULT:`
line. Fully local, no network.

## Run (agent path)
```bash
.agents/skills/wdbx-roundtrip/roundtrip.sh                          # profile=abi, default metadata
.agents/skills/wdbx-roundtrip/roundtrip.sh aviva '{"note":"hi"}'    # custom profile / metadata JSON
```
Steps and asserted markers:
1. `wdbx db init <store>` → `initialized empty WDBX v2`
2. `wdbx block insert <store> <profile> <json>` → `appended block:`, `blocks=1`
3. `wdbx query <store>` → `"blocks":1`
4. `wdbx db verify <store>` → `v2 verify OK:`, `merged_chain_valid=true`

Prints `RESULT: PASS` (exit 0) or a FAIL count.

Current Rust driver: proves v2 initialization, a UUID-backed hashed audit block,
reopen-visible stats, and successful merged audit-DAG verification.

## Gotchas
- The scratch store and activated generation live under one `mktemp` directory
  and are removed on exit, so the default `.abi/` store is never opened.
- `wdbx query <store>` on a freshly-block-inserted segment reports `kv_entries:0
  vectors:0 blocks:1` — blocks are the append-only content-addressed log; kv and
  vectors are separate surfaces. `mode:cpu_fallback` remains honest even when
  the detected backend label is `metal`.
- For semantic/vector queries (embeddings, personas) use `abi wdbx query <store>
  "<text>" <persona>`; the round-trip driver checks the structural path.
- For a source-level tour of the HNSW index, MVCC snapshot chain, and WAL, use
  the `wdbx-explorer` subagent.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | Check nightly via `./tools/cargo.sh --version`, then `./tools/check.sh`. |
| `merged_chain_valid=true` missing | V2 audit-DAG or recovery regression — inspect `crates/abi-wdbx/src/v2/` and `versioned.rs`. |
