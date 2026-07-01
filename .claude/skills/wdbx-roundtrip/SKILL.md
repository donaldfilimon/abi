---
name: wdbx-roundtrip
description: Build the abi CLI and drive a full WDBX persistence round-trip on a scratch segment — db init → block insert → query → db verify — proving the on-disk checkpoint + WAL chain stays valid. Use to smoke-test WDBX persistence/durability after touching the store, checkpoint, or WAL code, or to demo the block lifecycle.
---

# wdbx-roundtrip — drive abi's WDBX persistence lifecycle

Driver: **`.claude/skills/wdbx-roundtrip/roundtrip.sh`** (paths relative to repo root).
Builds the CLI and runs the four-step store lifecycle against a scratch segment
under `zig-out/` (created and removed by the driver). Evidence is the `RESULT:`
line. Fully local, no network.

## Run (agent path)
```bash
.claude/skills/wdbx-roundtrip/roundtrip.sh                          # profile=abi, default metadata
.claude/skills/wdbx-roundtrip/roundtrip.sh aviva '{"note":"hi"}'    # custom profile / metadata JSON
```
Steps and asserted markers:
1. `wdbx db init <store>` → `initialized empty WDBX`
2. `wdbx block insert <store> <profile> <json>` → `appended block:`, `blocks=1`
3. `wdbx query <store>` → `"blocks":1`
4. `wdbx db verify <store>` → `checkpoint OK:`, `chain_valid=true`

Prints `RESULT: PASS` (exit 0) or a FAIL count.

Verified this session: **PASS** on Zig master `0.17.0-dev.1099` — insert appends a
hashed block, query reports `blocks:1`, verify confirms `chain_valid=true` and
`WAL OK`.

## Gotchas
- The scratch store is `zig-out/skill-wdbx-roundtrip.jsonl` — deleted before and
  after the run, so it never touches your default `.abi/` store.
- `wdbx query <store>` on a freshly-block-inserted segment reports `kv_entries:0
  vectors:0 blocks:1` — blocks are the append-only content-addressed log; kv and
  vectors are separate surfaces. `backend:metal mode:native_gpu` is normal.
- For semantic/vector queries (embeddings, personas) use `abi wdbx query <store>
  "<text>" <persona>`; the round-trip driver checks the structural path.
- For a source-level tour of the HNSW index, MVCC snapshot chain, and WAL, use
  the `wdbx-explorer` subagent.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | Check `zig version` (see `/zig-pin`), then `./build.sh check`. |
| `chain_valid=true` missing | Checkpoint/WAL regression — inspect `src/features/wdbx/` (checkpoint + WAL merge path). |
