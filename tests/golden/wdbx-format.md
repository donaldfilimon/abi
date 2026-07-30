# WDBX v1 on-disk format

Reverse-engineered from the live store at `~/.abi/` on 2026-07-30, cross-checked
against `src/features/wdbx/`. **The Rust store must read this**: that store holds
~300 segments and ~180 MB of the user's real completions and embeddings, so a
format-incompatible rewrite silently orphans all of it.

## Layout

```
~/.abi/
  wdbx                 # binary index (603 KB in the observed store)
  wdbx.manifest        # which segments are live
  wdbx.seg.<epoch>.jsonl
```

`ABI_WDBX_PATH` overrides the directory; `ABI_WDBX_PERSIST` gates whether writes
are durable at all.

## Manifest

Line-oriented plain text:

```
# ABI-WDBX-SEGMENTS v1
next_epoch=301
active=0,1,2,3,...,299
```

- `# ABI-WDBX-SEGMENTS v1` — magic line, must match exactly.
- `next_epoch` — the epoch number the next new segment takes. Monotonic.
- `active` — comma-separated epochs currently live. A segment file whose epoch is
  absent here is garbage awaiting collection and **must not** be read. The
  observed value listed all 300 of 300, so compaction had not yet dropped any;
  a reader that assumes `active` is dense would still be wrong.

## Segment files

First line is the magic `# ABI-WDBX v1`. Every subsequent line is one JSON
object — JSONL, so a torn final line is a truncated write and should be dropped
rather than failing the whole segment.

A segment may contain *only* the magic line (several observed), so "empty" is a
valid state, not corruption.

Three record types, discriminated by `type`. Census over the live store:

| `type` | Records | Shape |
|---|---:|---|
| `vector` | 100,296 | `{"type":"vector","id":u64,"values":[f32; 32]}` |
| `block` | 50,148 | see below |
| `kv` | 40,796 | `{"type":"kv","key":string,"value":string}` |

### `vector`

```json
{"type":"vector","id":1,"values":[0.1,0.2, ...]}
```

Every observed vector had **32** dimensions (sampled 634 across six segments), but
dimensionality is a property of the data rather than the format — do not hardcode
32 in the reader.

### `kv`

```json
{"type":"kv","key":"completion:1","value":"{\"kind\":\"completion\", ...}"}
```

`value` is an opaque string. In practice it frequently holds *JSON encoded as a
string*, so it is double-encoded — the reader must treat it as a string and leave
interpretation to the caller. Parsing it eagerly would fail on values that are not
JSON.

Keys repeat across segments (`completion:1` appears in both `seg.0` and `seg.1`).
Later epochs shadow earlier ones — this is the MVCC layering, so replay order
matters and a naive concatenation produces the wrong result.

### `block`

The audit chain. Each block links to its predecessor by hash:

```json
{"type":"block",
 "hash":[u8; 32],
 "prev_hash":"<hex or sentinel>",
 "timestamp_ms":i64,
 "sequence":u64,
 "profile":"abi",
 "query_id":u64,
 "response_id":u64,
 "metadata":"<opaque string>"}
```

Note the asymmetry, which is easy to get wrong: `hash` is an **array of 32 byte
integers**, while `prev_hash` is a **string**. A reader that models both the same
way fails on real data.

## What a compatible reader must do

1. Read `wdbx.manifest`, honour `active` — ignore segment files not listed.
2. Replay active segments in **ascending epoch order**, so later writes shadow
   earlier ones.
3. Verify each segment's magic line; treat a bad one as corruption, not as data.
4. Tolerate a segment with only a magic line.
5. Tolerate a truncated final line (interrupted write).
6. Keep `kv` values as opaque strings.
7. Preserve `hash`-as-array vs `prev_hash`-as-string.

Items 4, 5 and 7 are the ones a from-scratch implementation gets wrong, and each
would surface as a failure to open the user's existing store.
