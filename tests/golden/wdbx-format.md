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

**Six** record types, discriminated by `type`. The live store contains only
three of them, but the serializer emits all six and the parser accepts all six,
so a reader must handle every one or it will reject a segment written by a build
that used the temporal graph or the 3-D spatial index.

Census over the live store:

| `type` | Records | Shape |
|---|---:|---|
| `vector` | 100,296 | `{"type":"vector","id":u64,"values":[f32; 32]}` |
| `block` | 50,148 | see below |
| `kv` | 40,796 | `{"type":"kv","key":string,"value":string}` |
| `spatial` | 0 | `{"type":"spatial","id":u64,"x":f32,"y":f32,"z":f32,"payload":string}` |
| `temporal_node` | 0 | emitted by the serializer; absent from this store |
| `temporal_edge` | 0 | emitted by the serializer; absent from this store |
| `spatial` | 0 | `{"type":"spatial","id":u64,"x":f32,"y":f32,"z":f32,"payload":string}` |
| `temporal_node` | 0 | emitted by the serializer; absent from this store |
| `temporal_edge` | 0 | emitted by the serializer; absent from this store |

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

Keys repeat across segments (`completion:1` appears in both `seg.0` and `seg.1`)
because **each segment is a complete checkpoint of the whole store, not a delta** —
see "Checkpoint, not append-only" below.

### `block`

The audit chain. Each block links to its predecessor by hash:

```json
{"type":"block",
 "hash": <32 bytes, see below>,
 "prev_hash": <32 bytes, see below>,
 "timestamp_ms":i64,
 "sequence":u64,
 "profile":"abi",
 "query_id":u64,
 "response_id":u64,
 "metadata":"<opaque string>"}
```

**The two hash fields are each encoded one of two ways, and which one depends on
the data.** Both are `[32]u8` in the Zig source and both are written with the same
`w.write(&...)` call, but Zig's JSON stringify emits a byte array as a *string*
when the bytes are valid UTF-8 and as an *array of integers* when they are not.

Measured over 40 segments of the live store:

| Field | as array | as string |
|---|---:|---:|
| `hash` | 4176 | 0 |
| `prev_hash` | 4136 | 40 |

A SHA-256 digest is essentially never valid UTF-8, so `hash` is always an array.
`prev_hash` is an array too — *except* for the genesis block of each segment,
where it is all zero bytes, which **is** valid UTF-8 and so serializes as a
32-character string of `\u0000` escapes.

So a reader that models `hash` as an array and `prev_hash` as a string — the
shape a single sampled record suggests — fails on exactly the genesis block of
every segment. Both fields must accept both encodings.

## Checkpoint, not append-only — the thing that is easy to get wrong

`SegmentStore.flush` serializes the **entire store** into a brand-new epoch, and
`SegmentStore.loadLatest` reads **only the highest active epoch**. Older epochs are
historical checkpoints retained for recovery and dropped by `reclaim` /
`compactRetainingLatest`.

So loading is *not* a replay. A reader that concatenates or layers every active
segment produces wrong results, and the failure is asymmetric in a way that hides
it:

- `kv` and `vector` records are keyed, so re-applying 301 copies of the same
  checkpoint collapses back to the right values and looks correct;
- `block` records are an ordered chain with no key, so the same 301 copies
  **append**. Replaying the observed store yields 50,148 blocks where the real
  chain has ~166, and chain verification then fails because sequence numbers
  repeat.

That is how this was caught here: the keyed counts looked plausible and only the
block count gave it away.

## What a compatible reader must do

1. Read `wdbx.manifest`, honour `active` — ignore segment files not listed.
2. Load the **single highest active epoch**. Do not replay or merge the others.
3. Verify each segment's magic line; treat a bad one as corruption, not as data.
4. Tolerate a segment with only a magic line.
5. Tolerate a truncated final line (interrupted write).
6. Keep `kv` values as opaque strings.
7. Accept **both** encodings for `hash` and `prev_hash` (array or string).
8. Handle all six record types, not just the three present in this store.
9. Tolerate an optional `# checksum:<hex>` trailer (`CHECKSUM_PREFIX` in
   `persistence.zig`); the segments in this store do not carry one.

Items 4, 5, 7, 8 and 9 are the ones a from-scratch implementation gets wrong, and
each surfaces as a failure to open the user's existing store.
