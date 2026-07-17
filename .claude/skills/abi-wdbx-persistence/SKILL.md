---
name: abi-wdbx-persistence
description: WDBX persistence superpower. JSONL snapshots, CRC32-framed WAL, epoch-gated segment checkpoints, recovery, compaction.
superpower:
  command: "execute"
  parameters:
    - name: "action"
      type: "string"
      enum: ["init", "verify", "compact", "recover", "wal", "segments", "snapshot"]
      description: "Persistence action"
    - name: "path"
      type: "string"
      description: "Database path"
    - name: "keep"
      type: "integer"
      description: "Segments to keep during compaction"
---

# ABI Superpower: WDBX Persistence

Exposes the WDBX durability layer as a superpower. Layered snapshot + WAL + segment design with runtime recovery and compaction.

## Actions

### init
Initialize a new durable store:
```
/abi-wdbx-persistence init --path ./data
```

### verify
Cross-check checkpoint integrity, block-chain validity, WAL frame integrity, and merged checkpoint+WAL chain:
```
/abi-wdbx-persistence verify --path ./data
```

### compact
Retain newest N segment checkpoints, reclaim older ones:
```
/abi-wdbx-persistence compact --path ./data --keep 5
```

### recover
Show recovery status (latest checkpoint + WAL delta):
```
/abi-wdbx-persistence recover --path ./data
```

### wal
Inspect WAL frames:
```
/abi-wdbx-persistence wal --path ./data
```

### segments
List segment checkpoints and epochs:
```
/abi-wdbx-persistence segments --path ./data
```

### snapshot
Create JSONL snapshot (compatibility mirror):
```
/abi-wdbx-persistence snapshot --path ./data
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        WDBX Store                                │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
       │  WAL        │ │  Segments   │ │  Snapshot   │
       │  (wal.zig)  │ │ (segments.zig)│ (persistence)│
       │             │ │             │ │  (mirror)    │
       │ CRC32 frames│ │ Manifest +  │ │  JSONL +    │
       │ Append-only │ │ epoch files │ │  SHA-256    │
       │ Replay +    │ │ compactRetain│ │  checksum   │
       │ corruption  │ │ Latest-N    │ │             │
       │ detection   │ │             │ │             │
       └─────────────┘ └─────────────┘ └─────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    ┌─────────────────────┐
                    │  Recovery           │
                    │  (recovery.zig,     │
                    │   durable_store.zig)│
                    │                     │
                    │  1. Load latest     │
                    │     segment epoch   │
                    │  2. Replay WAL      │
                    │     delta forward   │
                    │  3. Checkpoint      │
                    │     runtime mutations
                    └─────────────────────┘
```

## File Formats

### WAL (`wal.zig`)
- CRC32-framed append-only records
- Supports: `putVector`, `putBlock`, `putSpatial`, `putTemporalNode`, `putTemporalEdge`, `deleteVector`
- Replay reconstructs state deterministically (reuses `persistence.deserialize`)
- Corruption detection: flipped-byte CRC rejection, bad-header rejection

### Segments (`segments.zig`)
- Manifest: `<base>.manifest` — lists epoch checkpoints in order
- Checkpoints: `<base>.seg.<epoch>.jsonl` — immutable epoch snapshots
- `loadLatest()` — loads newest epoch
- `listEpochs()` — lists active epochs
- `reclaimEpochs()` — removes old epochs
- `compactRetainingLatest(keep)` — retains newest N checkpoints

### Snapshot (`persistence.zig` + `persistence_parse.zig`)
- Header: `# ABI-WDBX v1`
- Records: minified JSON per record (`kv`, `vector`, `block`, `spatial`, `temporal_node`, `temporal_edge`)
- Trailer: `# checksum:<sha256-hex>` covering record body
- Load rejects `ChecksumMismatch` on tamper/truncation
- Checksum-less snapshots loadable (backward compatibility)
- Integer field overflow → `FieldOutOfRange` (no panic)
- Vector ID monotonic restore → `CorruptVectorId` on mismatch
- Block timestamp restore → SHA-256 chain reproduces exactly

## Recovery Flow (`recovery.zig` + `durable_store.zig`)

1. **Load latest segment checkpoint** (default runtime baseline)
2. **Replay sidecar WAL** delta forward
3. **Checkpoint runtime mutations** through segments
4. **Compatibility mirror** — monolithic JSONL snapshot still written

CLI commands (`wdbx db *`) recover WAL-ahead state before read/write:
- `block insert/get` → recover → operate
- `query` → recover → search
- `db verify` → cross-checks WAL replay vs current checkpoint block count

## CLI Surface (`abi wdbx db`)

| Command | Description |
|---------|-------------|
| `db init <path>` | Create new segment-backed store |
| `db verify <path>` | Full integrity check (checkpoint + WAL + block chain) |
| `db compact <path> [keep]` | Retain newest N segment checkpoints |
| `block insert <path> <profile> <metadata>` | Write segment checkpoint + WAL |
| `block get <path>` | Read latest block |
| `query <path> [text] [persona]` | Stats, hybrid search, or persona-isolated retrieval |

## Durability Invariants

| Invariant | Enforced By |
|-----------|-------------|
| Append-only WAL | `wal.zig` single-writer, CRC frames |
| Checkpoint immutability | `segments.zig` epoch files never modified |
| Snapshot integrity | SHA-256 trailer, `ChecksumMismatch` on load |
| Block chain integrity | SHA-256 link, `verifyBlocks()` |
| Epoch ordering | Manifest lists epochs sequentially |
| Vector ID monotonicity | `CorruptVectorId` on restore mismatch |
| WAL-ahead recovery | `recovery.zig` prefers segment + WAL delta |

## CLI Access

```
abi wdbx db init ./data
abi wdbx db verify ./data
abi wdbx db compact ./data 5
abi wdbx block insert ./data abbey '{"note":"test"}'
abi wdbx block get ./data
abi wdbx query ./data "search text" abbey
```

## Feature Gates

Requires `feat-wdbx=true` (default). When disabled:
- All write/search paths return `FeatureDisabled`
- No persistence surface exposed

## Claim Boundary

Per `docs/spec/wdbx-north-star.mdx` §3.1 and `docs/contracts/external-claims-audit.mdx`:
- ✅ JSONL snapshot + CRC32 WAL + segment checkpoints
- ✅ Runtime recovery (load epoch + replay WAL)
- ✅ `db verify` cross-checks all layers
- ✅ `db compact` retains newest N epochs
- ✅ Compatibility JSONL mirror for tooling
- ❌ NOT full MVCC visibility (block snapshots + epoch checkpoints only)
- ❌ NOT cross-process/concurrent checkpoint coordination
- ❌ NOT at-rest encryption (proposed Phase 4)
- ❌ NOT signed snapshots (proposed Phase 4)