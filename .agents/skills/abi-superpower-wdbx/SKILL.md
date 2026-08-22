---
name: abi-superpower-wdbx
description: WDBX vector store operations superpower. Includes insert, query, stats, compaction, secure demos, and REST API.
superpower:
  command: "execute"
  parameters:
    - name: "action"
      type: "string"
      enum: ["insert", "query", "stats", "compact", "benchmark", "secure", "cluster", "api"]
      description: "WDBX action to perform"
    - name: "path"
      type: "string"
      description: "Database path"
    - name: "query"
      type: "string"
      description: "Query text for semantic search"
    - name: "keep"
      type: "integer"
      description: "Segments to keep during compaction"
---

> **WDBX moved out of this repository on 2026-08-22.** It now lives in the
> sibling repo `~/dev/active/wdbx` together with `abi-compute`,
> `abi-foundation`, `abi-core`, and `abi-telemetry`; `abi` consumes them by
> relative path. Source paths below therefore read `../wdbx/crates/...`. Run
> WDBX-only tests from that repo (`cargo test --workspace`), and `abi`'s gate
> (`./tools/check.sh`) from here.
>
> Under the Abbey System Constitution
> (`docs/superpowers/specs/2026-08-22-abbey-system-constitution.md`) WDBX is the
> **provenance-aware episodic substrate**, not a vector store. Most of the
> evidence half of its specification is unimplemented; the measured gap list is
> in `docs/superpowers/specs/2026-08-22-wdbx-conformance-gap-analysis.md`. Do not
> describe an episodic capability as Current on the strength of the vector-store
> features that do exist.

# ABI Superpower: WDBX

Exposes WDBX vector store as a superpower for opencode.

## Actions

### insert
Initialize a store and append a block:
```
abi wdbx db init ./data
abi wdbx block insert ./data abbey '{"note":"text to embed"}'
```

### query
Semantic/persona-scoped retrieval:
```
abi wdbx query ./data "search text" abbey
```

### stats
Show store statistics:
```
abi wdbx query ./data
```

### compact
Retain newest segments:
```
abi wdbx db compact ./data 5
```

### benchmark
Insert/search timing:
```
abi wdbx benchmark 1000
```

### secure
Run compression + homomorphic demo:
```
abi wdbx secure demo
```

### cluster
Consensus demo or serve:
```
abi wdbx cluster demo 3
abi wdbx cluster serve 8090 node1 127.0.0.1
```

### api
Serve REST API:
```
abi wdbx api serve 8081
```

## Implementation

Maps to:
- `../wdbx/crates/abi-wdbx/src/store.rs` - `Store`, `search()`, `putVector()`
- `../wdbx/crates/abi-wdbx/src/segments.rs` - compaction, checkpoints
- `../wdbx/crates/abi-wdbx/src/rest.rs` - REST server
- `../wdbx/crates/abi-wdbx/src/cluster_rpc.rs` - Raft consensus

## Build and runtime boundary

`abi-wdbx` is a normal Rust workspace crate; there is no `feat-wdbx` switch or
`FeatureDisabled` stub. Use the real `abi wdbx` subcommands shown by
`abi wdbx help`. Tests must use a scratch path or disable persistence so they
never open the user's live `~/.abi` store.
