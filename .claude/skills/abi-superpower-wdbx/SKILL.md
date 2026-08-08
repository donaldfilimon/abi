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
- `crates/abi-wdbx/src/store.rs` - `Store`, `search()`, `putVector()`
- `crates/abi-wdbx/src/segments.rs` - compaction, checkpoints
- `crates/abi-wdbx/src/rest.rs` - REST server
- `crates/abi-wdbx/src/cluster_rpc.rs` - Raft consensus

## Build and runtime boundary

`abi-wdbx` is a normal Rust workspace crate; there is no `feat-wdbx` switch or
`FeatureDisabled` stub. Use the real `abi wdbx` subcommands shown by
`abi wdbx help`. Tests must use a scratch path or disable persistence so they
never open the user's live `~/.abi` store.
