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
Insert vectors/blocks into store:
```
/abi-superpower-wdbx insert --path ./data --input "text to embed"
```

### query
Semantic/persona-scoped retrieval:
```
/abi-superpower-wdbx query --path ./data --query "search text" --persona abbey
```

### stats
Show store statistics:
```
/abi-superpower-wdbx stats --path ./data
```

### compact
Retain newest segments:
```
/abi-superpower-wdbx compact --path ./data --keep 5
```

### benchmark
Insert/search timing:
```
/abi-superpower-wdbx benchmark --path ./data --count 1000
```

### secure
Run compression + homomorphic demo:
```
/abi-superpower-wdbx secure
```

### cluster
Consensus demo or serve:
```
/abi-superpower-wdbx cluster --mode demo --nodes 3
/abi-superpower-wdbx cluster --mode serve --port 8080
```

### api
Serve REST API:
```
/abi-superpower-wdbx api --port 8081
```

## Implementation

Maps to:
- `src/features/wdbx/store.zig` - `Store`, `search()`, `putVector()`
- `src/features/wdbx/segments.zig` - compaction, checkpoints
- `src/features/wdbx/rest.zig` - REST server
- `src/features/wdbx/cluster_rpc.zig` - Raft consensus

## Feature Gate

Requires `feat-wdbx=true` (default). When disabled, returns `FeatureDisabled`.