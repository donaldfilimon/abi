---
name: wdbx
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

# WDBX Superpower Plugin

Core WDBX capabilities for OpenCode within the ABI framework.

## Capabilities

- WDBX subsystem integration
- Plugin framework registration
- Runtime lifecycle management
- Configuration and settings management
- Status monitoring and reporting

## Integration Points

- ABI's WDBX subsystem integration
- OpenCode plugin framework integration
- Runtime lifecycle management
- Configuration and settings management

## Actions

### insert
Insert vectors/blocks into store.

### query
Semantic/persona-scoped retrieval.

### stats
Show store statistics.

### compact
Retain newest segments.

### benchmark / secure / cluster / api
As per WDBX superpower.

## Implementation

Maps to:
- `crates/abi-wdbx/src/store.rs`
- `crates/abi-wdbx/src/segments.rs`
- `crates/abi-wdbx/src/rest.rs`
- `crates/abi-wdbx/src/cluster_rpc.rs`

## Build and runtime boundary

`abi-wdbx` is a normal Rust workspace crate; there is no `feat-wdbx` switch.
Use the real `abi wdbx` subcommands shown by `abi wdbx help`, with scratch
stores for tests.
