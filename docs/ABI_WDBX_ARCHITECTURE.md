---
title: ABI WDBX Architecture
purpose: Technical specification and architectural guidelines for WDBX
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2962+08416b44f
---

# ABI WDBX Architecture

WDBX is the historical name for ABI's semantic storage engine. In the current
repo, the public package surface is `abi.database`, while the deeper
implementation still contains WDBX-oriented modules and terminology.

This document explains the architecture in terms of the current codebase rather
than the older standalone-package model.

## Position in the repo

- Public package entry: `abi.database`
- Canonical implementation root: `src/core/database/`
- Feature-gated facade: `src/features/database/mod.zig`
- Compatibility and archival naming: `wdbx`, semantic memory, semantic store

The important boundary is:

- external callers use `abi.database`
- internal database implementation lives under `src/core/database/`

## Goals

The database subsystem is not just a vector store. It is intended to support:

- semantic similarity search
- metadata-aware retrieval
- weighted memory ranking
- backup and restore flows
- distributed coordination paths
- operational surfaces such as CLI, HTTP, MCP, and diagnostics

## Layers

### 1. Core and shared primitives

`src/core/database/core/`

Provides the small foundational building blocks used everywhere else:

- ids
- common types
- canonical error definitions
- time and allocator helpers

### 2. Durable storage and block layout

Key files:

- `src/core/database/storage.zig`
- `src/core/database/block/`
- `src/core/database/persistence.zig`
- `src/core/database/formats/`

This layer owns serialization, durable layout, backup materialization, and
format-specific mechanics such as block headers, segment logs, compression, and
binary codecs.

### 3. Retrieval engines and indexes

Key files:

- `src/core/database/semantic_store/mod.zig`
- `src/core/database/hnsw.zig`
- `src/core/database/index.zig`
- `src/core/database/fulltext.zig`
- `src/core/database/hybrid.zig`
- `src/core/database/filter.zig`
- `src/core/database/parallel_search.zig`

This is the primary retrieval plane. It combines vector search, metadata
filters, full-text indexing, and hybrid ranking paths.

### 4. Quantization, SIMD, and performance work

Key files:

- `src/core/database/simd.zig`
- `src/core/database/quantization.zig`
- `src/core/database/product_quantizer.zig`
- `src/core/database/parallel_hnsw.zig`
- `src/core/database/clustering.zig`

This layer exists to keep the database useful at larger scales where storage
size, search throughput, and approximation quality all matter.

### 5. Memory, ranking, and query composition

Key directories:

- `src/core/database/query/`
- `src/core/database/ranking/`
- `src/core/database/memory/`
- `src/core/database/trace/`
- `src/core/database/graph/`
- `src/core/database/vector/`
- `src/core/database/profile/`

These modules are where the "memory fabric" idea becomes concrete. They allow
the subsystem to reason about more than raw nearest-neighbor search by layering
ranking, traversal, scoring traces, and behavior-sensitive context assembly on
top of core storage.

### 6. Distributed and operational surfaces

Key files:

- `src/core/database/distributed/`
- `src/core/database/dist/`
- `src/core/database/http.zig`
- `src/core/database/cli/`
- `src/core/database/api/`

These modules handle clustering, replication, recovery, remote coordination, and
operator-facing access.

## Request Flow

A typical semantic retrieval request follows this shape:

1. Open or connect to a database handle.
2. Insert vectors plus metadata into the semantic store.
3. Query through vector, metadata, or hybrid retrieval paths.
4. Rank or filter the candidate set.
5. Return search results, diagnostics, or derived context to a caller.

At the public ABI layer this usually looks like:

- `abi.database.open`
- `abi.database.insert`
- `abi.database.search`
- `abi.database.backup`
- `abi.database.restore`

At the more detailed surface it often routes through:

- `abi.database.semantic_store`
- `abi.database.neural`
- `abi.database.fulltext`
- `abi.database.hybrid`

## Why WDBX terminology still exists

WDBX still appears in:

- historical filenames
- lower-level engine modules
- plan documents
- dataset conversion paths
- compatibility-oriented docs and comments

That naming does not mean there is still a public `@import("wdbx")` package
surface. The public package contract is now the ABI database feature namespace.

## How to think about the subsystem

The best mental model is:

- storage layer for durable vector and metadata state
- retrieval layer for semantic and hybrid search
- memory layer for context shaping and ranking
- operational layer for backups, diagnostics, and distributed coordination

In other words, WDBX is the engine lineage; `abi.database` is the
current consumer-facing API.

## Practical Guidance

- Update `src/features/database/mod.zig` and `src/features/database/stub.zig` together when public signatures move.
- Prefer fixing public docs to reference `abi.database`.
- Keep generated docs and CLI help aligned when command or public-surface names change.
- Treat `src/core/database/` as the canonical place to understand behavior, even when the exported surface is smaller.
