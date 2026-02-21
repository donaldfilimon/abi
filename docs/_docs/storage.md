---
title: Storage
description: Unified file/object storage with vtable-based backend abstraction and path traversal protection
section: Data
order: 3
---

# Storage

The storage module (`src/features/storage/mod.zig`) provides unified
file and object storage with a vtable-based backend abstraction. It
ships with an in-memory backend and is designed for extension to local
filesystem, S3, and GCS backends. All keys are validated against path
traversal attacks.

## Features

- **Unified API**: Same `put`/`get`/`delete`/`list`/`exists` operations across all backends
- **Vtable abstraction**: Backend interface via function pointer table for runtime polymorphism
- **Memory backend**: In-memory `StringHashMap`-based storage (available now)
- **Path traversal protection**: All keys validated against `../` sequences and absolute paths
- **Object metadata**: Content type and up to 4 custom key-value metadata entries per object
- **Capacity limits**: Configurable maximum storage size with `StorageFull` error
- **Thread-safe**: RwLock for concurrent access
- **Planned backends**: Local filesystem, S3, GCS (return `BackendNotAvailable` until implemented)

## Build Configuration

```bash
# Enable (default)
zig build -Denable-storage=true

# Disable
zig build -Denable-storage=false
```

**Namespace**: `abi.storage`

## Quick Start

### Framework Integration

```zig
const abi = @import("abi");

// Initialize with storage enabled
var fw = try abi.Framework.init(allocator, .{
    .storage = .{
        .backend = .memory,
        .max_object_size_mb = 100,
    },
});
defer fw.deinit();
```

### Standalone Usage

```zig
const storage = abi.storage;

// Initialize the global storage singleton
try storage.init(allocator, .{
    .backend = .memory,
    .max_object_size_mb = 100,  // 100 MB capacity
});
defer storage.deinit();

// Store an object
try storage.putObject(allocator, "images/photo.jpg", image_data);

// Store with metadata
try storage.putObjectWithMetadata(allocator, "docs/readme.md", content, .{
    .content_type = "text/markdown",
});

// Retrieve an object
const data = try storage.getObject(allocator, "images/photo.jpg");
defer allocator.free(data);

// Check existence
const exists = try storage.objectExists("images/photo.jpg");

// Delete an object
const deleted = try storage.deleteObject("images/photo.jpg");

// List objects by prefix
const objects = try storage.listObjects(allocator, "images/");
defer allocator.free(objects);

for (objects) |obj| {
    std.debug.print("{s}: {} bytes ({s})\n", .{
        obj.key, obj.size, obj.content_type,
    });
}

// Get aggregate statistics
const st = storage.stats();
std.debug.print("Objects: {}, Total bytes: {}\n", .{
    st.total_objects, st.total_bytes,
});
```

## API Reference

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `init` | `(allocator, StorageConfig) !void` | Initialize the global storage singleton |
| `deinit` | `() void` | Tear down storage and free all objects |
| `isEnabled` | `() bool` | Returns `true` when storage is compiled in |
| `isInitialized` | `() bool` | Returns `true` after successful `init()` |
| `putObject` | `(allocator, key, data) !void` | Store an object |
| `putObjectWithMetadata` | `(allocator, key, data, meta) !void` | Store with metadata |
| `getObject` | `(allocator, key) ![]const u8` | Retrieve object data (caller owns returned slice) |
| `deleteObject` | `(key) !bool` | Delete an object; returns whether it existed |
| `objectExists` | `(key) !bool` | Check if an object exists |
| `listObjects` | `(allocator, prefix) ![]StorageObject` | List objects matching a key prefix |
| `stats` | `() StorageStats` | Aggregate statistics |

### Types

| Type | Description |
|------|-------------|
| `StorageConfig` | Configuration struct (backend type, max size) |
| `StorageBackend` | Enum: `.memory`, `.local`, `.s3`, `.gcs` |
| `StorageObject` | Object descriptor: key, size, content_type, last_modified |
| `ObjectMetadata` | Per-object metadata: content_type + up to 4 custom key-value pairs |
| `StorageStats` | Aggregate statistics: total_objects, total_bytes, backend |
| `StorageError` | Error set (see below) |

### Error Types

| Error | Description |
|-------|-------------|
| `FeatureDisabled` | Storage module not compiled in |
| `ObjectNotFound` | Key does not exist |
| `BucketNotFound` | Bucket/namespace not found |
| `PermissionDenied` | Access denied |
| `StorageFull` | Capacity limit exceeded |
| `OutOfMemory` | Allocation failure |
| `InvalidKey` | Key contains path traversal (`../`) or is absolute |
| `BackendNotAvailable` | Requested backend not implemented |

## Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | `StorageBackend` | `.memory` | Storage backend type |
| `max_object_size_mb` | `u32` | -- | Maximum total storage in MB (0 = unlimited) |

## Backend Architecture

The storage module uses a vtable-based backend abstraction. Each backend
implements seven function pointers:

```
VTable {
    put:       fn(*anyopaque, key, data, ?metadata) !void
    get:       fn(*anyopaque, allocator, key) ![]u8
    delete:    fn(*anyopaque, key) !bool
    list:      fn(*anyopaque, allocator, prefix) ![]StorageObject
    exists:    fn(*anyopaque, key) bool
    getStats:  fn(*anyopaque) StorageStats
    deinitFn:  fn(*anyopaque) void
}
```

The `Backend` struct wraps an `*anyopaque` pointer and a vtable pointer,
providing type-erased dispatch to the concrete implementation. This design
allows adding new backends (local filesystem, S3, GCS) without changing the
public API.

### Memory Backend

The built-in memory backend stores objects in a `StringHashMap` with
owned copies of keys, values, and content types. It enforces capacity
limits and supports overwriting existing keys (old data is freed
atomically after the new entry is committed).

### Planned Backends

| Backend | Status | Description |
|---------|--------|-------------|
| `.memory` | Available | In-memory HashMap storage |
| `.local` | Planned | Local filesystem (requires I/O backend init) |
| `.s3` | Planned | Amazon S3 (requires HTTP client) |
| `.gcs` | Planned | Google Cloud Storage (requires HTTP client) |

Requesting a planned backend currently returns `error.BackendNotAvailable`.

## Key Validation

All keys are validated before any operation:

- **Length**: Must be 1-4096 characters
- **No absolute paths**: Keys starting with `/` are rejected
- **No traversal**: `../` sequences (at start, end, or between slashes) are rejected

Invalid keys return `error.InvalidKey`.

```zig
// Valid keys
try storage.putObject(allocator, "data/file.txt", content);
try storage.putObject(allocator, "user/42/avatar.png", image);

// Invalid keys (return error.InvalidKey)
// storage.putObject(allocator, "../etc/passwd", data);  // path traversal
// storage.putObject(allocator, "/absolute/path", data);  // absolute path
```

## Disabling at Build Time

```bash
zig build -Denable-storage=false
```

When disabled, `abi.storage` resolves to `src/features/storage/stub.zig`,
which returns `error.FeatureDisabled` for all mutating operations and safe
defaults for read-only calls.

## Examples

See [`examples/storage.zig`](https://github.com/donaldfilimon/abi/blob/main/examples/storage.zig)
for a complete working example.

## Related

- [Database](database.html) -- Vector database with persistent storage
- [Cache](cache.html) -- In-memory caching (faster, no persistence)
- [Search](search.html) -- Full-text search index
- [Architecture](architecture.html) -- Comptime feature gating pattern

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
