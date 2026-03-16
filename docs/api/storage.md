---
title: storage API
purpose: Generated API reference for storage
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# storage

> Storage Module

Unified file/object storage with vtable-based backend abstraction.

Architecture:
- Backend vtable: put/get/delete/list/exists/deinit function pointers
- Memory backend: StringHashMap-based in-memory storage
- Local backend: Planned (requires I/O backend init)
- S3/GCS: Planned (requires HTTP client)

Security: path traversal validation on all keys.

**Source:** [`src/features/storage/mod.zig`](../../src/features/storage/mod.zig)

**Build flag:** `-Dfeat_storage=true`

---

## API

### <a id="pub-fn-init-allocator-std-mem-allocator-config-storageconfig-storageerror-void"></a>`pub fn init(allocator: std.mem.Allocator, config: StorageConfig) StorageError!void`

<sup>**fn**</sup> | [source](../../src/features/storage/mod.zig#L496)

Initialize the global storage singleton. The `memory` and `local` backends
are available; `s3` and `gcs` return `BackendNotAvailable`.

### <a id="pub-fn-deinit-void"></a>`pub fn deinit() void`

<sup>**fn**</sup> | [source](../../src/features/storage/mod.zig#L524)

Tear down the storage backend, freeing all stored objects.

### <a id="pub-fn-putobject-std-mem-allocator-key-const-u8-data-const-u8-storageerror-void"></a>`pub fn putObject( _: std.mem.Allocator, key: []const u8, data: []const u8, ) StorageError!void`

<sup>**fn**</sup> | [source](../../src/features/storage/mod.zig#L541)

Store an object by key. Validates the key for path traversal.

### <a id="pub-fn-putobjectwithmetadata-std-mem-allocator-key-const-u8-data-const-u8-metadata-objectmetadata-storageerror-void"></a>`pub fn putObjectWithMetadata( _: std.mem.Allocator, key: []const u8, data: []const u8, metadata: ObjectMetadata, ) StorageError!void`

<sup>**fn**</sup> | [source](../../src/features/storage/mod.zig#L556)

Store an object with custom metadata (content type, key-value pairs).

### <a id="pub-fn-getobject-allocator-std-mem-allocator-key-const-u8-storageerror-const-u8"></a>`pub fn getObject(allocator: std.mem.Allocator, key: []const u8) StorageError![]const u8`

<sup>**fn**</sup> | [source](../../src/features/storage/mod.zig#L572)

Retrieve an object's data by key. Caller owns the returned slice.

### <a id="pub-fn-deleteobject-key-const-u8-storageerror-bool"></a>`pub fn deleteObject(key: []const u8) StorageError!bool`

<sup>**fn**</sup> | [source](../../src/features/storage/mod.zig#L583)

Delete an object by key. Returns `true` if the key was present.

### <a id="pub-fn-objectexists-key-const-u8-storageerror-bool"></a>`pub fn objectExists(key: []const u8) StorageError!bool`

<sup>**fn**</sup> | [source](../../src/features/storage/mod.zig#L594)

Check whether an object exists without reading it.

### <a id="pub-fn-listobjects-allocator-std-mem-allocator-prefix-const-u8-storageerror-storageobject"></a>`pub fn listObjects( allocator: std.mem.Allocator, prefix: []const u8, ) StorageError![]StorageObject`

<sup>**fn**</sup> | [source](../../src/features/storage/mod.zig#L605)

List objects whose keys start with `prefix`. Caller owns the returned slice.

### <a id="pub-fn-stats-storagestats"></a>`pub fn stats() StorageStats`

<sup>**fn**</sup> | [source](../../src/features/storage/mod.zig#L618)

Snapshot object count, byte usage, and active backend type.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` on supported hosts. On Darwin 25+ / 26+, use `zig fmt --check ...` plus `./tools/scripts/run_build.sh <step>`. For docs generation, use `zig build gendocs` or `./tools/scripts/run_build.sh gendocs` on Darwin.
