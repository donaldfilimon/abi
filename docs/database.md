# Database (WDBX)

**WDBX** is ABI's built-in vector database solution, optimized for high-dimensional embedding storage and retrieval.

## Features

- **Vector Search**: Dot product, Cosine similarity, L2 Euclidean distance.
- **Zero-Copy**: Leverages Zig's manual memory management for efficiency.
- **Backup/Restore**: Simple snapshotting capabilities.

## Usage

```zig
const wdbx = abi.wdbx;

var db = try wdbx.createDatabase(allocator, .{ .dimension = 1536 });
defer db.deinit();

// Insert
try db.insertVector(id, embedding_slice);

// Search
const results = try db.searchVectors(query_embedding, 10);
```

## Security: Backup & Restore

> [!IMPORTANT]
> **Security Advisory**: Improper path validation in versions prior to 0.2.1 allowed directory traversal.

**Safe Practices**:

1.  **Restricted Directory**: All backup/restore operations are confined to the `backups/` directory.
2.  **Input Validation**: Filenames must **not** contain:
    - Path traversal sequences (`..`)
    - Absolute paths (`/etc/passwd`, `C:\Windows`)
    - Drive letters
3.  **Validation Error**: The API will return `PathValidationError` if an unsafe path is detected.

```zig
// GOOD
try db.restore("snapshot_2025.db");

// BAD (Will fail)
try db.restore("../../../secret.txt");
```
