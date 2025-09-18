# Vector Database API

This document provides comprehensive API documentation for the `database` module.

## Table of Contents

- [Overview](#overview)
- [Core Types](#core-types)
- [Functions](#functions)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Overview

The WDBX vector database provides high-performance storage and retrieval of high-dimensional vectors.

### Key Features

- **SIMD-optimized** vector operations
- **Binary format** for efficient storage
- **k-NN search** with configurable distance metrics
- **Memory-mapped** file support

## Core Types

### `Db`

The main database structure.

```zig
pub const Db = struct {
    file_path: []const u8,
    dimension: usize,
    row_count: usize,
    // ...
};
```

## Functions

### `open`

Opens or creates a database file.

```zig
pub fn open(path: []const u8, create: bool) !Db
```

**Parameters:**
- `path`: Path to the database file
- `create`: Create file if it doesn't exist

**Returns:** Database instance or error

## Examples

### Basic Usage

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var db = try abi.database.Db.open("vectors.wdbx", true);
    defer db.close();

    // Initialize with 384-dimensional vectors
    try db.init(384);

    // Add a vector
    const embedding = [_]f32{0.1, 0.2, 0.3} ++ ([_]f32{0.0} ** 381);
    const id = try db.addEmbedding(&embedding);
}
```

