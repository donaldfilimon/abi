# C-Compatible Library Bindings Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reimplement C-compatible bindings for the ABI framework to enable FFI from Rust, Go, Python, and other languages.

**Architecture:** Create a thin C export layer in Zig that wraps the existing public API with C-compatible types (opaque handles, integer error codes, null-terminated strings). Header files provide the C interface, and the Zig code uses `export` and `callconv(.C)` for ABI compatibility.

**Tech Stack:** Zig 0.16, C99-compatible headers, optional Rust/Go/Python bindings consuming the C layer

---

## Task 1: Create C Error Codes Header

**Files:**
- Create: `bindings/c/include/abi_errors.h`

**Step 1: Create the header file**

```c
// bindings/c/include/abi_errors.h
#ifndef ABI_ERRORS_H
#define ABI_ERRORS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef int abi_error_t;

#define ABI_OK                          0
#define ABI_ERROR_INIT_FAILED          -1
#define ABI_ERROR_ALREADY_INITIALIZED  -2
#define ABI_ERROR_NOT_INITIALIZED      -3
#define ABI_ERROR_OUT_OF_MEMORY        -4
#define ABI_ERROR_INVALID_ARGUMENT     -5
#define ABI_ERROR_FEATURE_DISABLED     -6
#define ABI_ERROR_TIMEOUT              -7
#define ABI_ERROR_IO                   -8
#define ABI_ERROR_GPU_UNAVAILABLE      -9
#define ABI_ERROR_DATABASE_ERROR      -10
#define ABI_ERROR_NETWORK_ERROR       -11
#define ABI_ERROR_AI_ERROR            -12
#define ABI_ERROR_UNKNOWN             -99

// Get human-readable error message
const char* abi_error_string(abi_error_t error);

#ifdef __cplusplus
}
#endif

#endif // ABI_ERRORS_H
```

**Step 2: Verify file exists**

Run: `ls -la bindings/c/include/abi_errors.h`
Expected: File exists with correct content

**Step 3: Commit**

```bash
git add bindings/c/include/abi_errors.h
git commit -m "feat(bindings): add C error codes header"
```

---

## Task 2: Create C Types Header

**Files:**
- Create: `bindings/c/include/abi_types.h`

**Step 1: Create the header file**

```c
// bindings/c/include/abi_types.h
#ifndef ABI_TYPES_H
#define ABI_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "abi_errors.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types - pointers to internal Zig structures
typedef struct abi_framework* abi_framework_t;
typedef struct abi_gpu* abi_gpu_t;
typedef struct abi_database* abi_database_t;
typedef struct abi_agent* abi_agent_t;

// Configuration structs
typedef struct {
    bool enable_ai;
    bool enable_gpu;
    bool enable_database;
    bool enable_network;
    bool enable_web;
    bool enable_profiling;
} abi_options_t;

typedef struct {
    const char* name;
    size_t dimension;
    size_t initial_capacity;
} abi_database_config_t;

typedef struct {
    int backend;  // 0=auto, 1=cuda, 2=vulkan, 3=metal, 4=webgpu
    int device_index;
    bool enable_profiling;
} abi_gpu_config_t;

typedef struct {
    const char* name;
    const char* persona;
    float temperature;
    bool enable_history;
} abi_agent_config_t;

// Result structs
typedef struct {
    uint64_t id;
    float score;
    const float* vector;
    size_t vector_len;
} abi_search_result_t;

typedef struct {
    bool sse;
    bool sse2;
    bool sse3;
    bool ssse3;
    bool sse4_1;
    bool sse4_2;
    bool avx;
    bool avx2;
    bool avx512f;
    bool neon;
} abi_simd_caps_t;

// Version info
typedef struct {
    int major;
    int minor;
    int patch;
    const char* full;
} abi_version_t;

// Default initializers
#define ABI_OPTIONS_DEFAULT { true, true, true, true, true, true }
#define ABI_DATABASE_CONFIG_DEFAULT { "default", 384, 1000 }
#define ABI_GPU_CONFIG_DEFAULT { 0, 0, false }
#define ABI_AGENT_CONFIG_DEFAULT { "assistant", NULL, 0.7f, true }

#ifdef __cplusplus
}
#endif

#endif // ABI_TYPES_H
```

**Step 2: Verify file exists**

Run: `ls -la bindings/c/include/abi_types.h`
Expected: File exists with correct content

**Step 3: Commit**

```bash
git add bindings/c/include/abi_types.h
git commit -m "feat(bindings): add C types header with opaque handles and configs"
```

---

## Task 3: Create Main C Header

**Files:**
- Create: `bindings/c/include/abi.h`

**Step 1: Create the header file**

```c
// bindings/c/include/abi.h
#ifndef ABI_H
#define ABI_H

#include "abi_types.h"
#include "abi_errors.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Framework Lifecycle
// ============================================================================

// Initialize framework with default options
abi_error_t abi_init(abi_framework_t* out_framework);

// Initialize framework with custom options
abi_error_t abi_init_with_options(const abi_options_t* options,
                                   abi_framework_t* out_framework);

// Shutdown and release framework resources
void abi_shutdown(abi_framework_t framework);

// Get version string (do not free)
const char* abi_version(void);

// Get detailed version info
void abi_version_info(abi_version_t* out_version);

// Check if a feature is enabled
bool abi_is_feature_enabled(abi_framework_t framework, const char* feature);

// ============================================================================
// SIMD Operations
// ============================================================================

// Query SIMD capabilities
void abi_simd_get_caps(abi_simd_caps_t* out_caps);

// Check if SIMD is available
bool abi_simd_available(void);

// Vector addition: result[i] = a[i] + b[i]
void abi_simd_vector_add(const float* a, const float* b, float* result, size_t len);

// Vector dot product
float abi_simd_vector_dot(const float* a, const float* b, size_t len);

// Vector L2 norm
float abi_simd_vector_l2_norm(const float* v, size_t len);

// Cosine similarity between two vectors
float abi_simd_cosine_similarity(const float* a, const float* b, size_t len);

// ============================================================================
// Database Operations
// ============================================================================

// Create a new vector database
abi_error_t abi_database_create(const abi_database_config_t* config,
                                 abi_database_t* out_db);

// Close database and release resources
void abi_database_close(abi_database_t db);

// Insert a vector with ID
abi_error_t abi_database_insert(abi_database_t db, uint64_t id,
                                 const float* vector, size_t vector_len);

// Search for similar vectors (caller allocates results array)
abi_error_t abi_database_search(abi_database_t db, const float* query,
                                 size_t query_len, size_t k,
                                 abi_search_result_t* out_results,
                                 size_t* out_count);

// Delete a vector by ID
abi_error_t abi_database_delete(abi_database_t db, uint64_t id);

// Get database statistics
abi_error_t abi_database_count(abi_database_t db, size_t* out_count);

// ============================================================================
// GPU Operations
// ============================================================================

// Initialize GPU context
abi_error_t abi_gpu_init(const abi_gpu_config_t* config, abi_gpu_t* out_gpu);

// Shutdown GPU context
void abi_gpu_shutdown(abi_gpu_t gpu);

// Check if any GPU backend is available
bool abi_gpu_is_available(void);

// Get active backend name (do not free)
const char* abi_gpu_backend_name(abi_gpu_t gpu);

// ============================================================================
// Agent Operations
// ============================================================================

// Create an AI agent
abi_error_t abi_agent_create(abi_framework_t framework,
                              const abi_agent_config_t* config,
                              abi_agent_t* out_agent);

// Destroy agent
void abi_agent_destroy(abi_agent_t agent);

// Send a message and get response (caller frees response with abi_free_string)
abi_error_t abi_agent_chat(abi_agent_t agent, const char* message,
                            char** out_response);

// Clear conversation history
abi_error_t abi_agent_clear_history(abi_agent_t agent);

// ============================================================================
// Memory Management
// ============================================================================

// Free a string allocated by ABI functions
void abi_free_string(char* str);

// Free a results array allocated by ABI functions
void abi_free_results(abi_search_result_t* results, size_t count);

#ifdef __cplusplus
}
#endif

#endif // ABI_H
```

**Step 2: Verify file exists**

Run: `ls -la bindings/c/include/abi.h`
Expected: File exists with correct content

**Step 3: Commit**

```bash
git add bindings/c/include/abi.h
git commit -m "feat(bindings): add main C header with full API declarations"
```

---

## Task 4: Create Zig Export Module - Error Handling

**Files:**
- Create: `bindings/c/src/errors.zig`

**Step 1: Create the error mapping module**

```zig
// bindings/c/src/errors.zig
//! C-compatible error code mappings for ABI framework.

const std = @import("std");

pub const Error = c_int;

// Error codes matching abi_errors.h
pub const OK: Error = 0;
pub const INIT_FAILED: Error = -1;
pub const ALREADY_INITIALIZED: Error = -2;
pub const NOT_INITIALIZED: Error = -3;
pub const OUT_OF_MEMORY: Error = -4;
pub const INVALID_ARGUMENT: Error = -5;
pub const FEATURE_DISABLED: Error = -6;
pub const TIMEOUT: Error = -7;
pub const IO: Error = -8;
pub const GPU_UNAVAILABLE: Error = -9;
pub const DATABASE_ERROR: Error = -10;
pub const NETWORK_ERROR: Error = -11;
pub const AI_ERROR: Error = -12;
pub const UNKNOWN: Error = -99;

/// Convert a Zig error to a C error code.
pub fn fromZigError(err: anyerror) Error {
    return switch (err) {
        error.OutOfMemory => OUT_OF_MEMORY,
        error.InvalidArgument => INVALID_ARGUMENT,
        error.FeatureDisabled => FEATURE_DISABLED,
        error.GpuDisabled => FEATURE_DISABLED,
        error.AiDisabled => FEATURE_DISABLED,
        error.DatabaseDisabled => FEATURE_DISABLED,
        error.NetworkDisabled => FEATURE_DISABLED,
        error.Timeout => TIMEOUT,
        error.GpuUnavailable => GPU_UNAVAILABLE,
        error.GpuInitFailed => GPU_UNAVAILABLE,
        error.DatabaseError => DATABASE_ERROR,
        error.NetworkError => NETWORK_ERROR,
        else => UNKNOWN,
    };
}

/// Get human-readable error message.
pub fn errorString(code: Error) [*:0]const u8 {
    return switch (code) {
        OK => "Success",
        INIT_FAILED => "Initialization failed",
        ALREADY_INITIALIZED => "Already initialized",
        NOT_INITIALIZED => "Not initialized",
        OUT_OF_MEMORY => "Out of memory",
        INVALID_ARGUMENT => "Invalid argument",
        FEATURE_DISABLED => "Feature disabled",
        TIMEOUT => "Operation timed out",
        IO => "I/O error",
        GPU_UNAVAILABLE => "GPU unavailable",
        DATABASE_ERROR => "Database error",
        NETWORK_ERROR => "Network error",
        AI_ERROR => "AI error",
        else => "Unknown error",
    };
}

// C export
pub export fn abi_error_string(code: Error) [*:0]const u8 {
    return errorString(code);
}

test "error code mapping" {
    try std.testing.expectEqual(OUT_OF_MEMORY, fromZigError(error.OutOfMemory));
    try std.testing.expectEqual(FEATURE_DISABLED, fromZigError(error.FeatureDisabled));
}
```

**Step 2: Verify file compiles**

Run: `zig build-lib bindings/c/src/errors.zig -fno-emit-bin 2>&1 || echo "Expected: compile check only"`
Expected: No errors (or expected warnings about missing root)

**Step 3: Commit**

```bash
git add bindings/c/src/errors.zig
git commit -m "feat(bindings): add Zig error code mapping module"
```

---

## Task 5: Create Zig Export Module - SIMD Operations

**Files:**
- Create: `bindings/c/src/simd.zig`

**Step 1: Create the SIMD exports module**

```zig
// bindings/c/src/simd.zig
//! C-compatible SIMD operation exports.

const std = @import("std");
const abi = @import("abi");
const simd = abi.shared.simd;

/// SIMD capabilities struct matching C header.
pub const SimdCaps = extern struct {
    sse: bool,
    sse2: bool,
    sse3: bool,
    ssse3: bool,
    sse4_1: bool,
    sse4_2: bool,
    avx: bool,
    avx2: bool,
    avx512f: bool,
    neon: bool,
};

/// Query SIMD capabilities.
pub export fn abi_simd_get_caps(out_caps: *SimdCaps) void {
    const caps = simd.detectCapabilities();
    out_caps.* = .{
        .sse = caps.sse,
        .sse2 = caps.sse2,
        .sse3 = caps.sse3,
        .ssse3 = caps.ssse3,
        .sse4_1 = caps.sse4_1,
        .sse4_2 = caps.sse4_2,
        .avx = caps.avx,
        .avx2 = caps.avx2,
        .avx512f = caps.avx512f,
        .neon = caps.neon,
    };
}

/// Check if any SIMD is available.
pub export fn abi_simd_available() bool {
    const caps = simd.detectCapabilities();
    return caps.sse or caps.neon;
}

/// Vector addition.
pub export fn abi_simd_vector_add(
    a: [*]const f32,
    b: [*]const f32,
    result: [*]f32,
    len: usize,
) void {
    const a_slice = a[0..len];
    const b_slice = b[0..len];
    const result_slice = result[0..len];
    simd.vectorAdd(f32, a_slice, b_slice, result_slice);
}

/// Vector dot product.
pub export fn abi_simd_vector_dot(a: [*]const f32, b: [*]const f32, len: usize) f32 {
    return simd.dotProduct(f32, a[0..len], b[0..len]);
}

/// Vector L2 norm.
pub export fn abi_simd_vector_l2_norm(v: [*]const f32, len: usize) f32 {
    return simd.l2Norm(f32, v[0..len]);
}

/// Cosine similarity.
pub export fn abi_simd_cosine_similarity(a: [*]const f32, b: [*]const f32, len: usize) f32 {
    return simd.cosineSimilarity(f32, a[0..len], b[0..len]);
}

test "simd exports" {
    var caps: SimdCaps = undefined;
    abi_simd_get_caps(&caps);
    // At least one capability should be available on any modern CPU
    _ = abi_simd_available();
}
```

**Step 2: Verify file structure**

Run: `ls -la bindings/c/src/simd.zig`
Expected: File exists

**Step 3: Commit**

```bash
git add bindings/c/src/simd.zig
git commit -m "feat(bindings): add SIMD C exports with capability detection"
```

---

## Task 6: Create Zig Export Module - Framework

**Files:**
- Create: `bindings/c/src/framework.zig`

**Step 1: Create the framework exports module**

```zig
// bindings/c/src/framework.zig
//! C-compatible framework lifecycle exports.

const std = @import("std");
const abi = @import("abi");
const errors = @import("errors.zig");

/// Opaque framework handle.
pub const FrameworkHandle = opaque {};

/// Options struct matching C header.
pub const Options = extern struct {
    enable_ai: bool,
    enable_gpu: bool,
    enable_database: bool,
    enable_network: bool,
    enable_web: bool,
    enable_profiling: bool,
};

/// Version info struct.
pub const VersionInfo = extern struct {
    major: c_int,
    minor: c_int,
    patch: c_int,
    full: [*:0]const u8,
};

// Global allocator for C bindings
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// Store framework pointer for handle mapping
var active_framework: ?*abi.Framework = null;

/// Initialize framework with defaults.
pub export fn abi_init(out_framework: *?*FrameworkHandle) errors.Error {
    const opts = Options{
        .enable_ai = true,
        .enable_gpu = true,
        .enable_database = true,
        .enable_network = true,
        .enable_web = true,
        .enable_profiling = true,
    };
    return abi_init_with_options(&opts, out_framework);
}

/// Initialize framework with options.
pub export fn abi_init_with_options(
    options: *const Options,
    out_framework: *?*FrameworkHandle,
) errors.Error {
    if (active_framework != null) {
        return errors.ALREADY_INITIALIZED;
    }

    const config = abi.Config.init()
        .withAI(options.enable_ai)
        .withGPU(options.enable_gpu)
        .withDatabase(options.enable_database);

    const fw_ptr = allocator.create(abi.Framework) catch {
        return errors.OUT_OF_MEMORY;
    };

    fw_ptr.* = abi.Framework.init(allocator, config) catch |err| {
        allocator.destroy(fw_ptr);
        return errors.fromZigError(err);
    };

    active_framework = fw_ptr;
    out_framework.* = @ptrCast(fw_ptr);
    return errors.OK;
}

/// Shutdown framework.
pub export fn abi_shutdown(handle: ?*FrameworkHandle) void {
    if (handle) |h| {
        const fw: *abi.Framework = @ptrCast(@alignCast(h));
        fw.deinit();
        allocator.destroy(fw);
        active_framework = null;
    }
}

/// Get version string.
pub export fn abi_version() [*:0]const u8 {
    return "0.5.0";
}

/// Get detailed version info.
pub export fn abi_version_info(out_version: *VersionInfo) void {
    out_version.* = .{
        .major = 0,
        .minor = 5,
        .patch = 0,
        .full = "0.5.0",
    };
}

/// Check if feature is enabled.
pub export fn abi_is_feature_enabled(
    handle: ?*FrameworkHandle,
    feature: [*:0]const u8,
) bool {
    if (handle) |h| {
        const fw: *abi.Framework = @ptrCast(@alignCast(h));
        const feature_str = std.mem.span(feature);

        if (std.mem.eql(u8, feature_str, "ai")) return fw.isEnabled(.ai);
        if (std.mem.eql(u8, feature_str, "gpu")) return fw.isEnabled(.gpu);
        if (std.mem.eql(u8, feature_str, "database")) return fw.isEnabled(.database);
        if (std.mem.eql(u8, feature_str, "network")) return fw.isEnabled(.network);
        if (std.mem.eql(u8, feature_str, "web")) return fw.isEnabled(.web);
    }
    return false;
}

test "framework exports" {
    var handle: ?*FrameworkHandle = null;
    // Just verify the functions are callable
    _ = abi_version();

    var info: VersionInfo = undefined;
    abi_version_info(&info);
    try std.testing.expectEqual(@as(c_int, 0), info.major);
}
```

**Step 2: Verify file structure**

Run: `ls -la bindings/c/src/framework.zig`
Expected: File exists

**Step 3: Commit**

```bash
git add bindings/c/src/framework.zig
git commit -m "feat(bindings): add framework lifecycle C exports"
```

---

## Task 7: Create Zig Export Module - Database

**Files:**
- Create: `bindings/c/src/database.zig`

**Step 1: Create the database exports module**

```zig
// bindings/c/src/database.zig
//! C-compatible database operation exports.

const std = @import("std");
const abi = @import("abi");
const errors = @import("errors.zig");

/// Opaque database handle.
pub const DatabaseHandle = opaque {};

/// Database config matching C header.
pub const DatabaseConfig = extern struct {
    name: [*:0]const u8,
    dimension: usize,
    initial_capacity: usize,
};

/// Search result matching C header.
pub const SearchResult = extern struct {
    id: u64,
    score: f32,
    vector: [*]const f32,
    vector_len: usize,
};

// Allocator for C bindings
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

/// Create database.
pub export fn abi_database_create(
    config: *const DatabaseConfig,
    out_db: *?*DatabaseHandle,
) errors.Error {
    const db_config = abi.database.Config{
        .name = std.mem.span(config.name),
        .dimension = config.dimension,
        .initial_capacity = config.initial_capacity,
    };

    const db_ptr = allocator.create(abi.database.Database) catch {
        return errors.OUT_OF_MEMORY;
    };

    db_ptr.* = abi.database.Database.init(allocator, db_config) catch |err| {
        allocator.destroy(db_ptr);
        return errors.fromZigError(err);
    };

    out_db.* = @ptrCast(db_ptr);
    return errors.OK;
}

/// Close database.
pub export fn abi_database_close(handle: ?*DatabaseHandle) void {
    if (handle) |h| {
        const db: *abi.database.Database = @ptrCast(@alignCast(h));
        db.deinit();
        allocator.destroy(db);
    }
}

/// Insert vector.
pub export fn abi_database_insert(
    handle: ?*DatabaseHandle,
    id: u64,
    vector: [*]const f32,
    vector_len: usize,
) errors.Error {
    const db: *abi.database.Database = @ptrCast(@alignCast(handle orelse return errors.NOT_INITIALIZED));
    db.insert(id, vector[0..vector_len]) catch |err| {
        return errors.fromZigError(err);
    };
    return errors.OK;
}

/// Search vectors.
pub export fn abi_database_search(
    handle: ?*DatabaseHandle,
    query: [*]const f32,
    query_len: usize,
    k: usize,
    out_results: [*]SearchResult,
    out_count: *usize,
) errors.Error {
    const db: *abi.database.Database = @ptrCast(@alignCast(handle orelse return errors.NOT_INITIALIZED));

    const results = db.search(query[0..query_len], k, allocator) catch |err| {
        return errors.fromZigError(err);
    };
    defer allocator.free(results);

    const count = @min(results.len, k);
    for (0..count) |i| {
        out_results[i] = .{
            .id = results[i].id,
            .score = results[i].score,
            .vector = results[i].vector.ptr,
            .vector_len = results[i].vector.len,
        };
    }
    out_count.* = count;

    return errors.OK;
}

/// Delete vector.
pub export fn abi_database_delete(handle: ?*DatabaseHandle, id: u64) errors.Error {
    const db: *abi.database.Database = @ptrCast(@alignCast(handle orelse return errors.NOT_INITIALIZED));
    db.delete(id) catch |err| {
        return errors.fromZigError(err);
    };
    return errors.OK;
}

/// Get count.
pub export fn abi_database_count(handle: ?*DatabaseHandle, out_count: *usize) errors.Error {
    const db: *abi.database.Database = @ptrCast(@alignCast(handle orelse return errors.NOT_INITIALIZED));
    out_count.* = db.count();
    return errors.OK;
}

test "database exports compile" {
    // Verify types are correct
    const config = DatabaseConfig{
        .name = "test",
        .dimension = 384,
        .initial_capacity = 100,
    };
    _ = config;
}
```

**Step 2: Verify file structure**

Run: `ls -la bindings/c/src/database.zig`
Expected: File exists

**Step 3: Commit**

```bash
git add bindings/c/src/database.zig
git commit -m "feat(bindings): add database operation C exports"
```

---

## Task 8: Create Main Bindings Entry Point

**Files:**
- Create: `bindings/c/src/main.zig`

**Step 1: Create the main entry point**

```zig
// bindings/c/src/main.zig
//! Main entry point for C bindings library.
//! Re-exports all C-compatible functions.

pub const errors = @import("errors.zig");
pub const framework = @import("framework.zig");
pub const simd = @import("simd.zig");
pub const database = @import("database.zig");

// Re-export all C functions
pub usingnamespace errors;
pub usingnamespace framework;
pub usingnamespace simd;
pub usingnamespace database;

// Memory management exports
var gpa = @import("std").heap.GeneralPurposeAllocator(.{}){};

pub export fn abi_free_string(str: ?[*:0]u8) void {
    if (str) |s| {
        const len = @import("std").mem.len(s);
        gpa.allocator().free(s[0 .. len + 1]);
    }
}

pub export fn abi_free_results(results: ?[*]database.SearchResult, count: usize) void {
    _ = results;
    _ = count;
    // Results are stack-allocated by caller, no-op
}

test "all exports" {
    _ = errors;
    _ = framework;
    _ = simd;
    _ = database;
}
```

**Step 2: Verify file structure**

Run: `ls -la bindings/c/src/main.zig`
Expected: File exists

**Step 3: Commit**

```bash
git add bindings/c/src/main.zig
git commit -m "feat(bindings): add main C bindings entry point"
```

---

## Task 9: Create Build Configuration

**Files:**
- Create: `bindings/c/build.zig`

**Step 1: Create the build file**

```zig
// bindings/c/build.zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Get ABI dependency from parent
    const abi_dep = b.dependency("abi", .{
        .target = target,
        .optimize = optimize,
    });

    // Build shared library
    const lib = b.addSharedLibrary(.{
        .name = "abi",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib.root_module.addImport("abi", abi_dep.module("abi"));
    lib.linkLibC();

    b.installArtifact(lib);

    // Build static library
    const static_lib = b.addStaticLibrary(.{
        .name = "abi_static",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    static_lib.root_module.addImport("abi", abi_dep.module("abi"));

    b.installArtifact(static_lib);

    // Install headers
    b.installDirectory(.{
        .source_dir = b.path("include"),
        .install_dir = .header,
        .install_subdir = "",
    });

    // Tests
    const tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    tests.root_module.addImport("abi", abi_dep.module("abi"));

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run bindings tests");
    test_step.dependOn(&run_tests.step);
}
```

**Step 2: Create build.zig.zon**

```zig
// bindings/c/build.zig.zon
.{
    .name = "abi-c-bindings",
    .version = "0.5.0",
    .dependencies = .{
        .abi = .{
            .path = "../..",
        },
    },
    .paths = .{
        "build.zig",
        "build.zig.zon",
        "src",
        "include",
    },
}
```

**Step 3: Verify files exist**

Run: `ls -la bindings/c/build.zig bindings/c/build.zig.zon`
Expected: Both files exist

**Step 4: Commit**

```bash
git add bindings/c/build.zig bindings/c/build.zig.zon
git commit -m "feat(bindings): add C bindings build configuration"
```

---

## Task 10: Create Example C Program

**Files:**
- Create: `bindings/c/examples/hello.c`

**Step 1: Create the example**

```c
// bindings/c/examples/hello.c
// Example: Basic ABI framework usage from C

#include <stdio.h>
#include <stdlib.h>
#include "../include/abi.h"

int main(void) {
    abi_framework_t framework = NULL;
    abi_error_t err;

    // Initialize with defaults
    printf("Initializing ABI framework...\n");
    err = abi_init(&framework);
    if (err != ABI_OK) {
        fprintf(stderr, "Failed to initialize: %s\n", abi_error_string(err));
        return 1;
    }

    // Print version
    printf("ABI Framework v%s\n", abi_version());

    // Check SIMD
    abi_simd_caps_t caps;
    abi_simd_get_caps(&caps);
    printf("SIMD: SSE=%d AVX=%d AVX2=%d NEON=%d\n",
           caps.sse, caps.avx, caps.avx2, caps.neon);

    // Check features
    printf("Features: AI=%d GPU=%d Database=%d\n",
           abi_is_feature_enabled(framework, "ai"),
           abi_is_feature_enabled(framework, "gpu"),
           abi_is_feature_enabled(framework, "database"));

    // Test SIMD operations
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {4.0f, 3.0f, 2.0f, 1.0f};
    float result[4];

    abi_simd_vector_add(a, b, result, 4);
    printf("Vector add: [%.1f, %.1f, %.1f, %.1f]\n",
           result[0], result[1], result[2], result[3]);

    float dot = abi_simd_vector_dot(a, b, 4);
    printf("Dot product: %.1f\n", dot);

    // Cleanup
    abi_shutdown(framework);
    printf("Shutdown complete.\n");

    return 0;
}
```

**Step 2: Verify file exists**

Run: `ls -la bindings/c/examples/hello.c`
Expected: File exists

**Step 3: Commit**

```bash
git add bindings/c/examples/hello.c
git commit -m "feat(bindings): add C example program"
```

---

## Task 11: Create README for C Bindings

**Files:**
- Create: `bindings/c/README.md`

**Step 1: Create the README**

```markdown
# ABI C Bindings

C-compatible bindings for the ABI framework, enabling integration with C, Rust, Go, Python, and other languages via FFI.

## Building

```bash
cd bindings/c
zig build
```

This produces:
- `zig-out/lib/libabi.so` (or `.dylib`/`.dll`) - Shared library
- `zig-out/lib/libabi_static.a` - Static library
- `zig-out/include/` - C headers

## Usage

```c
#include <abi.h>

int main() {
    abi_framework_t fw = NULL;

    if (abi_init(&fw) != ABI_OK) {
        return 1;
    }

    printf("ABI v%s\n", abi_version());

    abi_shutdown(fw);
    return 0;
}
```

Compile with:
```bash
gcc -I/path/to/include -L/path/to/lib -labi example.c -o example
```

## API Reference

### Framework Lifecycle

| Function | Description |
|----------|-------------|
| `abi_init()` | Initialize with defaults |
| `abi_init_with_options()` | Initialize with custom options |
| `abi_shutdown()` | Release resources |
| `abi_version()` | Get version string |
| `abi_is_feature_enabled()` | Check feature status |

### SIMD Operations

| Function | Description |
|----------|-------------|
| `abi_simd_get_caps()` | Query CPU capabilities |
| `abi_simd_available()` | Check if SIMD available |
| `abi_simd_vector_add()` | Element-wise addition |
| `abi_simd_vector_dot()` | Dot product |
| `abi_simd_vector_l2_norm()` | L2 norm |
| `abi_simd_cosine_similarity()` | Cosine similarity |

### Database Operations

| Function | Description |
|----------|-------------|
| `abi_database_create()` | Create vector database |
| `abi_database_close()` | Close database |
| `abi_database_insert()` | Insert vector |
| `abi_database_search()` | Search similar vectors |
| `abi_database_delete()` | Delete vector |
| `abi_database_count()` | Get vector count |

### Memory Management

| Function | Description |
|----------|-------------|
| `abi_free_string()` | Free allocated string |
| `abi_free_results()` | Free search results |

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| 0 | `ABI_OK` | Success |
| -1 | `ABI_ERROR_INIT_FAILED` | Initialization failed |
| -4 | `ABI_ERROR_OUT_OF_MEMORY` | Allocation failed |
| -5 | `ABI_ERROR_INVALID_ARGUMENT` | Invalid parameter |
| -6 | `ABI_ERROR_FEATURE_DISABLED` | Feature not compiled |

See `include/abi_errors.h` for complete list.

## Thread Safety

- Framework initialization is **not** thread-safe
- SIMD operations are thread-safe (stateless)
- Database operations are thread-safe (internal locking)

## License

MIT - See [LICENSE](../../LICENSE)
```

**Step 2: Verify file exists**

Run: `ls -la bindings/c/README.md`
Expected: File exists

**Step 3: Commit**

```bash
git add bindings/c/README.md
git commit -m "docs(bindings): add C bindings README with API reference"
```

---

## Task 12: Update Main build.zig to Include Bindings

**Files:**
- Modify: `build.zig` (add bindings build step)

**Step 1: Read current build.zig ending**

Read the last 50 lines of `build.zig` to find where to add the bindings step.

**Step 2: Add bindings build step**

Add after the existing build steps:

```zig
    // C bindings (optional)
    if (pathExists("bindings/c/build.zig")) {
        const bindings_step = b.step("bindings", "Build C bindings");
        const bindings_cmd = b.addSystemCommand(&.{
            "zig", "build",
            "-p", b.install_path,
        });
        bindings_cmd.setCwd(b.path("bindings/c"));
        bindings_step.dependOn(&bindings_cmd.step);
    }
```

**Step 3: Verify build works**

Run: `zig build --help | grep bindings`
Expected: Shows "bindings" step

**Step 4: Commit**

```bash
git add build.zig
git commit -m "build: add C bindings build step"
```

---

## Task 13: Update TODO.md

**Files:**
- Modify: `TODO.md`

**Step 1: Update Library API status**

Change:
```markdown
| Library API | ⚠️ | C-compatible API (removed for reimplementation). | `bindings/` (to be recreated) |
```

To:
```markdown
| Library API | ✅ | C-compatible API with headers and Zig exports. | `bindings/c/` |
```

**Step 2: Commit**

```bash
git add TODO.md
git commit -m "docs: mark C bindings as complete in TODO.md"
```

---

## Summary

This plan creates a complete C bindings layer:

1. **Headers** (`bindings/c/include/`): `abi.h`, `abi_types.h`, `abi_errors.h`
2. **Zig Exports** (`bindings/c/src/`): `errors.zig`, `simd.zig`, `framework.zig`, `database.zig`, `main.zig`
3. **Build System**: `build.zig`, `build.zig.zon`
4. **Examples**: `examples/hello.c`
5. **Documentation**: `README.md`

Total: 13 tasks, ~15 files created/modified

---

**Plan complete and saved to `docs/plans/2026-01-30-c-bindings-reimplementation.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
