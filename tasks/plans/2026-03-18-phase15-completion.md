# Phase 15 Completion — All 5 Domains Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete all remaining Phase 15 roadmap items (semantic store persistence, mobile native bridges, bare metal cleanup), unblock integration gates, and reconcile stale execution plans.

**Architecture:** Five independent domains executed in parallel. Domain 2 (persistence) wires existing stubbed subsystems to real disk I/O. Domain 3 (mobile) adds C exports and Swift/Kotlin wrapper packages. Domain 4 (integration) unifies test vectors and adds timeout enforcement. Domains 1 and 5 are pure housekeeping edits.

**Tech Stack:** Zig 0.16, C FFI, Swift Package Manager, Kotlin/JNI, POSIX I/O

---

## File Structure

### Domain 1: Housekeeping
- Modify: `tasks/todo.md`

### Domain 2: Semantic Store Persistence
- Modify: `src/core/database/block/segment_log.zig` — real append-only file I/O
- Modify: `src/core/database/block/compression.zig` — simple RLE compression (no external deps)
- Modify: `src/core/database/block/compaction.zig` — real merge/dedupe using codec + segment log
- Modify: `src/core/database/storage.zig` — add HNSW graph block (type 0x03), streaming read via mmap
- Modify: `src/core/database/distributed/wal.zig` — wire file I/O for WAL path
- Modify: `src/core/database/block_chain.zig` — persist MvccStore via segment log

### Domain 3: Mobile Native Bridges
- Modify: `bindings/c/include/abi.h` — add `abi_mobile_*` C API section
- Modify: `bindings/c/src/abi_c.zig` — implement `abi_mobile_*` exports
- Create: `lang/swift/Package.swift` — Swift Package manifest
- Create: `lang/swift/Sources/ABI/ABI.swift` — idiomatic Swift wrapper
- Create: `lang/swift/Sources/CABI/module.modulemap` — C module map for Swift
- Create: `lang/swift/Sources/CABI/shim.h` — umbrella header including abi.h
- Create: `lang/kotlin/src/main/java/com/abi/ABI.kt` — Kotlin wrapper class
- Create: `lang/kotlin/src/main/java/com/abi/ABIJni.kt` — JNI declarations
- Create: `lang/kotlin/jni/abi_jni.c` — JNI glue C code
- Create: `lang/kotlin/build.gradle.kts` — Gradle build

### Domain 4: Integration Gates Unblock
- Modify: `build/cli_smoke_runner.zig` — import vectors from matrix manifest, add timeout enforcement
- Modify: `tests/integration/matrix_manifest.zig` — add export for smoke runner consumption
- Modify: `tests/integration/preflight.zig` — add JSON output mode, Darwin pipeline check
- Modify: `build/cli_tests.zig` — wire `cli-tests-full` step
- Modify: `build.zig` — register `cli-tests-full`
- Modify: `docs/plans/integration-gates-v1.md` — update status to Unblocked

### Domain 5: Stale Plan Cleanup
- Modify: `docs/plans/index.md` — update plan statuses
- Modify: `docs/plans/cli-framework-local-agents.md` — mark completed items
- Modify: `docs/plans/docs-roadmap-sync-v2.md` — mark completed items
- Modify: `docs/plans/feature-modules-restructure-v1.md` — mark completed
- Modify: `docs/plans/gpu-redesign-v3.md` — mark completed items
- Modify: `docs/plans/tui-modular-v2.md` — mark completed items

---

## Domain 1: Housekeeping

### Task 1.1: Reconcile tasks/todo.md

**Files:**
- Modify: `tasks/todo.md`

- [ ] **Step 1: Mark Bare Metal Examples complete**

The bare metal examples already exist as `examples/bare_metal_riscv32.zig` and `examples/bare_metal_thumb.zig`. Change `[ ]` to `[x]` for the "Bare Metal Examples" item and add a note that files already exist.

- [ ] **Step 2: Update Stage 4 status**

Change Stage 4 from "NOT STARTED" to "IN PROGRESS" and update the review date to 2026-03-18.

- [ ] **Step 3: Commit**

```bash
git add -f tasks/todo.md
git commit -m "chore: reconcile todo.md — mark bare metal complete, update stage status"
```

---

## Domain 2: Semantic Store Persistence

### Task 2.1: Wire SegmentLog to Disk I/O

**Files:**
- Modify: `src/core/database/block/segment_log.zig`

- [ ] **Step 1: Write the test**

Add inline `test "SegmentLog append and read back"` at the bottom of `segment_log.zig`:

```zig
test "SegmentLog append and read back" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create temp file path
    var tmp_buf: [256]u8 = undefined;
    const tmp_path = "/tmp/abi_test_segment.log";

    var log = try SegmentLog.init(allocator, tmp_path, 0);
    defer log.deinit();

    // Create a test block
    const test_block = block.StoredBlock{
        .header = .{
            .id = .{ .id = [_]u8{0x01} ** 32 },
            .kind = .data,
            .version = 1,
            .content_hash = [_]u8{0} ** 32,
            .timestamp = .{ .counter = 42 },
            .size = 5,
            .flags = 0,
            .compression_marker = 0,
        },
        .payload = "hello",
    };

    const offset = try log.append(test_block);
    try testing.expectEqual(@as(u64, 0), offset);

    // Read back
    const read_back = try log.readAt(allocator, 0);
    defer allocator.free(read_back.payload);
    try testing.expectEqualSlices(u8, "hello", read_back.payload);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `zig test src/core/database/block/segment_log.zig -fno-emit-bin`
Expected: FAIL — `readAt` doesn't exist, `append` doesn't write

- [ ] **Step 3: Implement real disk I/O**

Replace the stub `append` with actual file writes using the codec:

```zig
const codec = @import("codec.zig");

pub const SegmentLog = struct {
    allocator: std.mem.Allocator,
    path: []const u8,
    current_offset: u64,

    pub fn init(allocator: std.mem.Allocator, path: []const u8, initial_size: u64) !SegmentLog {
        return .{
            .allocator = allocator,
            .path = path,
            .current_offset = initial_size,
        };
    }

    pub fn deinit(self: *SegmentLog) void {
        _ = self;
    }

    pub fn append(self: *SegmentLog, b: block.StoredBlock) !u64 {
        const offset = self.current_offset;
        const encoded = try codec.encodeBlock(self.allocator, b);
        defer self.allocator.free(encoded);

        // Write length prefix (u32 LE) + encoded data
        const len_bytes: [4]u8 = @bitCast(@as(u32, @intCast(encoded.len)));

        // Open file in append mode using POSIX
        const fd = try std.posix.open(
            self.path,
            .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true },
            0o644,
        );
        defer std.posix.close(fd);

        _ = try std.posix.write(fd, &len_bytes);
        _ = try std.posix.write(fd, encoded);

        self.current_offset += 4 + encoded.len;
        return offset;
    }

    pub fn readAt(self: *SegmentLog, allocator: std.mem.Allocator, offset: u64) !block.StoredBlock {
        const fd = try std.posix.open(self.path, .{ .ACCMODE = .RDONLY }, 0);
        defer std.posix.close(fd);

        // Seek to offset
        _ = try std.posix.lseek(fd, @intCast(offset), .SET);

        // Read length prefix
        var len_buf: [4]u8 = undefined;
        const len_read = try std.posix.read(fd, &len_buf);
        if (len_read < 4) return error.UnexpectedEof;

        const data_len = std.mem.readInt(u32, &len_buf, .little);
        const data = try allocator.alloc(u8, data_len);
        errdefer allocator.free(data);

        const data_read = try std.posix.read(fd, data);
        if (data_read < data_len) return error.UnexpectedEof;

        // Decode — codec.decodeBlock dupes the payload so we can free data
        var decoded = try codec.decodeBlock(allocator, data);
        allocator.free(data);
        return decoded;
    }
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `zig test src/core/database/block/segment_log.zig -fno-emit-bin`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/database/block/segment_log.zig
git commit -m "feat: wire SegmentLog to real disk I/O with length-prefixed codec"
```

### Task 2.2: Implement Simple Compression

**Files:**
- Modify: `src/core/database/block/compression.zig`

- [ ] **Step 1: Write test for RLE compression**

```zig
test "RLE round-trip" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const input = "aaabbccccdddd";
    const compressed = try compress(alloc, input, .lz4); // repurpose lz4 as RLE
    defer alloc.free(compressed);
    const decompressed = try decompress(alloc, compressed, .lz4);
    defer alloc.free(decompressed);
    try testing.expectEqualSlices(u8, input, decompressed);
}
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement simple RLE compression**

Replace `.lz4 => error.NotImplemented` with a simple run-length encoder (no external deps). RLE is appropriate for vector data with repeated zero padding and metadata blocks.

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/core/database/block/compression.zig
git commit -m "feat: implement RLE compression for block storage"
```

### Task 2.3: Add HNSW Graph Block to Storage v2

**Files:**
- Modify: `src/core/database/storage.zig`

- [ ] **Step 1: Write test for HNSW block serialization round-trip**

Add test that creates an HNSW index, saves with `saveDatabase` (which should now write a vector_index block), loads back, and verifies the graph structure.

- [ ] **Step 2: Run test — expect FAIL** (vector_index block not written)

- [ ] **Step 3: Implement `writeHnswBlock` in storage.zig**

Add a new function that serializes the HNSW graph into a `BlockType.vector_index` block:
- Entry point node ID (u32)
- Max layer (u32)
- Per node: layer count (u32), per layer: neighbor count (u32) + neighbor IDs ([]u32)

Wire it into `saveDatabase` after `writeVectorBlocks`, gated by a new `has_index` flag.

- [ ] **Step 4: Implement `readHnswBlock` in the load path**

When `BlockType.vector_index` is encountered during load, deserialize the graph and attach to a reconstructed HNSW index.

- [ ] **Step 5: Run test — expect PASS**

- [ ] **Step 6: Commit**

```bash
git add src/core/database/storage.zig
git commit -m "feat: persist HNSW graph as vector_index block in storage v2"
```

### Task 2.4: Wire WAL to File I/O

**Files:**
- Modify: `src/core/database/distributed/wal.zig`

- [ ] **Step 1: Write test for WAL file persistence**

```zig
test "WAL write and recover from file" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var wal = WalWriter.init(alloc, "/tmp/abi_test.wal", 1);
    defer wal.deinit();

    try wal.appendInsert(42, 128, &[_]u8{1, 2, 3, 4});
    try wal.flush(); // new method

    // Read back
    var wal2 = try WalWriter.recover(alloc, "/tmp/abi_test.wal");
    defer wal2.deinit();

    try testing.expectEqual(@as(usize, 1), wal2.entries.items.len);
    try testing.expectEqual(@as(u64, 42), wal2.entries.items[0].vector_id);
}
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement `flush()` and `recover()`**

`flush()`: serialize WAL header + all entries + data buffer to the file at `self.path`.
`recover()`: read the file, deserialize header, replay entries into a new WalWriter.

Use the existing `serialize`/`deserialize` methods which already work correctly.

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/core/database/distributed/wal.zig
git commit -m "feat: wire WAL to disk with flush/recover for crash recovery"
```

### Task 2.5: Implement Real Compaction

**Files:**
- Modify: `src/core/database/block/compaction.zig`

- [ ] **Step 1: Write test for merge compaction**

Test that creating multiple blocks in a segment log, then merging, produces a single consolidated block.

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement real merge and dedupe**

`merge()`: read blocks from segment log via `readAt`, combine payloads, write a new consolidated block.
`dedupe()`: read blocks, skip those with duplicate `content_hash`, write survivors.
`rewrite()`: read all blocks from old segment, write to new segment, swap files.

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/core/database/block/compaction.zig
git commit -m "feat: implement real merge/dedupe/rewrite compaction"
```

### Task 2.6: Persist BlockChain via SegmentLog

**Files:**
- Modify: `src/core/database/block_chain.zig`

- [ ] **Step 1: Add `save` and `load` methods to MvccStore**

Wire through to SegmentLog for each ConversationBlock. Serialize embedding vectors and metadata using the block codec.

- [ ] **Step 2: Write test for MvccStore round-trip**

- [ ] **Step 3: Run test — expect PASS**

- [ ] **Step 4: Commit**

```bash
git add src/core/database/block_chain.zig
git commit -m "feat: persist MvccStore conversation blocks via SegmentLog"
```

---

## Domain 3: Mobile Native Bridges

### Task 3.1: Add Mobile C Exports

**Files:**
- Modify: `bindings/c/include/abi.h`
- Modify: `bindings/c/src/abi_c.zig`

- [ ] **Step 1: Add C type definitions to abi.h**

```c
/* Mobile */
typedef struct abi_mobile_context abi_mobile_context_t;

typedef struct {
    uint32_t screen_width;
    uint32_t screen_height;
    float battery_level;
    bool is_charging;
    const char *platform;
    const char *os_version;
    const char *device_model;
} abi_device_info_t;

typedef struct {
    uint64_t timestamp_ms;
    float values[3];
} abi_sensor_data_t;

#define ABI_SENSOR_ACCELEROMETER  0
#define ABI_SENSOR_GYROSCOPE      1
#define ABI_SENSOR_MAGNETOMETER   2
#define ABI_SENSOR_GPS            3

#define ABI_PERMISSION_CAMERA     0
#define ABI_PERMISSION_LOCATION   2
#define ABI_PERMISSION_NOTIFICATIONS 3

#define ABI_PERM_GRANTED       0
#define ABI_PERM_DENIED        1
#define ABI_PERM_NOT_REQUESTED 2

int abi_mobile_init(abi_mobile_context_t **ctx);
void abi_mobile_destroy(abi_mobile_context_t *ctx);
int abi_mobile_read_sensor(abi_mobile_context_t *ctx, int sensor_type, abi_sensor_data_t *out);
int abi_mobile_send_notification(abi_mobile_context_t *ctx, const char *title, const char *body, int priority);
int abi_mobile_get_device_info(abi_mobile_context_t *ctx, abi_device_info_t *out);
int abi_mobile_check_permission(abi_mobile_context_t *ctx, int permission);
int abi_mobile_request_permission(abi_mobile_context_t *ctx, int permission);
```

- [ ] **Step 2: Implement exports in abi_c.zig**

Follow the existing opaque-handle pattern (MobileWrapper holding Context + allocator).

- [ ] **Step 3: Verify compilation**

Run: `zig fmt --check bindings/c/src/abi_c.zig`
Run: `./tools/scripts/run_build.sh typecheck --summary all`

- [ ] **Step 4: Commit**

```bash
git add bindings/c/include/abi.h bindings/c/src/abi_c.zig
git commit -m "feat: add abi_mobile_* C exports for mobile context, sensors, notifications"
```

### Task 3.2: Create Swift Package

**Files:**
- Create: `lang/swift/Package.swift`
- Create: `lang/swift/Sources/CABI/module.modulemap`
- Create: `lang/swift/Sources/CABI/shim.h`
- Create: `lang/swift/Sources/ABI/ABI.swift`
- Create: `lang/swift/Sources/ABI/Mobile.swift`

- [ ] **Step 1: Create Package.swift**

```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ABI",
    platforms: [.iOS(.v15), .macOS(.v13)],
    products: [
        .library(name: "ABI", targets: ["ABI"]),
    ],
    targets: [
        .systemLibrary(
            name: "CABI",
            path: "Sources/CABI"
        ),
        .target(
            name: "ABI",
            dependencies: ["CABI"],
            path: "Sources/ABI"
        ),
    ]
)
```

- [ ] **Step 2: Create module.modulemap and shim.h**

`module.modulemap`:
```
module CABI {
    umbrella header "shim.h"
    export *
}
```

`shim.h`:
```c
#include "../../../bindings/c/include/abi.h"
```

- [ ] **Step 3: Create ABI.swift — core framework wrapper**

```swift
import CABI

public class ABIFramework {
    private var handle: OpaquePointer?

    public init() throws {
        var ptr: OpaquePointer?
        let err = abi_init(&ptr)
        guard err == ABI_OK else { throw ABIError(code: err) }
        self.handle = ptr
    }

    deinit { if let h = handle { abi_shutdown(h) } }

    public var version: String {
        String(cString: abi_version())
    }
}

public struct ABIError: Error {
    public let code: Int32
    public var description: String {
        String(cString: abi_error_string(code))
    }
}
```

- [ ] **Step 4: Create Mobile.swift — idiomatic mobile wrapper**

```swift
import CABI

public class MobileContext {
    private var handle: OpaquePointer?

    public init() throws {
        var ptr: OpaquePointer?
        let err = abi_mobile_init(&ptr)
        guard err == ABI_OK else { throw ABIError(code: err) }
        self.handle = ptr
    }

    deinit { if let h = handle { abi_mobile_destroy(h) } }

    public struct SensorData {
        public let timestampMs: UInt64
        public let values: (Float, Float, Float)
    }

    public enum SensorType: Int32 {
        case accelerometer = 0
        case gyroscope = 1
        case magnetometer = 2
        case gps = 3
    }

    public func readSensor(_ type: SensorType) throws -> SensorData {
        var data = abi_sensor_data_t()
        let err = abi_mobile_read_sensor(handle, type.rawValue, &data)
        guard err == ABI_OK else { throw ABIError(code: err) }
        return SensorData(
            timestampMs: data.timestamp_ms,
            values: (data.values.0, data.values.1, data.values.2)
        )
    }

    public func sendNotification(title: String, body: String) throws {
        let err = abi_mobile_send_notification(handle, title, body, 1)
        guard err == ABI_OK else { throw ABIError(code: err) }
    }

    public struct DeviceInfo {
        public let screenWidth: UInt32
        public let screenHeight: UInt32
        public let batteryLevel: Float
        public let isCharging: Bool
        public let platform: String
        public let osVersion: String
        public let deviceModel: String
    }

    public func getDeviceInfo() throws -> DeviceInfo {
        var info = abi_device_info_t()
        let err = abi_mobile_get_device_info(handle, &info)
        guard err == ABI_OK else { throw ABIError(code: err) }
        return DeviceInfo(
            screenWidth: info.screen_width,
            screenHeight: info.screen_height,
            batteryLevel: info.battery_level,
            isCharging: info.is_charging,
            platform: String(cString: info.platform),
            osVersion: String(cString: info.os_version),
            deviceModel: String(cString: info.device_model)
        )
    }
}
```

- [ ] **Step 5: Commit**

```bash
git add lang/swift/
git commit -m "feat: add Swift package with idiomatic MobileContext wrapper"
```

### Task 3.3: Create Kotlin/JNI Bridge

**Files:**
- Create: `lang/kotlin/build.gradle.kts`
- Create: `lang/kotlin/src/main/java/com/abi/ABI.kt`
- Create: `lang/kotlin/src/main/java/com/abi/Mobile.kt`
- Create: `lang/kotlin/jni/abi_jni.c`

- [ ] **Step 1: Create build.gradle.kts**

```kotlin
plugins {
    id("com.android.library") version "8.2.0"
    kotlin("android") version "1.9.22"
}

android {
    namespace = "com.abi"
    compileSdk = 34
    defaultConfig { minSdk = 26 }

    externalNativeBuild {
        ndkBuild { path = file("jni/Android.mk") }
    }
}
```

- [ ] **Step 2: Create Kotlin wrapper classes**

`ABI.kt`:
```kotlin
package com.abi

object ABI {
    init { System.loadLibrary("abi_jni") }
    external fun version(): String
    external fun init(): Long  // returns handle
    external fun shutdown(handle: Long)
}
```

`Mobile.kt`:
```kotlin
package com.abi

data class SensorData(val timestampMs: Long, val x: Float, val y: Float, val z: Float)
data class DeviceInfo(
    val screenWidth: Int, val screenHeight: Int,
    val batteryLevel: Float, val isCharging: Boolean,
    val platform: String, val osVersion: String, val deviceModel: String
)

class MobileContext : AutoCloseable {
    private val handle: Long = nativeInit()

    fun readSensor(type: Int): SensorData = nativeReadSensor(handle, type)
    fun sendNotification(title: String, body: String, priority: Int = 1) =
        nativeSendNotification(handle, title, body, priority)
    fun getDeviceInfo(): DeviceInfo = nativeGetDeviceInfo(handle)

    override fun close() = nativeDestroy(handle)

    private external fun nativeInit(): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeReadSensor(handle: Long, type: Int): SensorData
    private external fun nativeSendNotification(handle: Long, title: String, body: String, priority: Int)
    private external fun nativeGetDeviceInfo(handle: Long): DeviceInfo

    companion object { init { System.loadLibrary("abi_jni") } }
}
```

- [ ] **Step 3: Create JNI glue C code**

`abi_jni.c` — bridges JNI calls to `abi_mobile_*` C functions. Uses `GetStringUTFChars`/`NewStringUTF` for string conversion, creates Java objects via `FindClass`/`NewObject` for return types.

- [ ] **Step 4: Commit**

```bash
git add lang/kotlin/
git commit -m "feat: add Kotlin/JNI bridge with MobileContext wrapper"
```

### Task 3.4: Update lang/README.md

**Files:**
- Modify: `lang/README.md`

- [ ] **Step 1: Update README with Swift and Kotlin sections**

Document build instructions, usage examples, and architecture for both bridges.

- [ ] **Step 2: Commit**

```bash
git add -f lang/README.md
git commit -m "docs: update lang/README.md with Swift and Kotlin bridge documentation"
```

---

## Domain 4: Integration Gates Unblock

### Task 4.1: Export Matrix Manifest for Smoke Runner

**Files:**
- Modify: `tests/integration/matrix_manifest.zig`

- [ ] **Step 1: Add comptime export of safe vectors**

Add a public function `safeVectors()` that filters vectors safe for non-interactive execution (excludes TUI vectors that need a terminal):

```zig
pub fn safeVectors() []const IntegrationVector {
    comptime {
        var count: usize = 0;
        for (all_vectors) |v| {
            if (v.timeout != .tui) count += 1;
        }
        var result: [count]IntegrationVector = undefined;
        var i: usize = 0;
        for (all_vectors) |v| {
            if (v.timeout != .tui) {
                result[i] = v;
                i += 1;
            }
        }
        return &result;
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add tests/integration/matrix_manifest.zig
git commit -m "feat: export safe vectors from matrix manifest for smoke runner"
```

### Task 4.2: Unify Smoke Runner with Matrix Manifest

**Files:**
- Modify: `build/cli_smoke_runner.zig`

- [ ] **Step 1: Replace hardcoded vectors with manifest import**

Remove the `safe_function_vectors` array. Import from the matrix manifest module (requires build.zig to add it as a named import). Fall back to generating help vectors from CLI descriptors as before.

- [ ] **Step 2: Add timeout enforcement**

For each vector, use a deadline based on `TimeoutTier.toNs()`:
- Spawn child process
- Wait with timeout (use `std.posix.waitpid` + `WNOHANG` in a loop with `nanosleep`)
- If deadline exceeded, send `SIGKILL` and report timeout

- [ ] **Step 3: Verify compilation**

Run: `./tools/scripts/run_build.sh cli-tests --summary all`

- [ ] **Step 4: Commit**

```bash
git add build/cli_smoke_runner.zig
git commit -m "feat: unify smoke runner with matrix manifest, add timeout enforcement"
```

### Task 4.3: Wire cli-tests-full Build Step

**Files:**
- Modify: `build.zig`
- Modify: `build/cli_tests.zig`

- [ ] **Step 1: Add cli-tests-full step**

In `build/cli_tests.zig`, add a new public function `addFullCliTests()` that:
- Compiles the smoke runner with the matrix manifest module
- Runs all vectors (including feature-gated ones based on current build flags)
- On blocked Darwin, falls back to typecheck

- [ ] **Step 2: Register in build.zig**

```zig
const cli_tests_full = cli_tests.addFullCliTests(b, darwin_ctx, abi_module, ...);
const cli_tests_full_step = b.step("cli-tests-full", "Full CLI integration gate with manifest vectors");
cli_tests_full_step.dependOn(cli_tests_full);
```

- [ ] **Step 3: Add to full-check dependency**

Wire `cli-tests-full` as a dependency of `full-check`.

- [ ] **Step 4: Commit**

```bash
git add build.zig build/cli_tests.zig
git commit -m "feat: add cli-tests-full build step with matrix manifest integration"
```

### Task 4.4: Enhance Preflight Diagnostics

**Files:**
- Modify: `tests/integration/preflight.zig`

- [ ] **Step 1: Add JSON output mode**

Add `--json` flag that outputs a machine-readable JSON report instead of the human-readable table.

- [ ] **Step 2: Add Darwin pipeline check**

Add a check for `run_build.sh` availability and whether `darwinRelink` is functional.

- [ ] **Step 3: Add "blocked" vs "reduced" distinction**

Encode the distinction in exit codes: 0 = all ok, 1 = blocked (can't proceed), 2 = degraded (can proceed with reduced coverage).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/preflight.zig
git commit -m "feat: enhance preflight with JSON output, Darwin pipeline check, exit codes"
```

### Task 4.5: Update Integration Gates Plan Status

**Files:**
- Modify: `docs/plans/integration-gates-v1.md`

- [ ] **Step 1: Update criteria status**

Mark Criterion A (matrix + timeout) as complete.
Mark Criterion B (preflight diagnostics) as complete.
Mark Criterion C (environment docs) as complete (already was).
Change plan status from "Blocked" to "Complete".

- [ ] **Step 2: Commit**

```bash
git add -f docs/plans/integration-gates-v1.md
git commit -m "docs: mark integration gates v1 plan as complete — all criteria met"
```

---

## Domain 5: Stale Plan Cleanup

### Task 5.1: Update All Execution Plan Statuses

**Files:**
- Modify: `docs/plans/index.md`
- Modify: `docs/plans/cli-framework-local-agents.md`
- Modify: `docs/plans/docs-roadmap-sync-v2.md`
- Modify: `docs/plans/feature-modules-restructure-v1.md`
- Modify: `docs/plans/gpu-redesign-v3.md`
- Modify: `docs/plans/tui-modular-v2.md`

- [ ] **Step 1: Read each plan and identify completed milestones**

Cross-reference each plan's milestones against git history and current codebase state. The 18-phase restructuring (commit 04002b79) completed most of the work these plans describe.

- [ ] **Step 2: Update individual plan files**

For each plan:
- Mark completed milestones with checkmarks
- Add "Superseded by 18-phase restructuring (2026-03-17)" note where applicable
- Update status field

- [ ] **Step 3: Update index.md**

Change "In Progress" to "Complete" or "Superseded" for plans whose work is done.

- [ ] **Step 4: Commit**

```bash
git add -f docs/plans/*.md
git commit -m "docs: update execution plans — mark 5 plans as complete/superseded"
```

---

## Verification

After all domains are complete:

- [ ] **Run format check:** `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/`
- [ ] **Run full-check:** `./tools/scripts/run_build.sh full-check --summary all`
- [ ] **Verify mod/stub parity:** All changed features still pass `check-stub-parity`
- [ ] **Final commit:** Single integration commit if any loose ends

---

## Execution Order

All 5 domains are independent and can execute in parallel:

| Domain | Est. Complexity | Key Risk |
|--------|----------------|----------|
| 1: Housekeeping | Low | None |
| 2: Semantic Store | High | POSIX I/O compatibility with Zig 0.16 |
| 3: Mobile Bridges | Medium | Swift/Kotlin files can't be compiled without SDKs |
| 4: Integration Gates | Medium | Build system wiring complexity |
| 5: Plan Cleanup | Low | None |

Domains 1 and 5 should complete first (pure edits). Domains 2, 3, 4 run in parallel as the heavy lifts.
