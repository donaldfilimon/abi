//! Common test helpers and utilities.
//!
//! This module provides shared setup/teardown logic and assertion
//! helpers used across the test suite.

const std = @import("std");
const abi = @import("abi");
const time = abi.shared.time;

// ============================================================================
// Time Utilities
// ============================================================================

/// Sleep for a specified number of milliseconds.
/// On WASM, this is a no-op (can't block in WASM).
/// Re-exports from shared/time.zig for test convenience.
pub const sleepMs = time.sleepMs;

/// Sleep for a specified number of nanoseconds.
pub const sleepNs = time.sleepNs;

/// Test allocator with leak detection.
/// Wraps GeneralPurposeAllocator with automatic leak checking on deinit.
pub const TestAllocator = struct {
    gpa: std.heap.GeneralPurposeAllocator(.{
        .stack_trace_frames = 10,
    }),

    pub fn init() TestAllocator {
        return .{ .gpa = .{} };
    }

    pub fn allocator(self: *TestAllocator) std.mem.Allocator {
        return self.gpa.allocator();
    }

    /// Deinitializes and checks for memory leaks.
    /// Panics if leaks are detected.
    pub fn deinit(self: *TestAllocator) void {
        const check = self.gpa.deinit();
        if (check == .leak) {
            @panic("Memory leak detected in test");
        }
    }
};

/// Skip test if GPU hardware is not available.
/// Use at the start of GPU-dependent tests.
pub fn skipIfNoGpu() error{SkipZigTest}!void {
    if (!hasGpuSupport()) {
        return error.SkipZigTest;
    }
}

fn hasGpuSupport() bool {
    const builtin = @import("builtin");
    // Skip GPU tests on WASM and freestanding
    return builtin.os.tag != .freestanding and builtin.cpu.arch != .wasm32;
}

/// Skip test if timer is not available on this platform.
pub fn skipIfNoTimer() error{SkipZigTest}!void {
    _ = time.Timer.start() catch return error.SkipZigTest;
}

// ============================================================================
// Vector Test Utilities
// ============================================================================

/// Generate a random vector for testing.
/// Fills the provided buffer with random values in the range [-1, 1].
pub fn generateRandomVector(rng: *std.Random.DefaultPrng, buffer: []f32) void {
    for (buffer) |*v| {
        v.* = rng.random().float(f32) * 2.0 - 1.0;
    }
}

/// Generate a random vector with allocation.
/// Returns a newly allocated slice of random values in [-1, 1].
pub fn generateRandomVectorAlloc(
    allocator: std.mem.Allocator,
    rng: *std.Random.DefaultPrng,
    dims: usize,
) ![]f32 {
    const vec = try allocator.alloc(f32, dims);
    generateRandomVector(rng, vec);
    return vec;
}

/// Normalize a vector to unit length.
pub fn normalizeVector(vec: []f32) void {
    var sum: f32 = 0;
    for (vec) |v| {
        sum += v * v;
    }
    const norm = @sqrt(sum);
    if (norm > 0) {
        for (vec) |*v| {
            v.* /= norm;
        }
    }
}

test "generateRandomVector produces valid values" {
    var rng = std.Random.DefaultPrng.init(42);
    var buffer: [128]f32 = undefined;
    generateRandomVector(&rng, &buffer);

    for (buffer) |v| {
        try std.testing.expect(v >= -1.0 and v <= 1.0);
    }
}

test "normalizeVector produces unit vector" {
    var vec = [_]f32{ 3.0, 4.0 };
    normalizeVector(&vec);

    var sum_sq: f32 = 0;
    for (vec) |v| {
        sum_sq += v * v;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum_sq, 0.0001);
}

// ============================================================================
// Temporary Directory Utilities
// ============================================================================

/// Creates a temporary directory for test files.
/// Returns the path to the created directory.
/// Caller is responsible for cleaning up the directory after use.
pub fn createTempDir(allocator: std.mem.Allocator) ![]const u8 {
    const builtin = @import("builtin");

    // Use platform-appropriate temp directory
    const base_dir = switch (builtin.os.tag) {
        .windows => blk: {
            // On Windows, use TEMP environment variable or fallback
            var env_map = std.process.Environ.createMap(std.process.Environ.empty, allocator) catch {
                break :blk try allocator.dupe(u8, "C:\\Temp");
            };
            defer env_map.deinit();
            if (env_map.get("TEMP")) |temp_val| {
                break :blk try allocator.dupe(u8, temp_val);
            }
            break :blk try allocator.dupe(u8, "C:\\Temp");
        },
        else => blk: {
            // On Unix-like systems, use TMPDIR or /tmp
            var env_map = std.process.Environ.createMap(std.process.Environ.empty, allocator) catch {
                break :blk try allocator.dupe(u8, "/tmp");
            };
            defer env_map.deinit();
            if (env_map.get("TMPDIR")) |tmpdir_val| {
                break :blk try allocator.dupe(u8, tmpdir_val);
            }
            break :blk try allocator.dupe(u8, "/tmp");
        },
    };
    defer allocator.free(base_dir);

    // Generate a unique directory name using absolute timestamp as seed
    // getSeed() uses timestampNs() on native, giving monotonically
    // increasing absolute time — unlike Timer.read() which gives ~0ns
    const seed: u64 = time.getSeed();
    var rng = std.Random.DefaultPrng.init(seed);
    const random_suffix = rng.random().int(u32);
    const random_suffix2 = rng.random().int(u32);

    // Format: abi-test-<random>-<random2>
    const dir_name = try std.fmt.allocPrint(
        allocator,
        "{s}{c}abi-test-{x}-{x}",
        .{ base_dir, std.fs.path.sep, random_suffix, random_suffix2 },
    );
    errdefer allocator.free(dir_name);

    // Create the directory using Zig 0.16 IO API
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    // Get the parent directory and create the temp dir
    const parent_path = std.fs.path.dirname(dir_name) orelse "/tmp";
    var parent_dir = std.Io.Dir.cwd().openDir(io, parent_path, .{}) catch |err| switch (err) {
        error.FileNotFound => return err,
        else => return err,
    };
    defer parent_dir.close(io);

    const basename = std.fs.path.basename(dir_name);
    parent_dir.createDir(io, basename, .default_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {}, // Already exists, that's fine
        else => return err,
    };

    return dir_name;
}

/// Removes a temporary directory and all its contents.
/// Use this to clean up directories created by createTempDir.
pub fn removeTempDir(allocator: std.mem.Allocator, path: []const u8) void {
    // Initialize I/O backend for file operations
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    // Get parent directory and delete the temp dir
    const parent_path = std.fs.path.dirname(path) orelse "/tmp";
    var parent_dir = std.Io.Dir.cwd().openDir(io, parent_path, .{}) catch return;
    defer parent_dir.close(io);

    const basename = std.fs.path.basename(path);
    parent_dir.deleteTree(io, basename) catch {};
    allocator.free(path);
}

/// Scoped temporary directory that cleans up automatically.
/// Use with defer for automatic cleanup.
pub const TempDir = struct {
    allocator: std.mem.Allocator,
    path: []const u8,

    pub fn init(allocator: std.mem.Allocator) !TempDir {
        const path = try createTempDir(allocator);
        return .{
            .allocator = allocator,
            .path = path,
        };
    }

    pub fn deinit(self: *TempDir) void {
        removeTempDir(self.allocator, self.path);
        self.* = undefined;
    }

    /// Get the directory path.
    pub fn getPath(self: *const TempDir) []const u8 {
        return self.path;
    }
};

test "TestAllocator detects leaks" {
    // This test verifies the allocator works - actual leak detection
    // would panic, so we just verify basic allocation/free works
    var ta = TestAllocator.init();
    defer ta.deinit();

    const alloc = ta.allocator();
    const slice = try alloc.alloc(u8, 100);
    alloc.free(slice);
}

test "createTempDir creates unique directory" {
    const allocator = std.testing.allocator;

    const dir1 = createTempDir(allocator) catch |err| switch (err) {
        error.PermissionDenied => return error.SkipZigTest, // sandbox restriction
        else => return err,
    };
    defer removeTempDir(allocator, dir1);

    const dir2 = createTempDir(allocator) catch |err| switch (err) {
        error.PermissionDenied => return error.SkipZigTest,
        else => return err,
    };
    defer removeTempDir(allocator, dir2);

    // Directories should be different
    try std.testing.expect(!std.mem.eql(u8, dir1, dir2));

    // Initialize I/O backend for directory operations
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    // Directories should exist
    var d1 = std.Io.Dir.cwd().openDir(io, dir1, .{}) catch |err| {
        std.debug.print("Failed to open dir1: {t}\n", .{err});
        return err;
    };
    d1.close(io);

    var d2 = std.Io.Dir.cwd().openDir(io, dir2, .{}) catch |err| {
        std.debug.print("Failed to open dir2: {t}\n", .{err});
        return err;
    };
    d2.close(io);
}

test "TempDir scoped cleanup" {
    const allocator = std.testing.allocator;

    // Initialize I/O backend for directory operations
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    var path_copy: []const u8 = undefined;

    {
        var temp = TempDir.init(allocator) catch |err| switch (err) {
            error.PermissionDenied => return error.SkipZigTest, // sandbox restriction
            else => return err,
        };
        defer temp.deinit();

        // Save path for verification after cleanup
        path_copy = try allocator.dupe(u8, temp.getPath());

        // Directory should exist while in scope
        var dir = try std.Io.Dir.cwd().openDir(io, temp.getPath(), .{});
        dir.close(io);
    }
    defer allocator.free(path_copy);

    // After deinit, directory should be gone
    const result = std.Io.Dir.cwd().openDir(io, path_copy, .{});
    try std.testing.expectError(error.FileNotFound, result);
}
