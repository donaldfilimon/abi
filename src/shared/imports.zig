//! Centralized Imports Module
//!
//! Provides standardized imports and type aliases to reduce duplication
//! and ensure consistency across the codebase.

// Standard library imports
pub const std = @import("std");
pub const builtin = @import("builtin");

// Common type aliases
pub const Allocator = std.mem.Allocator;
pub const ArrayList = std.ArrayList;
pub const HashMap = std.HashMap;
pub const AutoHashMap = std.AutoHashMap;
pub const StringHashMap = std.StringHashMap;
pub const Writer = std.io.AnyWriter;
pub const Reader = std.io.AnyReader;

// Testing utilities
pub const testing = std.testing;
pub const expect = testing.expect;
pub const expectEqual = testing.expectEqual;
pub const expectEqualStrings = testing.expectEqualStrings;
pub const expectError = testing.expectError;

// Memory utilities
pub const page_allocator = std.heap.page_allocator;
pub const GeneralPurposeAllocator = std.heap.GeneralPurposeAllocator;
pub const ArenaAllocator = std.heap.ArenaAllocator;
pub const FixedBufferAllocator = std.heap.FixedBufferAllocator;

// Common constants
pub const KB = 1024;
pub const MB = 1024 * KB;
pub const GB = 1024 * MB;

// Platform detection helpers
pub const Platform = enum {
    windows,
    linux,
    macos,
    wasm,
    other,
    
    pub fn current() Platform {
        return switch (builtin.os.tag) {
            .windows => .windows,
            .linux => .linux,
            .macos => .macos,
            .wasi => .wasm,
            else => .other,
        };
    }
    
    pub fn isUnix(self: Platform) bool {
        return self == .linux or self == .macos;
    }
    
    pub fn supportsGPU(self: Platform) bool {
        return switch (self) {
            .windows, .linux, .macos => true,
            .wasm => true, // WebGPU
            .other => false,
        };
    }
};

// Framework-specific imports
pub const framework = @import("../framework/mod.zig");
pub const features = @import("../features/mod.zig");
pub const shared = @import("../shared/mod.zig");
pub const patterns = @import("patterns/common.zig");

// Feature modules
pub const ai = features.ai;
pub const gpu = features.gpu;
pub const database = features.database;
pub const web = features.web;
pub const monitoring = features.monitoring;

// Shared utilities
pub const Logger = patterns.Logger;
pub const ErrorContext = patterns.ErrorContext;

// Math utilities
pub const math = struct {
    pub const pi = std.math.pi;
    pub const e = std.math.e;
    pub const inf = std.math.inf;
    pub const nan = std.math.nan;
    
    pub fn clamp(comptime T: type, value: T, min_val: T, max_val: T) T {
        return std.math.clamp(value, min_val, max_val);
    }
    
    pub fn lerp(comptime T: type, a: T, b: T, t: T) T {
        return a + (b - a) * t;
    }
    
    pub fn smoothstep(comptime T: type, edge0: T, edge1: T, x: T) T {
        const t = clamp(T, (x - edge0) / (edge1 - edge0), 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    }
};

// Time utilities
pub const time = struct {
    pub const ns_per_us = std.time.ns_per_us;
    pub const ns_per_ms = std.time.ns_per_ms;
    pub const ns_per_s = std.time.ns_per_s;
    pub const us_per_ms = std.time.us_per_ms;
    pub const us_per_s = std.time.us_per_s;
    pub const ms_per_s = std.time.ms_per_s;
    
    pub fn timestamp() i64 {
        return std.time.milliTimestamp();
    }
    
    pub fn nanoTimestamp() u64 {
        return std.time.nanoTimestamp();
    }
};

// String utilities
pub const string = struct {
    pub fn eql(a: []const u8, b: []const u8) bool {
        return std.mem.eql(u8, a, b);
    }
    
    pub fn startsWith(haystack: []const u8, needle: []const u8) bool {
        return std.mem.startsWith(u8, haystack, needle);
    }
    
    pub fn endsWith(haystack: []const u8, needle: []const u8) bool {
        return std.mem.endsWith(u8, haystack, needle);
    }
    
    pub fn indexOf(haystack: []const u8, needle: []const u8) ?usize {
        return std.mem.indexOf(u8, haystack, needle);
    }
    
    pub fn trim(s: []const u8) []const u8 {
        return std.mem.trim(u8, s, " \t\r\n");
    }
    
    pub fn split(allocator: Allocator, s: []const u8, delimiter: []const u8) ![][]const u8 {
        var result = ArrayList([]const u8).init(allocator);
        var iter = std.mem.split(u8, s, delimiter);
        while (iter.next()) |part| {
            try result.append(part);
        }
        return result.toOwnedSlice();
    }
};

// File system utilities
pub const fs = struct {
    pub fn readFileAlloc(allocator: Allocator, path: []const u8) ![]u8 {
        return try std.fs.cwd().readFileAlloc(allocator, path, std.math.maxInt(usize));
    }
    
    pub fn writeFile(path: []const u8, data: []const u8) !void {
        try std.fs.cwd().writeFile(path, data);
    }
    
    pub fn fileExists(path: []const u8) bool {
        std.fs.cwd().access(path, .{}) catch return false;
        return true;
    }
    
    pub fn createDir(path: []const u8) !void {
        try std.fs.cwd().makeDir(path);
    }
    
    pub fn createDirRecursive(path: []const u8) !void {
        try std.fs.cwd().makePath(path);
    }
};

// JSON utilities
pub const json = struct {
    pub fn parseFromSlice(comptime T: type, allocator: Allocator, source: []const u8) !T {
        return try std.json.parseFromSlice(T, allocator, source, .{});
    }
    
    pub fn stringify(value: anytype, allocator: Allocator) ![]u8 {
        var string = ArrayList(u8).init(allocator);
        try std.json.stringify(value, .{}, string.writer());
        return string.toOwnedSlice();
    }
};

// Process utilities
pub const process = struct {
    pub fn getEnvVar(allocator: Allocator, key: []const u8) ?[]const u8 {
        return std.process.getEnvVarOwned(allocator, key) catch null;
    }
    
    pub fn hasEnvVar(key: []const u8) bool {
        return std.process.hasEnvVarConstant(key);
    }
    
    pub fn exit(code: u8) noreturn {
        std.process.exit(code);
    }
};

// Crypto utilities (basic)
pub const crypto = struct {
    pub const Random = std.rand.Random;
    pub const DefaultPrng = std.rand.DefaultPrng;
    
    pub fn randomBytes(allocator: Allocator, len: usize) ![]u8 {
        var prng = DefaultPrng.init(blk: {
            var seed: u64 = undefined;
            try std.os.getrandom(std.mem.asBytes(&seed));
            break :blk seed;
        });
        
        const bytes = try allocator.alloc(u8, len);
        prng.random().bytes(bytes);
        return bytes;
    }
    
    pub fn hash(data: []const u8) [32]u8 {
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        hasher.update(data);
        return hasher.finalResult();
    }
};

test "Platform detection works correctly" {
    const platform = Platform.current();
    try expect(platform != .other or builtin.os.tag == .freestanding);
}

test "String utilities work correctly" {
    try expect(string.eql("hello", "hello"));
    try expect(!string.eql("hello", "world"));
    try expect(string.startsWith("hello world", "hello"));
    try expect(string.endsWith("hello world", "world"));
    try expect(string.indexOf("hello world", "world") == 6);
    try expectEqualStrings("hello", string.trim("  hello  "));
}

test "Math utilities work correctly" {
    try expectEqual(@as(f32, 5.0), math.clamp(f32, 10.0, 0.0, 5.0));
    try expectEqual(@as(f32, 5.0), math.lerp(f32, 0.0, 10.0, 0.5));
}

test "Time utilities work correctly" {
    const ts1 = time.timestamp();
    const ts2 = time.nanoTimestamp();
    try expect(ts1 > 0);
    try expect(ts2 > 0);
}