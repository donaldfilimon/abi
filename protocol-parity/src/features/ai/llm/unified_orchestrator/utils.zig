//! Utility helpers for the unified orchestrator (HTTP, JSON, process).

const std = @import("std");

// ── HTTP utilities ────────────────────────────────────────────────────────

pub fn httpGet(allocator: std.mem.Allocator, url: []const u8) ![]const u8 {
    _ = allocator;
    _ = url;
    return "";
}

// ── JSON utilities ────────────────────────────────────────────────────────

pub fn jsonParse(allocator: std.mem.Allocator, input: []const u8) !void {
    _ = allocator;
    _ = input;
}

pub fn jsonStringify(allocator: std.mem.Allocator) ![]const u8 {
    return allocator.dupe(u8, "{}");
}

// ── Process utilities ─────────────────────────────────────────────────────

pub fn runProcess(allocator: std.mem.Allocator, argv: []const []const u8) !std.ArrayListUnmanaged(u8) {
    _ = allocator;
    _ = argv;
    return .empty;
}

test {
    std.testing.refAllDecls(@This());
}
