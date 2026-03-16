//! Service module benchmarks
//!
//! Performance measurement for Phase 9+ feature modules:
//! cache, search, gateway, messaging, storage.

const std = @import("std");

pub const cache = @import("cache.zig");
pub const search = @import("search.zig");
pub const gateway = @import("gateway.zig");
pub const messaging = @import("messaging.zig");
pub const storage = @import("storage.zig");

pub fn runAllBenchmarks(allocator: std.mem.Allocator) !void {
    try cache.run(allocator);
    try search.run(allocator);
    try gateway.run(allocator);
    try messaging.run(allocator);
    try storage.run(allocator);
}
