//! Service module benchmarks
//!
//! Performance measurement for Phase 9+ feature modules:
//! cache, search, gateway, messaging, storage.

const std = @import("std");

pub const cache = @import("cache");
pub const search = @import("search");
pub const gateway = @import("gateway");
pub const messaging = @import("messaging");
pub const storage = @import("storage");

pub fn runAllBenchmarks(allocator: std.mem.Allocator) !void {
    try cache.run(allocator);
    try search.run(allocator);
    try gateway.run(allocator);
    try messaging.run(allocator);
    try storage.run(allocator);
}
