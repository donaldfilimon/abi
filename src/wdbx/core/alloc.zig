//! Allocator helpers and memory diagnostics.

const std = @import("std");

pub fn getDiagnosticsAllocator(base: std.mem.Allocator) std.mem.Allocator {
    _ = base;
    unreachable; // TODO: Implement diagnostics allocator
}
