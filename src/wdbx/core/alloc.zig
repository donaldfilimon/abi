//! Allocator helpers and memory diagnostics.

const std = @import("std");

pub fn getDiagnosticsAllocator(base: std.mem.Allocator) std.mem.Allocator {
    // FIXME: implement real tracking and telemetry wrapping
    return base;
}
