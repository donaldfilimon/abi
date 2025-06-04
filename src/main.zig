//! zvim: Ultra-high-performance CLI with GPU acceleration, lock-free concurrency,
//! and platform-optimized implementations for Zig, Swift, and C++ development.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

const perf = @import("performance.zig");
const gpu = @import("../zvim/gpu_renderer.zig");
const simd = @import("../zvim/simd_text.zig");
const lockfree = @import("lockfree.zig");
const platform = @import("platform.zig");

pub fn main() !void {
    std.debug.print("zvim starting...\n", .{});
}
