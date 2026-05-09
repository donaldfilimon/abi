//! Public always-on services and gated service-like domains.

const build_options = @import("build_options");

/// Shared foundations: logging, security, time/SIMD primitives.
pub const foundation = @import("../foundation/mod.zig");
/// Runtime services: task scheduling, event loops, resource management.
pub const runtime = @import("../runtime/mod.zig");
/// Platform abstraction: OS detection, capabilities, environment.
pub const platform = @import("../platform/mod.zig");
/// External service connectors: HTTP clients, API adapters.
pub const connectors = if (build_options.feat_connectors) @import("../connectors/mod.zig") else @import("../connectors/stub.zig");
/// Shared CLI helpers for command dispatch and serve parsing.
pub const cli = @import("../cli.zig");
/// C-ABI FFI endpoints for linking as a static library (libabi.a).
pub const ffi = @import("../ffi.zig");
/// Task management: async job queues, scheduling, progress tracking.
pub const tasks = if (build_options.feat_tasks) @import("../tasks/mod.zig") else @import("../tasks/stub.zig");
/// ML inference: engine, scheduler, sampler, paged KV cache.
pub const inference = if (build_options.feat_inference) @import("../inference/mod.zig") else @import("../inference/stub.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
