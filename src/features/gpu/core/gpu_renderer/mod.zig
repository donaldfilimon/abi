// Re-export commonly used gpu_renderer internals. Explicit re-exports are
// preferred over `usingnamespace` to maintain clarity and compatibility.
pub const config = @import("config.zig");
pub const pipelines = @import("pipelines.zig");
pub const buffers = @import("buffers.zig");

pub const backends = @import("backends.zig");
pub const types = @import("types.zig");

pub const GPURenderer = @import("renderer.zig").GPURenderer;
