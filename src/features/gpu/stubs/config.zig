const std = @import("std");
const backend_mod = @import("backend.zig");
const memory_mod = @import("memory.zig");

pub const GpuConfig = struct {
    backend: backend_mod.Backend = .vulkan,
    enable_profiling: bool = false,
    memory_mode: memory_mod.MemoryMode = .automatic,
};

test {
    std.testing.refAllDecls(@This());
}
